"""Reproducible Apple Depth Pro 2 benchmark & efficiency analysis suite.

Measures inference latency, throughput, memory footprint, quantitative elevation
accuracy on a deterministic synthetic ground-truth topography, cold CLI vs warm
HTTP server latency, and video temporal stability on Apple Silicon MPS / CPU.

Usage:
    cd pipeline && uv run python -m depth_terrain.benchmark --output ../docs/benchmarks
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import av
import click
import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats

from depth_terrain import depth_runner as dr
from depth_terrain import heightmap_export as he
from depth_terrain.fictional_generator import _compute_hillshade, _fbm, _lerp_ramp

SCHEMA_VERSION = 1
DEFAULT_RESOLUTIONS = (384, 512, 768, 1024)
DEFAULT_PRECISIONS = ("bf16", "fp16", "fp32")
DEFAULT_MODELS = ("depthpro2-s", "depthpro2-b")
ALL_SECTIONS = ("precision", "resolution", "modes", "cold_vs_warm", "video", "charts", "report")


# ---------------------------------------------------------------------------
# Hardware & memory instrumentation
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _git_info() -> Dict[str, str]:
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return info


def collect_hardware_info() -> Dict[str, Any]:
    """Capture Apple Silicon / host hardware and software environment metadata."""
    mem_raw = _sysctl("hw.memsize")
    ram_gb = round(int(mem_raw) / (1024**3), 1) if mem_raw.isdigit() else None
    git = _git_info()
    return {
        "chip": _sysctl("machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor() or "unknown",
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.mac_ver()[0] if sys.platform == 'darwin' else platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
    }


def get_process_rss_mb() -> float:
    """Return peak process Resident Set Size (RSS) in MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 2)
    return round(rss / 1024, 2)


def measure_memory_mb(device_type: str = "mps") -> Dict[str, float]:
    """Sample process peak RSS and PyTorch MPS allocated / driver memory (MiB)."""
    mps_alloc = 0.0
    mps_driver = 0.0
    if device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        mps_alloc = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        mps_driver = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
    return {
        "rss_max_mb": get_process_rss_mb(),
        "mps_allocated_mb": mps_alloc,
        "mps_driver_mb": mps_driver,
    }


def _sync_and_clear_mps() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _clear_runner_cache() -> None:
    """Clear optional in-process model cache if pipeline-perf added one."""
    for attr in ("clear_model_cache", "_clear_model_cache"):
        fn = getattr(dr, attr, None)
        if callable(fn):
            fn()
    cache = getattr(dr, "_MODEL_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    _sync_and_clear_mps()


# ---------------------------------------------------------------------------
# Deterministic synthetic ground-truth topographic testbench
# ---------------------------------------------------------------------------


def generate_synthetic_topography(
    size: int = 768,
    seed: int = 2025,
    width: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic analytic topographic heightfield + sunlit top-down RGB.

    Combines known analytic landforms (volcanic stratocone with summit caldera,
    alpine ridge arc, carved winding river canyon, and coastal shoreline) with
    multi-octave Perlin FBM geological detail, rendered via directional NW
    hillshading, ambient occlusion, and hypsometric biome coloring.

    Args:
        size: Height in pixels (and width when ``width`` is omitted).
        seed: Deterministic RNG seed.
        width: Optional rectangular width in pixels (e.g. for panning video strips).

    Returns:
        ``(ground_truth_height, rgb_uint8)`` where ``ground_truth_height`` is
        float32 in ``[0, 1]`` of shape ``(H, W)`` and ``rgb_uint8`` is uint8 of
        shape ``(H, W, 3)``.
    """
    h = int(size)
    w = int(width) if width is not None else h
    rng = np.random.default_rng(seed)

    # Generate square FBM at max(h, w) then crop to preserve isotropic scale
    fbm_size = max(h, w)
    base = _fbm(fbm_size, rng, octaves=5, base_grid=3, persistence=0.50, ridged=False)[:h, :w]
    ridges = _fbm(fbm_size, rng, octaves=5, base_grid=4, persistence=0.52, ridged=True)[:h, :w]
    micro = _fbm(fbm_size, rng, octaves=4, base_grid=20, persistence=0.55, ridged=False)[:h, :w]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    radius = np.hypot(nx, ny)

    # Known analytic landforms
    island_mask = np.clip(1.0 - (radius / 0.92) ** 2.4, 0.0, 1.0)

    # 1. Volcanic stratocone with summit caldera
    r_volc = np.hypot(nx + 0.16, ny + 0.10)
    cone = 0.85 * np.exp(-((r_volc / 0.38) ** 2.0))
    caldera = 0.32 * np.exp(-((r_volc / 0.10) ** 2.4))

    # 2. Alpine ridge arc
    ridge_dist = np.abs((nx - 0.34) - 0.12 * np.sin(ny * 2.8))
    ridge = 0.58 * np.exp(-((ridge_dist / 0.16) ** 2.0)) * np.exp(-((ny + 0.05) / 0.65) ** 2.0)

    # 3. Winding carved river canyon
    canyon_x = 0.08 * np.sin(ny * 3.0) + 0.08
    canyon = 0.42 * np.exp(-(((nx - canyon_x) / 0.09) ** 2.0))

    elev = island_mask * (0.26 * base + 0.36 * ridges + ridge) + cone - caldera - canyon * 0.65
    elev = np.clip(elev, 0.0, None)
    elev = (elev - elev.min()) / max(float(elev.max() - elev.min()), 1e-6)
    gt = elev.astype(np.float32)

    # Sunlit top-down photorealistic satellite rendering
    light, slope = _compute_hillshade(gt, azimuth_deg=315.0, altitude_deg=32.0, z_factor=34.0)
    albedo = _lerp_ramp(
        gt,
        [
            (0.00, (14, 46, 80)),
            (0.14, (24, 92, 126)),
            (0.20, (52, 148, 160)),
            (0.24, (198, 182, 140)),
            (0.30, (54, 98, 48)),
            (0.48, (42, 78, 40)),
            (0.64, (98, 90, 82)),
            (0.80, (142, 134, 126)),
            (0.90, (206, 210, 218)),
            (1.00, (246, 248, 252)),
        ],
    )
    rock = np.clip((slope - 0.30) * 2.4, 0.0, 0.72)[..., None]
    albedo = albedo * (1.0 - rock) + np.array([104, 98, 92], dtype=np.float32) * rock
    water_blend = np.clip((0.22 - gt) / 0.05, 0.0, 1.0)
    light = light * (1.0 - water_blend) + (0.93 + 0.07 * micro) * water_blend
    grain = (0.93 + 0.14 * micro)[..., None]
    rgb = np.clip(albedo * light[..., None] * grain, 0.0, 255.0).astype(np.uint8)
    return gt, rgb


# ---------------------------------------------------------------------------
# Quantitative elevation accuracy & video temporal stability metrics
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if af.size < 2 or np.std(af) < 1e-12 or np.std(bf) < 1e-12:
        return 0.0
    val = float(stats.pearsonr(af, bf).statistic)
    return 0.0 if not np.isfinite(val) else val


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if af.size < 2 or np.std(af) < 1e-12 or np.std(bf) < 1e-12:
        return 0.0
    val = float(stats.spearmanr(af, bf).statistic)
    return 0.0 if not np.isfinite(val) else val


def compute_accuracy_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    edge_sigma: float = 2.0,
) -> Dict[str, float]:
    """Compute quantitative elevation accuracy metrics against ground-truth height.

    Metrics:
        - ``pearson_r``: Linear correlation between robust-normalized prediction & GT.
        - ``spearman_rho``: Monotonic rank correlation (invariant to gamma/monotone warp).
        - ``nrmse``: Affine-aligned Normalized RMSE in ``[0, 1]`` (lower is better; 0.0 = exact).
        - ``sobel_edge_alignment``: Pearson correlation of interior Sobel gradient magnitudes.
        - ``composite_quality``: Weighted summary score in ``[0, 1]`` for Pareto comparison.
    """
    p_raw = np.asarray(pred, dtype=np.float64)
    p_finite = p_raw[np.isfinite(p_raw)]
    p_fill = float(np.median(p_finite)) if p_finite.size > 0 else 0.0
    p = np.where(np.isfinite(p_raw), p_raw, p_fill)

    g_raw = np.asarray(gt, dtype=np.float64)
    g_finite = g_raw[np.isfinite(g_raw)]
    g_fill = float(np.median(g_finite)) if g_finite.size > 0 else 0.0
    g = np.where(np.isfinite(g_raw), g_raw, g_fill)
    if p.shape != g.shape:
        p_pil = Image.fromarray(p.astype(np.float32)).resize((g.shape[1], g.shape[0]), Image.Resampling.BILINEAR)
        p = np.asarray(p_pil, dtype=np.float64)

    g_span = float(g.max() - g.min())
    g_norm = (g - g.min()) / g_span if g_span > 1e-12 else np.zeros_like(g)

    p1, p99 = float(np.percentile(p, 1.0)), float(np.percentile(p, 99.0))
    if p99 - p1 < 1e-12:
        p1, p99 = float(p.min()), float(p.max())
    p_norm = np.clip((p - p1) / (p99 - p1), 0.0, 1.0) if (p99 - p1) > 1e-12 else np.zeros_like(p)

    pearson_r = _safe_pearson(p_norm, g_norm)
    step = max(1, min(p_norm.shape) // 192)
    spearman_rho = _safe_spearman(p_norm[::step, ::step], g_norm[::step, ::step])

    # Affine-aligned Normalized RMSE (handles scale/shift invariance of monocular depth)
    if np.std(p_norm) > 1e-12:
        slope, intercept = np.polyfit(p_norm.ravel(), g_norm.ravel(), 1)
        p_aligned = np.clip(slope * p_norm + intercept, 0.0, 1.0)
    else:
        p_aligned = np.full_like(g_norm, float(np.mean(g_norm)))
    nrmse = float(np.sqrt(np.mean((p_aligned - g_norm) ** 2)))

    # Sobel edge alignment on interior crop to avoid border clipping artifacts
    ps = ndimage.gaussian_filter(p_norm, sigma=edge_sigma) if edge_sigma > 0 else p_norm
    gs = ndimage.gaussian_filter(g_norm, sigma=edge_sigma) if edge_sigma > 0 else g_norm
    margin = min(12, max(1, min(p_norm.shape) // 16))
    mag_p = np.hypot(ndimage.sobel(ps, axis=1), ndimage.sobel(ps, axis=0))[margin:-margin, margin:-margin]
    mag_g = np.hypot(ndimage.sobel(gs, axis=1), ndimage.sobel(gs, axis=0))[margin:-margin, margin:-margin]
    sobel_corr = _safe_pearson(mag_p, mag_g)

    composite = (
        0.40 * max(0.0, pearson_r)
        + 0.30 * max(0.0, spearman_rho)
        + 0.15 * max(0.0, sobel_corr)
        + 0.15 * max(0.0, 1.0 - nrmse)
    )
    return {
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "nrmse": round(nrmse, 4),
        "sobel_edge_alignment": round(sobel_corr, 4),
        "composite_quality": round(composite, 4),
    }


def compute_temporal_stability_metrics(
    frames_depth: Sequence[np.ndarray],
    pan_dx_px: int = 16,
) -> Dict[str, float]:
    """Measure frame-to-frame depth jitter across a synthetic horizontal camera pan.

    Aligns the known overlapping spatial region between consecutive frames
    ``t`` and ``t+1`` (shifted by ``pan_dx_px`` columns) so camera motion is
    factored out and pure temporal depth flicker is isolated.
    """
    if len(frames_depth) < 2:
        return {
            "num_frames": len(frames_depth),
            "overlap_mad": 0.0,
            "overlap_rmse": 0.0,
            "overlap_pearson_r": 1.0,
            "frame_mean_std": 0.0,
        }

    norm_frames: List[np.ndarray] = []
    frame_means: List[float] = []
    for f in frames_depth:
        arr = np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
        if hi - lo < 1e-12:
            norm = np.zeros_like(arr)
        else:
            norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        norm_frames.append(norm)
        frame_means.append(float(np.mean(norm)))

    mads: List[float] = []
    rmses: List[float] = []
    corrs: List[float] = []
    for prev_f, curr_f in zip(norm_frames[:-1], norm_frames[1:]):
        w = min(prev_f.shape[1], curr_f.shape[1])
        h = min(prev_f.shape[0], curr_f.shape[0])
        dx = max(0, min(int(pan_dx_px), w - 4))
        if dx > 0:
            a = prev_f[:h, dx:w]
            b = curr_f[:h, : w - dx]
        else:
            a = prev_f[:h, :w]
            b = curr_f[:h, :w]
        diff = a - b
        mads.append(float(np.mean(np.abs(diff))))
        rmses.append(float(np.sqrt(np.mean(diff * diff))))
        corrs.append(_safe_pearson(a, b))

    return {
        "num_frames": len(frames_depth),
        "overlap_mad": round(float(np.mean(mads)), 5),
        "overlap_rmse": round(float(np.mean(rmses)), 5),
        "overlap_pearson_r": round(float(np.mean(corrs)), 4),
        "frame_mean_std": round(float(np.std(frame_means)), 5),
    }


def generate_synthetic_pan_video(
    output_path: Path,
    num_frames: int = 12,
    frame_size: int = 512,
    pan_dx_px: int = 16,
    fps: int = 12,
    seed: int = 2025,
) -> Tuple[Path, List[np.ndarray]]:
    """Create a deterministic synthetic aerial pan MP4 video for temporal stability testing."""
    strip_w = frame_size + max(0, num_frames - 1) * pan_dx_px
    gt_strip, rgb_strip = generate_synthetic_topography(size=frame_size, width=strip_w, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    gt_frames: List[np.ndarray] = []
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = frame_size
        stream.height = frame_size
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 6_000_000

        for idx in range(num_frames):
            x0 = idx * pan_dx_px
            rgb_crop = np.ascontiguousarray(rgb_strip[:, x0 : x0 + frame_size])
            gt_crop = gt_strip[:, x0 : x0 + frame_size].copy()
            gt_frames.append(gt_crop)
            vf = av.VideoFrame.from_ndarray(rgb_crop, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    finally:
        container.close()

    return output_path, gt_frames


# ---------------------------------------------------------------------------
# Guided-filter helper detection (adapts to pipeline-perf's implementation)
# ---------------------------------------------------------------------------


def _apply_optional_guided_filter(
    inv_depth: np.ndarray,
    depth_m: np.ndarray,
    albedo_rgb: np.ndarray,
) -> Optional[Tuple[np.ndarray, float]]:
    """Apply heightmap_export's RGB-guided filter if available, returning (filtered, ms)."""
    # 1. Check direct guided filter function in heightmap_export
    for fn_name in ("guided_filter_heightmap", "apply_guided_filter", "guided_filter_gray", "guided_filter"):
        fn = getattr(he, fn_name, None)
        if callable(fn):
            p = np.nan_to_num(inv_depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            p_lo, p_hi = float(np.percentile(p, 2.0)), float(np.percentile(p, 98.0))
            norm = np.clip((p - p_lo) / max(p_hi - p_lo, 1e-8), 0.0, 1.0).astype(np.float32)
            t0 = time.perf_counter()
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            if len(params) >= 2 and ("guide" in params[0].lower() or "albedo" in params[0].lower() or "rgb" in params[0].lower()):
                out = fn(albedo_rgb, norm)
            else:
                out = fn(norm, albedo_rgb)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return np.asarray(out, dtype=np.float32), elapsed_ms

    # 2. Check ExportConfig.guided_filter via depth_to_normalized_height
    cfg_fields = getattr(he.ExportConfig, "__dataclass_fields__", {})
    if "guided_filter" in cfg_fields:
        cfg = he.ExportConfig(smooth_sigma=0.0, guided_filter=True)
        d_sig = inspect.signature(he.depth_to_normalized_height)
        t0 = time.perf_counter()
        if "guide_rgb" in d_sig.parameters:
            norm, _ = he.depth_to_normalized_height(depth_m, inv_depth, cfg, guide_rgb=albedo_rgb)
        elif "albedo_rgb" in d_sig.parameters:
            norm, _ = he.depth_to_normalized_height(depth_m, inv_depth, cfg, albedo_rgb=albedo_rgb)
        else:
            norm, _ = he.depth_to_normalized_height(depth_m, inv_depth, cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(norm, dtype=np.float32), elapsed_ms

    return None


def _load_model_fresh(model_name: str, device_name: str, precision_name: Optional[str]) -> Tuple[Any, float]:
    """Load a Depth Pro 2 bundle cleanly and return (bundle, load_ms)."""
    _clear_runner_cache()
    t0 = time.perf_counter()
    bundle = dr.load_depth_pro_2(
        model_name=model_name,
        device_name=device_name,
        precision_name=precision_name,
    )
    if device_name == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000.0
    return bundle, load_ms


def _timed_infer(
    image_rgb: np.ndarray,
    model_name: str,
    topdown_mode: str,
    loaded_bundle: Tuple[Any, Any, torch.device],
    repeats: int = 1,
) -> Tuple[dr.DepthInferenceResult, float, List[float]]:
    """Run ``repeats`` timed inference passes and return (last_result, median_wall_ms, all_ms)."""
    _, _, device = loaded_bundle
    timings: List[float] = []
    last_res: Optional[dr.DepthInferenceResult] = None
    for _ in range(max(1, repeats)):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        last_res = dr.infer_image_depth(
            image_rgb,
            model_name=model_name,
            topdown_mode=topdown_mode,
            loaded_bundle=loaded_bundle,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)
    assert last_res is not None
    return last_res, float(np.median(timings)), [round(t, 2) for t in timings]


# ---------------------------------------------------------------------------
# Benchmark Section A: Model & Precision Matrix (+ CPU reference)
# ---------------------------------------------------------------------------


def run_precision_matrix(
    resolution: int = 768,
    models: Sequence[str] = DEFAULT_MODELS,
    precisions: Sequence[str] = DEFAULT_PRECISIONS,
    include_cpu_reference: bool = True,
    repeats: int = 2,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Benchmark models across bf16/fp16/fp32 on MPS plus a CPU fp32 reference point."""
    gt, img_rgb = generate_synthetic_topography(resolution)
    mp = (resolution * resolution) / 1e6
    rows: List[Dict[str, Any]] = []

    configs: List[Tuple[str, str, str]] = []
    if torch.backends.mps.is_available():
        for m in models:
            for p in precisions:
                configs.append((m, "mps", p))
    if include_cpu_reference:
        configs.append((models[0], "cpu", "fp32"))

    for model_name, device_name, prec in configs:
        try:
            bundle, load_ms = _load_model_fresh(model_name, device_name, prec)
            # Warmup
            _ = dr.infer_image_depth(img_rgb, model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)
            eff_repeats = 1 if device_name == "cpu" else repeats
            res, wall_ms, runs_ms = _timed_infer(img_rgb, model_name, "raw", bundle, repeats=eff_repeats)
            mem = measure_memory_mb(device_name)
            acc = compute_accuracy_metrics(res.inverse_depth, gt)
            row = {
                "model": model_name,
                "device": device_name,
                "precision": prec,
                "resolved_precision": res.precision,
                "resolution": resolution,
                "mode": "raw",
                "load_ms": round(load_ms, 2),
                "inference_ms": round(res.inference_ms, 2),
                "wall_ms": round(wall_ms, 2),
                "runs_ms": runs_ms,
                "ms_per_mp": round(wall_ms / mp, 2),
                "mp_per_sec": round((mp * 1000.0) / max(wall_ms, 1e-6), 3),
                "fps": round(1000.0 / max(wall_ms, 1e-6), 3),
                **mem,
                "accuracy": acc,
            }
            rows.append(row)
            log(
                f"[PRECISION] {model_name:11s} {device_name:3s} {prec:4s}: "
                f"{wall_ms:7.1f} ms ({row['fps']:5.2f} fps) | r={acc['pearson_r']:+.4f} | mps={mem['mps_allocated_mb']:.0f}MB"
            )
            del bundle
            _sync_and_clear_mps()
        except Exception as exc:
            log(f"[PRECISION] {model_name} {device_name} {prec} FAILED: {exc}")
            rows.append(
                {
                    "model": model_name,
                    "device": device_name,
                    "precision": prec,
                    "resolution": resolution,
                    "mode": "raw",
                    "error": str(exc),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Benchmark Section B: Resolution Scaling
# ---------------------------------------------------------------------------


def run_resolution_scaling(
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    models: Sequence[str] = DEFAULT_MODELS,
    precision: str = "bf16",
    repeats: int = 2,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Measure latency, throughput (MP/s & fps), memory, and accuracy across resolutions."""
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    eff_prec = precision if device_name == "mps" else "fp32"
    rows: List[Dict[str, Any]] = []

    datasets = {r: generate_synthetic_topography(r) for r in resolutions}

    for model_name in models:
        bundle, load_ms = _load_model_fresh(model_name, device_name, eff_prec)
        # Warmup at smallest resolution
        _ = dr.infer_image_depth(datasets[resolutions[0]][1], model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)

        for res_px in resolutions:
            gt, img_rgb = datasets[res_px]
            mp = (res_px * res_px) / 1e6
            # Warmup for this resolution shape
            _ = dr.infer_image_depth(img_rgb, model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)
            res, wall_ms, runs_ms = _timed_infer(img_rgb, model_name, "raw", bundle, repeats=repeats)
            mem = measure_memory_mb(device_name)
            acc = compute_accuracy_metrics(res.inverse_depth, gt)
            row = {
                "model": model_name,
                "device": device_name,
                "precision": eff_prec,
                "resolution": res_px,
                "megapixels": round(mp, 4),
                "load_ms": round(load_ms, 2),
                "inference_ms": round(res.inference_ms, 2),
                "wall_ms": round(wall_ms, 2),
                "runs_ms": runs_ms,
                "ms_per_mp": round(wall_ms / mp, 2),
                "mp_per_sec": round((mp * 1000.0) / max(wall_ms, 1e-6), 3),
                "fps": round(1000.0 / max(wall_ms, 1e-6), 3),
                **mem,
                "accuracy": acc,
            }
            rows.append(row)
            log(
                f"[RESOLUTION] {model_name:11s} {res_px:4d}px ({mp:.2f}MP): "
                f"{wall_ms:7.1f} ms | {row['mp_per_sec']:5.2f} MP/s | {row['fps']:5.2f} fps | "
                f"mps={mem['mps_allocated_mb']:.0f}MB (drv={mem['mps_driver_mb']:.0f}MB)"
            )
        del bundle
        _sync_and_clear_mps()
    return rows


# ---------------------------------------------------------------------------
# Benchmark Section C: Top-Down Modes (Speed vs Quality + Guided Filter)
# ---------------------------------------------------------------------------


def run_topdown_modes(
    resolution: int = 768,
    models: Sequence[str] = DEFAULT_MODELS,
    precision: str = "bf16",
    baseline: Optional[Dict[str, Any]] = None,
    repeats: int = 2,
    log: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Compare raw, tta4, fast_multiscale, multiscale_tta (with & without guided filter)."""
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    eff_prec = precision if device_name == "mps" else "fp32"
    gt, img_rgb = generate_synthetic_topography(resolution)
    mp = (resolution * resolution) / 1e6

    # Order modes from fastest to heaviest
    preferred_order = ("raw", "tta4", "fast_multiscale", "multiscale_tta")
    available_modes = [m for m in preferred_order if m in dr.TOPDOWN_MODES]
    for m in dr.TOPDOWN_MODES:
        if m not in available_modes:
            available_modes.append(m)

    baseline_lookup: Dict[Tuple[str, str], float] = {}
    if baseline and isinstance(baseline.get("runs"), list):
        for b_run in baseline["runs"]:
            if b_run.get("resolution") == resolution:
                baseline_lookup[(b_run["model"], b_run["mode"])] = float(b_run["wall_ms"])

    rows: List[Dict[str, Any]] = []
    for model_name in models:
        bundle, load_ms = _load_model_fresh(model_name, device_name, eff_prec)
        # Warmup
        _ = dr.infer_image_depth(img_rgb, model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)

        for mode in available_modes:
            eff_repeats = 1 if mode == "multiscale_tta" else repeats
            res, wall_ms, runs_ms = _timed_infer(img_rgb, model_name, mode, bundle, repeats=eff_repeats)
            mem = measure_memory_mb(device_name)
            acc = compute_accuracy_metrics(res.inverse_depth, gt)

            base_ms = baseline_lookup.get((model_name, mode))
            speedup = round(base_ms / wall_ms, 2) if (base_ms and wall_ms > 0) else None
            quality_per_sec = round(acc["composite_quality"] / max(wall_ms / 1000.0, 1e-6), 4)

            row_plain: Dict[str, Any] = {
                "model": model_name,
                "device": device_name,
                "precision": eff_prec,
                "resolution": resolution,
                "mode": mode,
                "guided_filter": False,
                "load_ms": round(load_ms, 2),
                "inference_ms": round(res.inference_ms, 2),
                "guided_filter_ms": 0.0,
                "wall_ms": round(wall_ms, 2),
                "runs_ms": runs_ms,
                "ms_per_mp": round(wall_ms / mp, 2),
                "fps": round(1000.0 / max(wall_ms, 1e-6), 3),
                "baseline_wall_ms": round(base_ms, 2) if base_ms else None,
                "speedup_vs_baseline": speedup,
                "quality_per_sec": quality_per_sec,
                **mem,
                "accuracy": acc,
            }
            rows.append(row_plain)
            spd_str = f" ({speedup:.2f}x vs baseline)" if speedup else ""
            log(
                f"[MODES] {model_name:11s} {mode:15s} gf=off: {wall_ms:7.1f} ms{spd_str} | "
                f"r={acc['pearson_r']:+.4f} rho={acc['spearman_rho']:+.4f} sobel={acc['sobel_edge_alignment']:+.4f}"
            )

            # Evaluate with RGB-guided edge-preserving filter if available
            gf_res = _apply_optional_guided_filter(res.inverse_depth, res.depth_m, img_rgb)
            if gf_res is not None:
                filtered_map, gf_ms = gf_res
                total_ms = wall_ms + gf_ms
                acc_gf = compute_accuracy_metrics(filtered_map, gt)
                row_gf: Dict[str, Any] = {
                    "model": model_name,
                    "device": device_name,
                    "precision": eff_prec,
                    "resolution": resolution,
                    "mode": mode,
                    "guided_filter": True,
                    "load_ms": round(load_ms, 2),
                    "inference_ms": round(res.inference_ms, 2),
                    "guided_filter_ms": round(gf_ms, 2),
                    "wall_ms": round(total_ms, 2),
                    "runs_ms": runs_ms,
                    "ms_per_mp": round(total_ms / mp, 2),
                    "fps": round(1000.0 / max(total_ms, 1e-6), 3),
                    "baseline_wall_ms": round(base_ms, 2) if base_ms else None,
                    "speedup_vs_baseline": round(base_ms / total_ms, 2) if (base_ms and total_ms > 0) else None,
                    "quality_per_sec": round(acc_gf["composite_quality"] / max(total_ms / 1000.0, 1e-6), 4),
                    **mem,
                    "accuracy": acc_gf,
                }
                rows.append(row_gf)
                log(
                    f"[MODES] {model_name:11s} {mode:15s} gf=on : {total_ms:7.1f} ms (+{gf_ms:.1f}ms) | "
                    f"r={acc_gf['pearson_r']:+.4f} rho={acc_gf['spearman_rho']:+.4f} sobel={acc_gf['sobel_edge_alignment']:+.4f}"
                )

        del bundle
        _sync_and_clear_mps()
    return rows


# ---------------------------------------------------------------------------
# Benchmark Section D: Cold CLI vs Warm HTTP Server Latency
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120.0) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return int(resp.status), json.loads(body)


def run_cold_vs_warm_server(
    resolution: int = 768,
    models: Sequence[str] = DEFAULT_MODELS,
    warm_requests: int = 3,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Compare cold ``depth-terrain from-image`` subprocess vs warm ``depth-terrain serve`` HTTP."""
    pipeline_dir = Path(__file__).resolve().parents[2]
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    topdown_mode = "fast_multiscale" if "fast_multiscale" in dr.TOPDOWN_MODES else "tta4"
    _, img_rgb = generate_synthetic_topography(resolution)

    comparisons: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="depth_bench_srv_") as tmp:
        tmp_path = Path(tmp)
        input_png = tmp_path / "bench_input.png"
        Image.fromarray(img_rgb).save(input_png)

        for model_name in models:
            # 1. Cold CLI subprocess measurement
            cold_out = tmp_path / f"cold_{model_name}"
            cmd = [
                sys.executable,
                "-m",
                "depth_terrain.cli",
                "from-image",
                "-i",
                str(input_png),
                "-m",
                model_name,
                "--device",
                device_name,
                "--topdown-mode",
                topdown_mode,
                "-o",
                str(cold_out),
            ]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=str(pipeline_dir), capture_output=True, text=True, check=False)
            cold_wall_ms = (time.perf_counter() - t0) * 1000.0
            cold_infer_ms: Optional[float] = None
            if proc.returncode == 0:
                meta_file = cold_out / "metadata.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    cold_infer_ms = float(meta.get("inferenceMs", 0.0))
            else:
                log(f"[COLD CLI] {model_name} failed: {proc.stderr[-300:]}")

            # 2. Warm HTTP Server measurement (spawn `depth_terrain.cli serve`)
            port = _find_free_port()
            srv_cmd = [
                sys.executable,
                "-m",
                "depth_terrain.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                model_name,
                "--device",
                device_name,
            ]
            srv_proc = subprocess.Popen(
                srv_cmd,
                cwd=str(pipeline_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            warm_latencies_ms: List[float] = []
            warm_infer_ms: List[float] = []
            server_startup_ms: Optional[float] = None
            server_error: Optional[str] = None
            base_url = f"http://127.0.0.1:{port}"
            try:
                t_start = time.perf_counter()
                deadline = t_start + 60.0
                ready = False
                while time.perf_counter() < deadline:
                    if srv_proc.poll() is not None:
                        _, err = srv_proc.communicate(timeout=2)
                        server_error = f"Server exited early ({srv_proc.returncode}): {err[-300:]}"
                        break
                    try:
                        code, health = _http_json(f"{base_url}/health", timeout=2.0)
                        if code == 200 and health.get("status") == "ok":
                            server_startup_ms = (time.perf_counter() - t_start) * 1000.0
                            ready = True
                            break
                    except Exception:
                        time.sleep(0.15)

                if ready:
                    # 1 warmup HTTP inference
                    warm_body = {
                        "inputPath": str(input_png),
                        "outputDir": str(tmp_path / f"warm_{model_name}_warmup"),
                        "model": model_name,
                        "device": device_name,
                        "topDownMode": topdown_mode,
                        "resolution": 513,
                    }
                    _http_json(f"{base_url}/infer/image", payload=warm_body, timeout=120.0)

                    # Timed warm HTTP requests
                    for idx in range(max(1, warm_requests)):
                        req_body = dict(warm_body, outputDir=str(tmp_path / f"warm_{model_name}_{idx}"))
                        t_req = time.perf_counter()
                        code, resp = _http_json(f"{base_url}/infer/image", payload=req_body, timeout=120.0)
                        req_ms = (time.perf_counter() - t_req) * 1000.0
                        if code == 200 and resp.get("status") == "ok":
                            warm_latencies_ms.append(round(req_ms, 2))
                            meta = resp.get("metadata") or {}
                            if "inferenceMs" in meta:
                                warm_infer_ms.append(round(float(meta["inferenceMs"]), 2))
                elif not server_error:
                    server_error = "Timed out waiting for /health"
            except Exception as exc:
                server_error = str(exc)
            finally:
                srv_proc.terminate()
                try:
                    srv_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    srv_proc.kill()

            warm_median_ms = round(float(np.median(warm_latencies_ms)), 2) if warm_latencies_ms else None
            speedup = round(cold_wall_ms / warm_median_ms, 2) if (warm_median_ms and warm_median_ms > 0) else None
            saved_ms = round(cold_wall_ms - warm_median_ms, 2) if warm_median_ms else None

            entry = {
                "model": model_name,
                "device": device_name,
                "resolution": resolution,
                "topdown_mode": topdown_mode,
                "cold_cli_wall_ms": round(cold_wall_ms, 2),
                "cold_cli_inference_ms": round(cold_infer_ms, 2) if cold_infer_ms is not None else None,
                "cold_overhead_ms": round(cold_wall_ms - cold_infer_ms, 2) if cold_infer_ms is not None else None,
                "server_startup_ms": round(server_startup_ms, 2) if server_startup_ms is not None else None,
                "warm_http_runs_ms": warm_latencies_ms,
                "warm_http_median_ms": warm_median_ms,
                "warm_http_inference_median_ms": round(float(np.median(warm_infer_ms)), 2) if warm_infer_ms else None,
                "warm_speedup_vs_cold": speedup,
                "latency_saved_ms": saved_ms,
                "server_error": server_error,
            }
            comparisons.append(entry)
            log(
                f"[COLD vs WARM] {model_name:11s} ({topdown_mode}): "
                f"cold={cold_wall_ms:.1f}ms | warm={warm_median_ms}ms | speedup={speedup}x"
                + (f" | err={server_error}" if server_error else "")
            )

    return {
        "resolution": resolution,
        "topdown_mode": topdown_mode,
        "warm_requests": warm_requests,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Benchmark Section E: Video Temporal Stability
# ---------------------------------------------------------------------------


def run_video_stability(
    model_name: str = "depthpro2-s",
    num_frames: int = 12,
    frame_size: int = 512,
    pan_dx_px: int = 16,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Compare buffered video vs unbuffered video vs frame-by-frame single-image jitter."""
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    strategies: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="depth_bench_vid_") as tmp:
        video_path = Path(tmp) / "synthetic_pan.mp4"
        _, gt_frames = generate_synthetic_pan_video(
            video_path, num_frames=num_frames, frame_size=frame_size, pan_dx_px=pan_dx_px
        )

        # Decode exact frames for single-image comparison so compression artifacts match identically
        decoded_rgbs: List[np.ndarray] = []
        container = av.open(str(video_path))
        try:
            for frame in container.decode(video=0):
                decoded_rgbs.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()

        # 1 & 2. infer_video_depth (buffered vs unbuffered)
        for vmode, overlap in (("buffered", 2), ("unbuffered", 0)):
            _clear_runner_cache()
            # Pre-load model so we measure pure video inference + temporal consistency
            t0 = time.perf_counter()
            vid_results = list(
                dr.infer_video_depth(
                    video_path=video_path,
                    model_name=model_name,
                    device_name=device_name,
                    mode=vmode,
                    window=8,
                    overlap=overlap,
                    max_frames=num_frames,
                    frame_stride=1,
                    max_side=frame_size,
                    detrend_frames=True,
                )
            )
            if device_name == "mps":
                torch.mps.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            invs = [r.inverse_depth for _, _, r in vid_results]
            per_frame_infer_ms = float(vid_results[0][2].inference_ms) if vid_results else 0.0
            stab = compute_temporal_stability_metrics(invs, pan_dx_px=pan_dx_px)
            mid_acc = compute_accuracy_metrics(invs[len(invs) // 2], gt_frames[len(invs) // 2])
            row = {
                "strategy": f"infer_video_{vmode}",
                "model": model_name,
                "device": device_name,
                "num_frames": len(invs),
                "frame_size": frame_size,
                "pan_dx_px": pan_dx_px,
                "total_wall_ms": round(wall_ms, 2),
                "per_frame_ms": round(per_frame_infer_ms, 2),
                "fps": round(1000.0 / max(per_frame_infer_ms, 1e-6), 2),
                "stability": stab,
                "mid_frame_accuracy": mid_acc,
            }
            strategies.append(row)
            log(
                f"[VIDEO] {row['strategy']:22s}: {per_frame_infer_ms:6.1f} ms/frame ({row['fps']:5.2f} fps) | "
                f"jitter MAD={stab['overlap_mad']:.5f} | RMSE={stab['overlap_rmse']:.5f} | r={stab['overlap_pearson_r']:.4f}"
            )

        # 3. Single-frame raw inference loop (with detrending to compare apples-to-apples)
        bundle, _ = _load_model_fresh(model_name, device_name, "bf16" if device_name == "mps" else "fp32")
        _ = dr.infer_image_depth(decoded_rgbs[0], model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)
        if device_name == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        single_invs: List[np.ndarray] = []
        for rgb in decoded_rgbs[:num_frames]:
            res = dr.infer_image_depth(rgb, model_name=model_name, topdown_mode="raw", loaded_bundle=bundle)
            detr = dr.debias_positive_radial_vignette(dr.detrend_linear_plane(res.inverse_depth))
            single_invs.append(detr)
        if device_name == "mps":
            torch.mps.synchronize()
        total_single_ms = (time.perf_counter() - t0) * 1000.0
        per_frame_single_ms = total_single_ms / max(len(single_invs), 1)
        stab_single = compute_temporal_stability_metrics(single_invs, pan_dx_px=pan_dx_px)
        mid_acc_single = compute_accuracy_metrics(single_invs[len(single_invs) // 2], gt_frames[len(single_invs) // 2])
        row_single = {
            "strategy": "single_image_raw",
            "model": model_name,
            "device": device_name,
            "num_frames": len(single_invs),
            "frame_size": frame_size,
            "pan_dx_px": pan_dx_px,
            "total_wall_ms": round(total_single_ms, 2),
            "per_frame_ms": round(per_frame_single_ms, 2),
            "fps": round(1000.0 / max(per_frame_single_ms, 1e-6), 2),
            "stability": stab_single,
            "mid_frame_accuracy": mid_acc_single,
        }
        strategies.append(row_single)
        log(
            f"[VIDEO] {row_single['strategy']:22s}: {per_frame_single_ms:6.1f} ms/frame ({row_single['fps']:5.2f} fps) | "
            f"jitter MAD={stab_single['overlap_mad']:.5f} | RMSE={stab_single['overlap_rmse']:.5f} | r={stab_single['overlap_pearson_r']:.4f}"
        )
        del bundle
        _sync_and_clear_mps()

    return {
        "model": model_name,
        "num_frames": num_frames,
        "frame_size": frame_size,
        "pan_dx_px": pan_dx_px,
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# Schema validation (tested in test_benchmark.py)
# ---------------------------------------------------------------------------


def validate_benchmark_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate structure and required keys of a benchmark results JSON payload."""
    errors: List[str] = []
    required_top = ("schema_version", "captured_at", "hardware", "precision_matrix", "resolution_scaling", "topdown_modes")
    for key in required_top:
        if key not in payload:
            errors.append(f"Missing required top-level key: '{key}'")

    hw = payload.get("hardware")
    if isinstance(hw, dict):
        for hw_key in ("chip", "cpu_cores", "os", "python", "torch", "mps_available"):
            if hw_key not in hw:
                errors.append(f"Missing hardware key: '{hw_key}'")
    else:
        errors.append("'hardware' must be a dict")

    for section_name in ("precision_matrix", "resolution_scaling", "topdown_modes"):
        rows = payload.get(section_name)
        if not isinstance(rows, list):
            errors.append(f"'{section_name}' must be a list")
            continue
        for idx, row in enumerate(rows):
            if "error" in row:
                continue
            for field in ("model", "device", "precision", "resolution", "wall_ms", "accuracy"):
                if field not in row:
                    errors.append(f"{section_name}[{idx}] missing '{field}'")
            acc = row.get("accuracy")
            if isinstance(acc, dict):
                for m_key in ("pearson_r", "spearman_rho", "nrmse", "sobel_edge_alignment"):
                    if m_key not in acc:
                        errors.append(f"{section_name}[{idx}].accuracy missing '{m_key}'")

    return errors


# ---------------------------------------------------------------------------
# Publication-grade Matplotlib chart generation
# ---------------------------------------------------------------------------


def generate_charts(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate crisp PNG benchmark charts in ``output_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e2e8f0",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f1f5f9",
            "grid.color": "#334155",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )

    generated: List[Path] = []
    chip = results.get("hardware", {}).get("chip", "Apple Silicon")

    # ------------------------------------------------------------------
    # Chart 1: Before vs After Optimization Latency & Speedup
    # ------------------------------------------------------------------
    mode_rows = [r for r in results.get("topdown_modes", []) if not r.get("guided_filter") and "error" not in r]
    if mode_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200)
        fig.suptitle(f"Apple Depth Pro 2 Optimization — Before vs After Latency ({chip} MPS, 768x768 bf16)", fontsize=13.5, fontweight="bold", y=0.98)

        for ax, model_name in zip(axes, ("depthpro2-s", "depthpro2-b")):
            m_rows = [r for r in mode_rows if r["model"] == model_name]
            modes = [r["mode"] for r in m_rows]
            after_vals = [r["wall_ms"] for r in m_rows]
            before_vals = [r["baseline_wall_ms"] or 0.0 for r in m_rows]
            x = np.arange(len(modes))
            w = 0.36

            bars_before = ax.bar(x - w / 2, before_vals, width=w, label="Before (Baseline)", color="#64748b", edgecolor="#94a3b8")
            bars_after = ax.bar(x + w / 2, after_vals, width=w, label="After (Optimized)", color="#38bdf8", edgecolor="#7dd3fc")

            for i, (b_val, a_val, row) in enumerate(zip(before_vals, after_vals, m_rows)):
                if b_val > 0:
                    ax.text(x[i] - w / 2, b_val + max(after_vals + before_vals) * 0.015, f"{b_val:.0f}ms", ha="center", va="bottom", fontsize=8.5, color="#cbd5e1")
                else:
                    ax.text(x[i] - w / 2, max(after_vals + before_vals) * 0.015, "NEW", ha="center", va="bottom", fontsize=8, color="#fbbf24", fontweight="bold")
                spd = row.get("speedup_vs_baseline")
                lbl = f"{a_val:.0f}ms\n({spd:.2f}x)" if spd else f"{a_val:.0f}ms"
                ax.text(x[i] + w / 2, a_val + max(after_vals + before_vals) * 0.015, lbl, ha="center", va="bottom", fontsize=8.5, color="#38bdf8", fontweight="bold")

            ax.set_title(f"{model_name}")
            ax.set_xticks(x)
            ax.set_xticklabels(modes, rotation=12)
            ax.set_ylabel("Wall-Clock Latency (ms) — lower is better")
            ax.set_ylim(0, max(after_vals + before_vals + [100.0]) * 1.22)
            ax.grid(axis="y")
            ax.legend(framealpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        p1 = output_dir / "latency_before_after.png"
        fig.savefig(p1, dpi=200)
        plt.close(fig)
        generated.append(p1)

    # ------------------------------------------------------------------
    # Chart 2: Resolution Scaling + Precision/Device Matrix
    # ------------------------------------------------------------------
    res_rows = [r for r in results.get("resolution_scaling", []) if "error" not in r]
    prec_rows = [r for r in results.get("precision_matrix", []) if "error" not in r]
    if res_rows or prec_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200)
        fig.suptitle(f"Resolution Scaling & Precision / MPS vs CPU Speedup ({chip})", fontsize=13.5, fontweight="bold", y=0.98)

        # Left: Resolution scaling
        ax = axes[0]
        if res_rows:
            colors = {"depthpro2-s": "#34d399", "depthpro2-b": "#38bdf8"}
            for model_name in ("depthpro2-s", "depthpro2-b"):
                m_rows = sorted([r for r in res_rows if r["model"] == model_name], key=lambda r: r["resolution"])
                if not m_rows:
                    continue
                xs = [r["resolution"] for r in m_rows]
                ys = [r["wall_ms"] for r in m_rows]
                ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=7, label=f"{model_name} (MPS bf16)", color=colors.get(model_name, "#a78bfa"))
                for x_v, y_v, r in zip(xs, ys, m_rows):
                    y_off = -24 if model_name == "depthpro2-s" else 9
                    ax.annotate(
                        f"{y_v:.0f}ms\n({r['mp_per_sec']:.1f} MP/s)",
                        (x_v, y_v),
                        textcoords="offset points",
                        xytext=(0, y_off),
                        ha="center",
                        fontsize=8,
                        color=colors.get(model_name, "#f1f5f9"),
                    )
            ax.set_title("Resolution Scaling (Raw Inference)")
            ax.set_xlabel("Input Resolution (px, square)")
            ax.set_ylabel("Latency (ms)")
            ax.set_xticks(sorted({r["resolution"] for r in res_rows}))
            ax.set_ylim(-max(r["wall_ms"] for r in res_rows) * 0.10, max(r["wall_ms"] for r in res_rows) * 1.28)
            ax.grid(True)
            ax.legend(framealpha=0.3)

        # Right: Precision & Device comparison
        ax2 = axes[1]
        if prec_rows:
            labels = [f"{r['model'].replace('depthpro2-', '')}\n{r['device'].upper()} {r['precision']}" for r in prec_rows]
            vals = [r["wall_ms"] for r in prec_rows]
            bar_colors = ["#f97316" if r["device"] == "cpu" else ("#34d399" if r["model"] == "depthpro2-s" else "#38bdf8") for r in prec_rows]
            x = np.arange(len(labels))
            ax2.bar(x, vals, color=bar_colors, edgecolor="#cbd5e1", linewidth=0.6)
            ax2.set_yscale("log")
            for i, v in enumerate(vals):
                ax2.text(x[i], v * 1.12, f"{v:.0f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax2.set_title("Precision & Device Latency (768x768, log scale)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8.5)
            ax2.set_ylabel("Wall-Clock Latency (ms, log scale)")
            ax2.set_ylim(min(vals) * 0.5, max(vals) * 2.2)
            ax2.grid(axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        p2 = output_dir / "resolution_scaling.png"
        fig.savefig(p2, dpi=200)
        plt.close(fig)
        generated.append(p2)

    # ------------------------------------------------------------------
    # Chart 3: Speed vs Quality Pareto Frontier
    # ------------------------------------------------------------------
    all_modes = [r for r in results.get("topdown_modes", []) if "error" not in r]
    if all_modes:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200)
        fig.suptitle(f"Speed vs Elevation Quality Pareto Frontier ({chip} MPS, 768x768)", fontsize=13.5, fontweight="bold", y=0.98)

        mode_markers = {"raw": "o", "tta4": "s", "fast_multiscale": "D", "multiscale_tta": "^"}
        model_colors = {"depthpro2-s": "#34d399", "depthpro2-b": "#38bdf8"}

        for ax, (metric_key, ylabel) in zip(
            axes,
            [
                ("pearson_r", "Elevation Correlation (Pearson r) — higher is better"),
                ("sobel_edge_alignment", "Sobel Edge Alignment — higher is better"),
            ],
        ):
            pts = []
            for r in all_modes:
                x_ms = r["wall_ms"]
                y_val = r["accuracy"][metric_key]
                gf = r.get("guided_filter", False)
                marker = mode_markers.get(r["mode"], "o")
                color = model_colors.get(r["model"], "#a78bfa")
                ax.scatter(
                    x_ms,
                    y_val,
                    s=115 if not gf else 95,
                    marker=marker,
                    facecolors=color if not gf else "none",
                    edgecolors=color if gf else "#f8fafc",
                    linewidths=2.0 if gf else 1.0,
                    zorder=4,
                )
                if not gf:
                    tag = f"{r['model'].replace('depthpro2-', '')}:{r['mode']}"
                    dx_off = -8 if (x_ms > 3000) else 8
                    ha_align = "right" if (x_ms > 3000) else "left"
                    dy_off = 9 if r["mode"] != "fast_multiscale" else -15
                    ax.annotate(
                        tag,
                        (x_ms, y_val),
                        textcoords="offset points",
                        xytext=(dx_off, dy_off),
                        ha=ha_align,
                        fontsize=8.0,
                        color="#e2e8f0",
                    )
                pts.append((x_ms, y_val))

            # Legend proxy for guided-filter hollow ring
            ax.scatter([], [], s=85, facecolors="none", edgecolors="#e2e8f0", linewidths=2.0, label="Hollow = +Guided Filter (+27–30ms)")

            # Draw Pareto frontier (lower latency, higher quality)
            pts_sorted = sorted(pts, key=lambda p: p[0])
            frontier = []
            best_y = -1e9
            for px, py in pts_sorted:
                if py > best_y:
                    frontier.append((px, py))
                    best_y = py
            if len(frontier) >= 2:
                ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", color="#fbbf24", linewidth=2.0, label="Pareto Frontier", zorder=3)

            ax.set_xscale("log")
            all_x = [p[0] for p in pts]
            ax.set_xlim(min(all_x) * 0.80, max(all_x) * 1.25)
            ax.set_xticks([250, 500, 1000, 2000, 5000])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("Wall-Clock Latency (ms, log scale) — lower is better")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(" — ")[0])
            ax.grid(True)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="lower right", framealpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        p3 = output_dir / "speed_vs_quality_pareto.png"
        fig.savefig(p3, dpi=200)
        plt.close(fig)
        generated.append(p3)

    # ------------------------------------------------------------------
    # Chart 4: Memory Footprint + Cold vs Warm Server + Video Stability
    # ------------------------------------------------------------------
    cold_warm = (results.get("cold_vs_warm") or {}).get("comparisons", [])
    vid_strats = (results.get("video_stability") or {}).get("strategies", [])
    if res_rows or cold_warm or vid_strats:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=200)
        fig.suptitle(f"Memory Footprint, Warm Server Speedup & Video Temporal Stability ({chip})", fontsize=13.5, fontweight="bold", y=0.98)

        # Panel 1: Memory footprint across models & resolutions
        ax0 = axes[0]
        if res_rows:
            labels = [f"{r['model'].replace('depthpro2-', '')}\n{r['resolution']}" for r in res_rows]
            mps_alloc = [r["mps_allocated_mb"] for r in res_rows]
            mps_drv = [r["mps_driver_mb"] for r in res_rows]
            x = np.arange(len(labels))
            w = 0.38
            ax0.bar(x - w / 2, mps_alloc, width=w, label="MPS Allocated", color="#38bdf8")
            ax0.bar(x + w / 2, mps_drv, width=w, label="MPS Driver Pool", color="#818cf8")
            ax0.set_title("Apple MPS Memory (MiB)")
            ax0.set_xticks(x)
            ax0.set_xticklabels(labels, fontsize=7.5)
            ax0.set_ylabel("Memory (MiB)")
            ax0.grid(axis="y")
            ax0.legend(framealpha=0.3, fontsize=8)

        # Panel 2: Cold CLI vs Warm HTTP Server
        ax1 = axes[1]
        if cold_warm:
            models_cw = [c["model"] for c in cold_warm]
            cold_vals = [c["cold_cli_wall_ms"] for c in cold_warm]
            warm_vals = [c["warm_http_median_ms"] or 0.0 for c in cold_warm]
            x = np.arange(len(models_cw))
            w = 0.35
            ax1.bar(x - w / 2, cold_vals, width=w, label="Cold CLI Subprocess", color="#f97316")
            ax1.bar(x + w / 2, warm_vals, width=w, label="Warm HTTP Server", color="#34d399")
            for i, (cv, wv, c) in enumerate(zip(cold_vals, warm_vals, cold_warm)):
                ax1.text(x[i] - w / 2, cv + max(cold_vals) * 0.02, f"{cv:.0f}ms", ha="center", va="bottom", fontsize=8)
                if wv > 0:
                    spd = c.get("warm_speedup_vs_cold")
                    ax1.text(x[i] + w / 2, wv + max(cold_vals) * 0.02, f"{wv:.0f}ms\n({spd:.1f}x)", ha="center", va="bottom", fontsize=8, color="#34d399", fontweight="bold")
            ax1.set_title("Cold CLI vs Warm HTTP Server")
            ax1.set_xticks(x)
            ax1.set_xticklabels(models_cw)
            ax1.set_ylabel("End-to-End Latency (ms)")
            ax1.set_ylim(0, max(cold_vals + [100.0]) * 1.24)
            ax1.grid(axis="y")
            ax1.legend(framealpha=0.3, fontsize=8)

        # Panel 3: Video Temporal Stability (Overlap Jitter MAD)
        ax2 = axes[2]
        if vid_strats:
            v_labels = [s["strategy"].replace("infer_video_", "video\n").replace("single_image_", "single\n") for s in vid_strats]
            mads = [s["stability"]["overlap_mad"] for s in vid_strats]
            v_colors = ["#34d399", "#38bdf8", "#f43f5e"][: len(v_labels)]
            x = np.arange(len(v_labels))
            ax2.bar(x, mads, color=v_colors, edgecolor="#cbd5e1", linewidth=0.6)
            for i, (m_val, s) in enumerate(zip(mads, vid_strats)):
                ax2.text(x[i], m_val + max(mads + [0.001]) * 0.03, f"{m_val:.4f}\n({s['per_frame_ms']:.0f}ms/f)", ha="center", va="bottom", fontsize=8)
            ax2.set_title("Video Temporal Jitter (Overlap MAD)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(v_labels, fontsize=8.5)
            ax2.set_ylabel("Overlap MAD — lower is smoother")
            ax2.set_ylim(0, max(mads + [0.01]) * 1.28)
            ax2.grid(axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        p4 = output_dir / "memory_server_video.png"
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        generated.append(p4)

    return generated


# ---------------------------------------------------------------------------
# Full benchmark orchestration + CLI
# ---------------------------------------------------------------------------


def run_full_benchmark(
    output_dir: Path,
    baseline_path: Optional[Path] = None,
    sections: Sequence[str] = ALL_SECTIONS,
    quick: bool = False,
    repeats: int = 2,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Execute the requested benchmark sections, persist JSON, and render charts."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    section_set = set(sections)

    baseline: Optional[Dict[str, Any]] = None
    bpath = baseline_path or (output_dir / "baseline.json")
    if bpath.exists():
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        log(f"Loaded pre-optimization baseline from {bpath}")

    resolutions = (384, 512) if quick else DEFAULT_RESOLUTIONS
    precisions = ("bf16", "fp32") if quick else DEFAULT_PRECISIONS
    models = ("depthpro2-s",) if quick else DEFAULT_MODELS
    eff_repeats = 1 if quick else repeats

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": collect_hardware_info(),
        "baseline_file": str(bpath.name) if baseline else None,
        "precision_matrix": [],
        "resolution_scaling": [],
        "topdown_modes": [],
        "cold_vs_warm": {},
        "video_stability": {},
    }

    # Load existing results.json if only running a subset of sections
    results_json_path = output_dir / "results.json"
    if results_json_path.exists() and section_set != set(ALL_SECTIONS):
        existing = json.loads(results_json_path.read_text(encoding="utf-8"))
        for k in ("precision_matrix", "resolution_scaling", "topdown_modes", "cold_vs_warm", "video_stability"):
            if k in existing:
                payload[k] = existing[k]

    if "precision" in section_set:
        log("\n=== [1/5] Model & Precision Matrix (MPS + CPU reference) ===")
        payload["precision_matrix"] = run_precision_matrix(
            resolution=512 if quick else 768,
            models=models,
            precisions=precisions,
            include_cpu_reference=True,
            repeats=eff_repeats,
            log=log,
        )

    if "resolution" in section_set:
        log("\n=== [2/5] Resolution Scaling (384..1024 px) ===")
        payload["resolution_scaling"] = run_resolution_scaling(
            resolutions=resolutions,
            models=models,
            precision="bf16",
            repeats=eff_repeats,
            log=log,
        )

    if "modes" in section_set:
        log("\n=== [3/5] Top-Down Modes & Guided Filter (Speed vs Quality) ===")
        payload["topdown_modes"] = run_topdown_modes(
            resolution=512 if quick else 768,
            models=models,
            precision="bf16",
            baseline=baseline,
            repeats=eff_repeats,
            log=log,
        )

    if "cold_vs_warm" in section_set:
        log("\n=== [4/5] Cold CLI vs Warm HTTP Server Latency ===")
        payload["cold_vs_warm"] = run_cold_vs_warm_server(
            resolution=512 if quick else 768,
            models=models,
            warm_requests=2 if quick else 3,
            log=log,
        )

    if "video" in section_set:
        log("\n=== [5/5] Video Temporal Stability (12-frame Synthetic Pan) ===")
        payload["video_stability"] = run_video_stability(
            model_name="depthpro2-s",
            num_frames=8 if quick else 12,
            frame_size=384 if quick else 512,
            pan_dx_px=16,
            log=log,
        )

    results_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"\nSaved benchmark results JSON -> {results_json_path}")

    if "charts" in section_set:
        charts = generate_charts(payload, output_dir)
        log(f"Generated {len(charts)} charts: {[c.name for c in charts]}")

    return payload


@click.command()
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("../docs/benchmarks"),
    show_default=True,
    help="Output directory for results.json and PNG charts.",
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional baseline.json path for BEFORE vs AFTER speedup comparison.",
)
@click.option(
    "--sections",
    type=str,
    default="all",
    show_default=True,
    help="Comma-separated sections: all or subset of precision,resolution,modes,cold_vs_warm,video,charts.",
)
@click.option("--repeats", type=int, default=2, show_default=True, help="Timed runs per measurement (median reported).")
@click.option("--quick", is_flag=True, help="Run reduced quick smoke benchmark.")
def main(
    output_dir: Path,
    baseline_path: Optional[Path],
    sections: str,
    repeats: int,
    quick: bool,
) -> None:
    """Run the Apple Depth Pro 2 reproducible benchmark & efficiency suite."""
    if sections.strip().lower() == "all":
        selected = ALL_SECTIONS
    else:
        selected = tuple(s.strip().lower() for s in sections.split(",") if s.strip())
    run_full_benchmark(
        output_dir=output_dir,
        baseline_path=baseline_path,
        sections=selected,
        quick=quick,
        repeats=repeats,
    )


if __name__ == "__main__":
    main()
