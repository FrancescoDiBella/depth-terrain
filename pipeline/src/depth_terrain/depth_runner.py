"""Apple Depth Pro 2 model management and image/video inference runner.

Includes top-down nadir adaptation strategies (4-way rotational TTA, fast 2x2
multiscale TTA, linear plane detrending, perspective-tunnel radial debiasing,
and overlapping multi-scale patch fusion) so monocular perspective depth from
Apple Depth Pro 2 converts cleanly into rotation-invariant topographic
heightmaps for Unity Terrains.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import laplace

from depth_pro_2 import PRESETS, create_model_and_transforms, load_rgb, upright
from depth_pro_2 import model as depth_pro_2_model
from depth_pro_2.utils import default_device

LOGGER = logging.getLogger(__name__)

VALID_MODELS = tuple(sorted(PRESETS.keys()))
TOPDOWN_MODES = ("multiscale_tta", "fast_multiscale", "tta4", "raw")
PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
COREML_REPO_ID = "apple/DepthPro2-coreml"

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: "OrderedDict[Tuple[str, str, str, str], Tuple[Any, Any, torch.device]]" = OrderedDict()
_MAX_CACHED_MODELS = 2


def get_default_checkpoints_dir() -> Path:
    """Return local gitignored directory for Depth Pro 2 model weights."""
    return Path(__file__).resolve().parents[2] / "checkpoints"


def configure_checkpoint_dir(checkpoints_dir: Optional[Path] = None) -> Path:
    """Point depth_pro_2's checkpoint resolver to a persistent local folder."""
    target = (checkpoints_dir or get_default_checkpoints_dir()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    depth_pro_2_model.CHECKPOINT_DIR = target
    return target


def download_model_weights(
    model_name: str = "depthpro2-b",
    checkpoints_dir: Optional[Path] = None,
    include_fov: bool = True,
) -> dict[str, Path]:
    """Download Apple Depth Pro 2 checkpoint(s) directly from Apple's official CDN."""
    if model_name not in PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(VALID_MODELS)}")

    target_dir = configure_checkpoint_dir(checkpoints_dir)
    preset = PRESETS[model_name]
    downloaded: dict[str, Path] = {}

    urls = [("model", preset.checkpoint_uri)]
    if include_fov and preset.fov_checkpoint_uri:
        urls.append(("fov", preset.fov_checkpoint_uri))

    for key, uri in urls:
        if not uri:
            continue
        filename = uri.rsplit("/", 1)[-1]
        dest = target_dir / filename
        if not dest.exists():
            LOGGER.info("Downloading %s (%s) -> %s", key, uri, dest)
            torch.hub.download_url_to_file(uri, str(dest), progress=True)
        else:
            LOGGER.info("Using cached %s checkpoint: %s", key, dest)
        downloaded[key] = dest
    return downloaded


@dataclass
class DepthInferenceResult:
    """Single-frame or video-frame Depth Pro 2 inference output."""

    depth_m: np.ndarray  # (H, W) float32 meters
    inverse_depth: np.ndarray  # (H, W) float32 focal-normalized inverse depth (or fused elevation signal)
    focallength_px: float
    model_name: str
    device: str
    precision: str
    inference_ms: float
    topdown_mode: str = "raw"


def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve PyTorch compute device (auto prefers CUDA, then Apple MPS, then CPU)."""
    if not device_name or device_name == "auto":
        return default_device()
    return torch.device(device_name)


def resolve_precision_name(device: torch.device, precision_name: Optional[str] = None) -> str:
    """Resolve effective precision label (`bf16`, `fp16`, or `fp32`) for a target device."""
    if precision_name:
        key = precision_name.strip().lower()
        aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
        key = aliases.get(key, key)
        if key not in PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision_name}'. Choose one of: {', '.join(sorted(PRECISIONS))}"
            )
        return key
    return "fp32" if device.type == "cpu" else "bf16"


def load_depth_pro_2(
    model_name: str = "depthpro2-b",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
):
    """Load Depth Pro 2 model and preprocessing transform onto target device."""
    if model_name not in PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(VALID_MODELS)}")

    configure_checkpoint_dir(checkpoints_dir)
    download_model_weights(model_name, checkpoints_dir=checkpoints_dir, include_fov=True)

    device = resolve_device(device_name)
    eff_prec_name = resolve_precision_name(device, precision_name)
    precision = PRECISIONS[eff_prec_name]
    model, transform = create_model_and_transforms(model_name, device=device, precision=precision)
    model.eval()
    return model, transform, device


def get_cached_model(
    model_name: str = "depthpro2-b",
    device: str = "auto",
    precision: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    max_cache_size: int = _MAX_CACHED_MODELS,
):
    """Return a warm Depth Pro 2 `(model, transform, device)` bundle from an in-process LRU cache."""
    resolved_dev = resolve_device(device)
    eff_prec = resolve_precision_name(resolved_dev, precision)
    ckpt_dir = (checkpoints_dir or get_default_checkpoints_dir()).resolve()
    key = (model_name, str(resolved_dev), eff_prec, str(ckpt_dir))

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(key)
            return cached

    loaded = load_depth_pro_2(
        model_name=model_name,
        device_name=str(resolved_dev),
        precision_name=eff_prec,
        checkpoints_dir=ckpt_dir,
    )

    with _MODEL_CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            _MODEL_CACHE.move_to_end(key)
            return existing
        _MODEL_CACHE[key] = loaded
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > max(1, int(max_cache_size)):
            _, evicted = _MODEL_CACHE.popitem(last=False)
            evicted_dev = evicted[2]
            del evicted
            if evicted_dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif evicted_dev.type == "cuda":
                torch.cuda.empty_cache()
    return loaded


def list_cached_models() -> list[str]:
    """Return unique Depth Pro 2 model names currently warm in the in-process LRU cache."""
    with _MODEL_CACHE_LOCK:
        seen: list[str] = []
        for model_name, _, _, _ in _MODEL_CACHE.keys():
            if model_name not in seen:
                seen.append(model_name)
        return seen


def clear_model_cache() -> None:
    """Evict all warm Depth Pro 2 models from the in-process LRU cache."""
    with _MODEL_CACHE_LOCK:
        devices = {bundle[2].type for bundle in _MODEL_CACHE.values()}
        _MODEL_CACHE.clear()
    if "mps" in devices and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if "cuda" in devices:
        torch.cuda.empty_cache()


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def detrend_linear_plane(field: np.ndarray) -> np.ndarray:
    """Fit and subtract a best-fit 2D plane ax + by + c to cancel perspective camera tilt.

    Exploits the orthogonality of centered regular grid coordinates xn, yn in [-1, 1]
    to solve the least-squares plane analytically in O(H*W) without SVD/lstsq overhead.
    """
    f = np.asarray(field, dtype=np.float64)
    h, w = f.shape
    xn = np.linspace(-1.0, 1.0, w, dtype=np.float64) if w > 1 else np.zeros(1, dtype=np.float64)
    yn = np.linspace(-1.0, 1.0, h, dtype=np.float64) if h > 1 else np.zeros(1, dtype=np.float64)

    col_sum = f.sum(axis=0)
    row_sum = f.sum(axis=1)
    c = float(col_sum.sum()) / float(max(h * w, 1))

    xn_sq = float(np.dot(xn, xn))
    yn_sq = float(np.dot(yn, yn))
    a = float(np.dot(col_sum, xn)) / (h * xn_sq) if xn_sq > 1e-12 else 0.0
    b = float(np.dot(row_sum, yn)) / (w * yn_sq) if yn_sq > 1e-12 else 0.0

    plane = (a * xn[None, :] + b * yn[:, None] + c)
    return (f - plane).astype(np.float32)


def debias_positive_radial_vignette(field: np.ndarray) -> np.ndarray:
    """Remove perspective tunnel bias (corners artificially closer than center) when present.

    Only subtracts positive r^2 curvature so genuine central peaks/islands are preserved.
    Solved analytically via 1D covariance/variance regression for minimal CPU overhead.
    """
    f = np.asarray(field, dtype=np.float64)
    h, w = f.shape
    xn = np.linspace(-1.0, 1.0, w, dtype=np.float64) if w > 1 else np.zeros(1, dtype=np.float64)
    yn = np.linspace(-1.0, 1.0, h, dtype=np.float64) if h > 1 else np.zeros(1, dtype=np.float64)
    r2 = (xn[None, :] ** 2) + (yn[:, None] ** 2)

    r2_mean = float(r2.mean())
    r2_c = r2 - r2_mean
    denom = float(np.sum(r2_c * r2_c))
    if denom <= 1e-12:
        return field.astype(np.float32)

    f_mean = float(f.mean())
    slope = float(np.sum(r2_c * (f - f_mean))) / denom
    if slope > 0.0:
        intercept = f_mean - slope * r2_mean
        return (f - (slope * r2 + intercept)).astype(np.float32)
    return field.astype(np.float32)


def _infer_primary_once(
    image_rgb: np.ndarray,
    model,
    transform,
    f_px: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run primary frame inference. Skips the 94MB FOV network when ``f_px`` is already known."""
    writable = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")
    h, w = writable.shape[:2]
    with torch.inference_mode():
        tensor = transform(writable)
        if f_px is not None:
            inv_t = model.predict_inverse_depth(tensor.unsqueeze(0))[0]
            inv_depth = inv_t.detach().float().cpu().numpy()
            focal_val = float(f_px)
            depth_m = (focal_val / np.maximum(w * inv_depth, 1e-6)).astype(np.float32)
            return depth_m, inv_depth, focal_val

        prediction = model.infer(tensor, f_px=None)
        inv_depth = prediction["inverse_depth"].detach().float().cpu().numpy()
        depth_tensor = prediction["depth"]
        focal_tensor = prediction["focallength_px"]
        focal_val = float(focal_tensor.detach().cpu().item()) if focal_tensor is not None else float(w)
        if depth_tensor is not None:
            depth_m = depth_tensor.detach().float().cpu().numpy()
        else:
            depth_m = (focal_val / np.maximum(w * inv_depth, 1e-6)).astype(np.float32)
        return depth_m, inv_depth, focal_val


def _infer_inverse_depth_only(
    image_rgb: np.ndarray,
    model,
    transform,
) -> np.ndarray:
    """Run inverse depth head directly via ``predict_inverse_depth`` (no redundant FOV pass)."""
    writable = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")
    with torch.inference_mode():
        tensor = transform(writable).unsqueeze(0)
        inv_t = model.predict_inverse_depth(tensor)[0]
        return inv_t.detach().float().cpu().numpy()


def _predict_independent_batch(model, x_batch: torch.Tensor) -> torch.Tensor:
    """Batched independent single-frame inverse depth without cross-frame mixing.

    Preserves prefix tokens on ``cross_frame_blocks`` so each item in the batch is
    mathematically equivalent to calling ``model.predict_inverse_depth`` with T=1.
    """
    height, width = x_batch.shape[-2:]
    input_mult = depth_pro_2_model.INPUT_MULTIPLE
    pad_h = -height % input_mult
    pad_w = -width % input_mult
    if pad_h or pad_w:
        x_batch = F.pad(x_batch, (0, pad_w, 0, pad_h), mode="replicate")

    enc = model.encoder
    npt = enc.num_prefix_tokens
    grid_size = (x_batch.shape[-2] // enc.patch_size, x_batch.shape[-1] // enc.patch_size)
    rope = enc.rope.get_embed(shape=grid_size)

    with depth_pro_2_model.autocast_for(x_batch.device, model.precision)():
        tokens = enc.embed(x_batch)
        feats = []
        for idx, block in enumerate(enc.blocks):
            if idx in enc.cross_frame_blocks:
                prefix = tokens[:, :npt]
                out_blk = block(tokens, rope=rope)
                tokens = torch.cat([prefix, out_blk[:, npt:]], dim=1)
            else:
                tokens = block(tokens, rope=rope)
            if idx in enc.reassembled_blocks:
                feats.append(tokens[:, npt:])
        decoded = model.decoder(enc.reassemble_tokens(feats, grid_size))
    return model.head(decoded.float())[:, 0, :height, :width]


def _infer_inverse_depth_patches(
    patches_rgb: Sequence[np.ndarray],
    model,
    transform,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> list[np.ndarray]:
    """Run FOV-free inverse depth across equal-sized local patches.

    Benchmarks on Apple Silicon M3 MPS show sequential ``predict_inverse_depth`` (bs=1)
    is faster and lower-memory than batched execution because M3 saturates at B=1 and
    avoids Metal graph recompilation. On CUDA GPUs, independent batching (bs=4) is used.
    """
    if not patches_rgb:
        return []

    eff_bs = batch_size
    if eff_bs is None:
        eff_bs = 4 if (device.type == "cuda" and hasattr(model, "encoder")) else 1

    if eff_bs <= 1 or len(patches_rgb) == 1 or not hasattr(model, "encoder"):
        return [_infer_inverse_depth_only(p, model, transform) for p in patches_rgb]

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(patches_rgb), eff_bs):
            chunk = patches_rgb[start : start + eff_bs]
            tensors = torch.stack(
                [transform(np.array(p, dtype=np.uint8, copy=True, order="C")) for p in chunk],
                dim=0,
            )
            preds = _predict_independent_batch(model, tensors).detach().float().cpu().numpy()
            for idx in range(preds.shape[0]):
                outputs.append(preds[idx])
    return outputs


def infer_image_depth(
    image_rgb: np.ndarray,
    model_name: str = "depthpro2-b",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    f_px: Optional[float] = None,
    topdown_mode: str = "multiscale_tta",
    checkpoints_dir: Optional[Path] = None,
    loaded_bundle: Optional[Tuple] = None,
    use_cache: bool = True,
    patch_batch_size: Optional[int] = None,
) -> DepthInferenceResult:
    """Run Apple Depth Pro 2 on a single (H, W, 3) uint8 RGB image.

    Args:
        topdown_mode:
            - ``raw``: Direct single-pass Depth Pro 2 inference (fastest; ideal for oblique views).
            - ``tta4``: 4-way rotational TTA (0/90/180/270) + plane detrending + radial debias (4 passes, 1 FOV).
            - ``fast_multiscale``: 2-way global TTA (0/180) + 2x2 overlapping local patches (6 passes, 1 FOV; ~3x faster).
            - ``multiscale_tta``: Global 4-way TTA fused with 3x3 overlapping 0/180 local patches (22 passes, 1 FOV).
    """
    if topdown_mode not in TOPDOWN_MODES:
        raise ValueError(f"Unknown topdown_mode '{topdown_mode}'. Choose one of: {', '.join(TOPDOWN_MODES)}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image_rgb.shape}.")

    writable = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")

    if loaded_bundle is not None:
        model, transform, device = loaded_bundle
    elif use_cache:
        model, transform, device = get_cached_model(
            model_name=model_name,
            device=device_name,
            precision=precision_name,
            checkpoints_dir=checkpoints_dir,
        )
    else:
        model, transform, device = load_depth_pro_2(
            model_name=model_name,
            device_name=device_name,
            precision_name=precision_name,
            checkpoints_dir=checkpoints_dir,
        )

    _sync_device(device)
    t0 = time.perf_counter()

    if topdown_mode == "raw":
        depth_m, inv_depth, focal_val = _infer_primary_once(writable, model, transform, f_px=f_px)
    else:
        h, w = writable.shape[:2]
        g_acc = np.zeros((h, w), dtype=np.float32)

        # Primary pass (k=0): runs FOV estimator at most once (or skips if f_px is provided)
        base_depth_m, inv_0, focal_val = _infer_primary_once(writable, model, transform, f_px=f_px)
        det0 = detrend_linear_plane(inv_0)
        g_acc += det0 / (float(np.std(det0)) or 1.0)

        rot_ks = (2,) if topdown_mode == "fast_multiscale" else (1, 2, 3)
        for k in rot_ks:
            rot = np.array(np.rot90(writable, k), dtype=np.uint8, copy=True, order="C")
            inv_k = _infer_inverse_depth_only(rot, model, transform)
            unrot = np.rot90(inv_k, -k)
            detrended = detrend_linear_plane(unrot)
            std = float(np.std(detrended)) or 1.0
            g_acc += detrended / std

        g_map = debias_positive_radial_vignette(g_acc / float(1 + len(rot_ks)))
        g_map = g_map / (float(np.std(g_map)) or 1.0)

        if topdown_mode == "fast_multiscale" and min(h, w) >= 384:
            # 2x2 overlapping local patches (60% span -> 20% overlap) = 4 passes, 0 FOV calls
            patch_h = min(h, max(256, int(round(h * 0.6))))
            patch_w = min(w, max(256, int(round(w * 0.6))))
            ys = sorted({0, h - patch_h})
            xs = sorted({0, w - patch_w})
            win = np.outer(np.hanning(patch_h + 2)[1:-1], np.hanning(patch_w + 2)[1:-1]).astype(np.float32)

            coords = [(y0, x0) for y0 in ys for x0 in xs]
            patches = [writable[y0 : y0 + patch_h, x0 : x0 + patch_w] for y0, x0 in coords]
            inv_patches = _infer_inverse_depth_patches(
                patches, model, transform, device, batch_size=patch_batch_size
            )

            p_acc = np.zeros((h, w), dtype=np.float32)
            w_acc = np.zeros((h, w), dtype=np.float32)
            for (y0, x0), inv_p in zip(coords, inv_patches):
                pd = detrend_linear_plane(inv_p)
                pd = pd / (float(np.std(pd)) or 1.0)
                p_acc[y0 : y0 + patch_h, x0 : x0 + patch_w] += pd * win
                w_acc[y0 : y0 + patch_h, x0 : x0 + patch_w] += win

            p_map = p_acc / np.maximum(w_acc, 1e-6)
            p_map = p_map / (float(np.std(p_map)) or 1.0)
            inv_depth = (0.60 * g_map + 0.40 * p_map).astype(np.float32)
        elif topdown_mode == "multiscale_tta" and min(h, w) >= 384:
            # 3x3 overlapping local patches with 0/180 rotational averaging = 18 passes, 0 FOV calls
            patch_h = max(256, h // 2)
            patch_w = max(256, w // 2)
            ys = sorted({0, (h - patch_h) // 2, h - patch_h})
            xs = sorted({0, (w - patch_w) // 2, w - patch_w})
            win = np.outer(np.hanning(patch_h + 2)[1:-1], np.hanning(patch_w + 2)[1:-1]).astype(np.float32)

            coords = [(y0, x0) for y0 in ys for x0 in xs]
            patches_0 = [writable[y0 : y0 + patch_h, x0 : x0 + patch_w] for y0, x0 in coords]
            patches_180 = [np.rot90(p, 2) for p in patches_0]
            all_preds = _infer_inverse_depth_patches(
                patches_0 + patches_180, model, transform, device, batch_size=patch_batch_size
            )
            n_patches = len(coords)
            inv_0_list = all_preds[:n_patches]
            inv_180_list = all_preds[n_patches:]

            p_acc = np.zeros((h, w), dtype=np.float32)
            w_acc = np.zeros((h, w), dtype=np.float32)
            for (y0, x0), inv0, inv2_rot in zip(coords, inv_0_list, inv_180_list):
                inv2 = np.rot90(inv2_rot, -2)
                pd = detrend_linear_plane(0.5 * (inv0 + inv2))
                pd = pd / (float(np.std(pd)) or 1.0)
                p_acc[y0 : y0 + patch_h, x0 : x0 + patch_w] += pd * win
                w_acc[y0 : y0 + patch_h, x0 : x0 + patch_w] += win

            p_map = p_acc / np.maximum(w_acc, 1e-6)
            p_map = p_map / (float(np.std(p_map)) or 1.0)
            inv_depth = (0.60 * g_map + 0.40 * p_map).astype(np.float32)
        else:
            inv_depth = g_map.astype(np.float32)

        depth_m = base_depth_m

    _sync_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    prec_label = str(getattr(model, "precision", "float32")).removeprefix("torch.")

    return DepthInferenceResult(
        depth_m=depth_m,
        inverse_depth=inv_depth,
        focallength_px=focal_val,
        model_name=model_name,
        device=str(device),
        precision=prec_label,
        inference_ms=elapsed_ms,
        topdown_mode=topdown_mode,
    )


def load_local_image(image_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Load an image file (PNG/JPG/WebP/HEIC) as uint8 RGB + optional EXIF focal length."""
    rgb, f_px = load_rgb(image_path)
    return np.array(rgb, dtype=np.uint8, copy=True), (float(f_px) if f_px is not None else None)


def score_video_frame(rgb: np.ndarray, inverse_depth: np.ndarray) -> Tuple[float, float]:
    """Compute (sharpness, depth_relief) metrics for smart aerial video keyframe selection.

    - sharpness: Laplacian variance of grayscale luminance (penalizes motion blur).
    - depth_relief: robust interquartile/p95-p05 elevation spread + std of inverse depth.
    """
    rgb_f = np.asarray(rgb, dtype=np.float32)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    sharpness = float(np.var(laplace(gray)))

    inv = np.nan_to_num(np.asarray(inverse_depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p95 = float(np.percentile(inv, 95.0))
    p05 = float(np.percentile(inv, 5.0))
    relief = max(0.0, p95 - p05) + float(np.std(inv))
    return sharpness, relief


def select_best_video_keyframe(
    results: Sequence[Tuple[int, np.ndarray, DepthInferenceResult]],
) -> int:
    """Pick the sharpest, richest-relief video frame index rather than blindly using the midpoint."""
    if not results:
        raise ValueError("Cannot select keyframe from empty video results.")
    if len(results) == 1:
        return 0

    raw_scores = [score_video_frame(rgb, res.inverse_depth) for _, rgb, res in results]
    sharp_arr = np.array([s[0] for s in raw_scores], dtype=np.float64)
    relief_arr = np.array([s[1] for s in raw_scores], dtype=np.float64)

    def _norm(arr: np.ndarray) -> np.ndarray:
        span = float(arr.max() - arr.min())
        if span < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - arr.min()) / span

    sharp_n = _norm(sharp_arr)
    relief_n = _norm(relief_arr)

    # Slight center tie-breaker (1% weight) when multiple frames score identically
    n = len(results)
    mid = (n - 1) * 0.5
    center_bias = np.array([1.0 - abs(i - mid) / max(mid, 1.0) for i in range(n)], dtype=np.float64)

    combined = 0.54 * sharp_n + 0.45 * relief_n + 0.01 * center_bias
    return int(np.argmax(combined))


def infer_video_depth(
    video_path: Path,
    model_name: str = "depthpro2-s",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    mode: str = "buffered",
    window: int = 8,
    overlap: int = 2,
    max_frames: int = 24,
    frame_stride: int = 1,
    max_side: int = 768,
    detrend_frames: bool = True,
    checkpoints_dir: Optional[Path] = None,
    loaded_bundle: Optional[Tuple] = None,
    use_cache: bool = True,
) -> Iterator[Tuple[int, np.ndarray, DepthInferenceResult]]:
    """Run temporally consistent Apple Depth Pro 2 inference across aerial video frames."""
    from PIL import Image

    if mode not in ("buffered", "unbuffered"):
        raise ValueError(f"Unknown video mode '{mode}'. Choose 'buffered' or 'unbuffered'.")

    if loaded_bundle is not None:
        model, transform, device = loaded_bundle
    elif use_cache:
        model, transform, device = get_cached_model(
            model_name=model_name,
            device=device_name,
            precision=precision_name,
            checkpoints_dir=checkpoints_dir,
        )
    else:
        model, transform, device = load_depth_pro_2(
            model_name=model_name,
            device_name=device_name,
            precision_name=precision_name,
            checkpoints_dir=checkpoints_dir,
        )

    container = av.open(str(video_path))
    raw_frames: list[Tuple[int, np.ndarray]] = []
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % max(1, frame_stride) != 0:
                continue
            rgb = np.array(upright(frame), dtype=np.uint8, copy=True)
            if max_side > 0 and max(rgb.shape[0], rgb.shape[1]) > max_side:
                scale = max_side / float(max(rgb.shape[0], rgb.shape[1]))
                new_w = max(32, int(round(rgb.shape[1] * scale)))
                new_h = max(32, int(round(rgb.shape[0] * scale)))
                rgb = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR), copy=True)
            raw_frames.append((idx, rgb))
            if len(raw_frames) >= max_frames:
                break
    finally:
        container.close()

    if not raw_frames:
        raise ValueError(f"No video frames decoded from {video_path}.")

    eff_window = max(1, min(window, len(raw_frames)))
    eff_overlap = min(overlap, max(0, eff_window - 1)) if mode == "buffered" else 0
    prec_label = str(getattr(model, "precision", "float32")).removeprefix("torch.")

    if mode == "unbuffered":
        # True low-latency streaming: yield each frame immediately as it completes
        stream = model.stream(window=eff_window, overlap=0, mode="unbuffered")
        with torch.inference_mode():
            for frame_idx, rgb in raw_frames:
                t0 = time.perf_counter()
                preds = stream.step(transform(rgb))
                _sync_device(device)
                frame_ms = (time.perf_counter() - t0) * 1000.0
                for pred in preds:
                    inv_depth = pred["inverse_depth"].detach().float().cpu().numpy()
                    if detrend_frames:
                        inv_depth = debias_positive_radial_vignette(detrend_linear_plane(inv_depth))
                    depth_m = pred["depth"].detach().float().cpu().numpy()
                    focal = float(pred["focallength_px"].detach().cpu().item())
                    yield (
                        frame_idx,
                        rgb,
                        DepthInferenceResult(
                            depth_m=depth_m,
                            inverse_depth=inv_depth,
                            focallength_px=focal,
                            model_name=model_name,
                            device=str(device),
                            precision=prec_label,
                            inference_ms=frame_ms,
                            topdown_mode="video_unbuffered_detrended" if detrend_frames else "video_unbuffered",
                        ),
                    )
            for pred in stream.flush():
                frame_idx, rgb = raw_frames[-1]
                inv_depth = pred["inverse_depth"].detach().float().cpu().numpy()
                if detrend_frames:
                    inv_depth = debias_positive_radial_vignette(detrend_linear_plane(inv_depth))
                depth_m = pred["depth"].detach().float().cpu().numpy()
                focal = float(pred["focallength_px"].detach().cpu().item())
                yield (
                    frame_idx,
                    rgb,
                    DepthInferenceResult(
                        depth_m=depth_m,
                        inverse_depth=inv_depth,
                        focallength_px=focal,
                        model_name=model_name,
                        device=str(device),
                        precision=prec_label,
                        inference_ms=0.0,
                        topdown_mode="video_unbuffered_detrended" if detrend_frames else "video_unbuffered",
                    ),
                )
        return

    tensors = (transform(rgb) for _, rgb in raw_frames)
    t0 = time.perf_counter()
    with torch.inference_mode():
        preds = list(model.infer_video(tensors, window=eff_window, overlap=eff_overlap, mode="buffered"))
    _sync_device(device)
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_frame_ms = total_ms / max(len(preds), 1)

    for (frame_idx, rgb), pred in zip(raw_frames, preds):
        inv_depth = pred["inverse_depth"].detach().float().cpu().numpy()
        if detrend_frames:
            inv_depth = debias_positive_radial_vignette(detrend_linear_plane(inv_depth))
        depth_m = pred["depth"].detach().float().cpu().numpy()
        focal = float(pred["focallength_px"].detach().cpu().item())
        yield (
            frame_idx,
            rgb,
            DepthInferenceResult(
                depth_m=depth_m,
                inverse_depth=inv_depth,
                focallength_px=focal,
                model_name=model_name,
                device=str(device),
                precision=prec_label,
                inference_ms=per_frame_ms,
                topdown_mode="video_detrended" if detrend_frames else "raw",
            ),
        )


def get_model_info(checkpoints_dir: Optional[Path] = None) -> dict[str, Any]:
    """Inspect cached Depth Pro 2 checkpoints, Apple Silicon MPS, and Core ML readiness."""
    target_dir = (checkpoints_dir or get_default_checkpoints_dir()).resolve()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")

    if cuda_available:
        rec_device = "cuda"
        rec_precision = "bf16"
    elif mps_available:
        rec_device = "mps"
        rec_precision = "bf16"
    else:
        rec_device = "cpu"
        rec_precision = "fp32"

    checkpoints: dict[str, dict[str, Any]] = {}
    for name in VALID_MODELS:
        preset = PRESETS[name]
        model_file = preset.checkpoint_uri.rsplit("/", 1)[-1] if preset.checkpoint_uri else ""
        fov_file = preset.fov_checkpoint_uri.rsplit("/", 1)[-1] if preset.fov_checkpoint_uri else ""
        model_path = target_dir / model_file if model_file else None
        fov_path = target_dir / fov_file if fov_file else None
        downloaded = bool(model_path and model_path.exists())
        fov_downloaded = bool(fov_path and fov_path.exists())
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 1) if downloaded and model_path else 0.0
        checkpoints[name] = {
            "downloaded": downloaded,
            "path": str(model_path) if model_path else "",
            "sizeMb": size_mb,
            "fovDownloaded": fov_downloaded,
        }

    coreml_spec = importlib.util.find_spec("coremltools")
    coreml_installed = coreml_spec is not None
    coreml_version: Optional[str] = None
    if coreml_installed:
        try:
            import coremltools  # type: ignore

            coreml_version = getattr(coremltools, "__version__", "unknown")
        except Exception:
            coreml_version = "import_error"

    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "appleSilicon": is_apple_silicon,
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
        },
        "acceleration": {
            "mpsAvailable": mps_available,
            "cudaAvailable": cuda_available,
            "recommendedDevice": rec_device,
            "recommendedPrecision": rec_precision,
            "recommendedTopDownMode": "fast_multiscale",
            "recommendedModel": "depthpro2-b" if (mps_available or cuda_available) else "depthpro2-s",
        },
        "checkpointsDir": str(target_dir),
        "checkpoints": checkpoints,
        "cachedInMemory": list_cached_models(),
        "coreml": {
            "repoId": COREML_REPO_ID,
            "coremltoolsInstalled": coreml_installed,
            "coremltoolsVersion": coreml_version,
            "appleSiliconReady": is_apple_silicon,
            "notes": (
                "Pre-converted Apple Neural Engine / Metal Core ML packages are available at "
                f"https://huggingface.co/{COREML_REPO_ID}. Install `coremltools` on macOS to "
                "inspect or export custom Core ML packages."
            ),
        },
    }
