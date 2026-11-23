"""Convert Depth Pro 2 metric/inverse depth arrays to Unity 6 Terrain assets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace, uniform_filter

VALID_UNITY_HEIGHTMAP_RESOLUTIONS = (33, 65, 129, 257, 513, 1025, 2049)
TURBO_LUT = (colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)


@dataclass
class ExportConfig:
    """Heightmap normalization and Unity export settings."""

    resolution: int = 513
    use_inverse_depth: bool = True
    invert: bool = False
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    smooth_sigma: float = 1.0
    gamma: float = 1.0
    water_clamp: float = 0.0
    save_npz: bool = False
    default_terrain_width_m: float = 500.0
    default_terrain_height_m: float = 120.0
    guided_filter: bool = False
    guided_radius: int = 8
    guided_eps: float = 1e-3
    export_ply: bool = False
    ply_max_points: int = 300_000
    ply_binary: bool = True
    export_masks: bool = True


@dataclass
class ExportResult:
    """Paths and summary statistics for an exported Unity terrain bundle."""

    output_dir: Path
    albedo_path: Path
    heightmap_raw_path: Path
    heightmap_png16_path: Path
    depth_preview_path: Path
    metadata_path: Path
    npz_path: Optional[Path]
    metadata: Dict[str, Any]
    terrain_masks_path: Optional[Path] = None
    pointcloud_ply_path: Optional[Path] = None


def nearest_unity_resolution(requested: int) -> int:
    """Snap any requested resolution to a valid Unity TerrainData resolution (2^n + 1)."""
    if requested in VALID_UNITY_HEIGHTMAP_RESOLUTIONS:
        return requested
    return min(VALID_UNITY_HEIGHTMAP_RESOLUTIONS, key=lambda r: abs(r - requested))


def guided_filter_gray(
    guide_rgb: np.ndarray,
    target: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Fast O(N) edge-preserving guided filter using RGB albedo luminance as guide.

    Uses box filters via ``scipy.ndimage.uniform_filter`` (He et al., 2013) to align
    heightmap discontinuities with building edges, cliffs, craters, and coastlines
    while suppressing flat-region estimation noise.
    """
    p = np.asarray(target, dtype=np.float32)
    h, w = p.shape

    guide = np.asarray(guide_rgb, dtype=np.float32)
    if guide.ndim == 3 and guide.shape[2] >= 3:
        lum = 0.299 * guide[..., 0] + 0.587 * guide[..., 1] + 0.114 * guide[..., 2]
    elif guide.ndim == 2:
        lum = guide
    else:
        raise ValueError(f"Expected 2D grayscale or (H, W, 3) RGB guide, got shape {guide.shape}.")

    if float(lum.max()) > 1.5:
        lum = lum / 255.0
    lum = np.clip(lum, 0.0, 1.0).astype(np.float32)

    if lum.shape != (h, w):
        pil_lum = Image.fromarray(lum)
        lum = np.asarray(pil_lum.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)

    r = max(1, int(radius))
    win = 2 * r + 1
    eps_f = max(float(eps), 1e-8)

    mean_i = uniform_filter(lum, size=win, mode="reflect")
    mean_p = uniform_filter(p, size=win, mode="reflect")
    corr_i = uniform_filter(lum * lum, size=win, mode="reflect")
    corr_ip = uniform_filter(lum * p, size=win, mode="reflect")

    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps_f)
    b = mean_p - a * mean_i

    mean_a = uniform_filter(a, size=win, mode="reflect")
    mean_b = uniform_filter(b, size=win, mode="reflect")

    filtered = mean_a * lum + mean_b
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def compute_terrain_masks(
    norm_height: np.ndarray,
    terrain_size_m: float = 500.0,
    terrain_height_m: float = 120.0,
) -> np.ndarray:
    """Compute topographic RGBA masks from a [0, 1] heightmap for Unity shading.

    Channel packing:
        - R: Topographic slope angle normalized to [0, 1] (0 = flat, 1 = vertical cliff).
        - G: Multi-scale ambient occlusion / cavity factor in [0, 1] (dark valleys, bright ridges).
        - B: Ridge curvature in [0, 1] (0.5 = planar, >0.5 = convex ridge/peak, <0.5 = concave valley).
        - A: 255 (opaque).
    """
    h_map = np.clip(np.asarray(norm_height, dtype=np.float32), 0.0, 1.0)
    h, w = h_map.shape
    cell_m = max(float(terrain_size_m) / max(min(h, w) - 1, 1), 1e-4)
    elev_m = h_map * float(max(terrain_height_m, 1e-3))

    # R: Physical slope angle normalized to [0, 1] (0..90 deg)
    gy, gx = np.gradient(elev_m, cell_m, cell_m)
    grad_mag = np.hypot(gx, gy)
    slope = np.arctan(grad_mag) / (0.5 * math.pi)
    slope_u8 = np.clip(np.round(slope * 255.0), 0, 255).astype(np.uint8)

    # G: Multi-scale local cavity ambient occlusion
    s_small = max(1.0, min(h, w) * 0.008)
    s_large = max(3.0, min(h, w) * 0.030)
    local_diff = (
        0.60 * (h_map - gaussian_filter(h_map, sigma=s_small))
        + 0.40 * (h_map - gaussian_filter(h_map, sigma=s_large))
    )
    ao = np.clip(0.85 + local_diff * 5.0, 0.10, 1.0)
    ao_u8 = np.clip(np.round(ao * 255.0), 0, 255).astype(np.uint8)

    # B: Ridge/valley Laplacian curvature centered at 0.5
    smooth_h = gaussian_filter(h_map, sigma=max(0.8, min(h, w) * 0.004))
    raw_curv = -laplace(smooth_h)
    curv_scale = float(np.percentile(np.abs(raw_curv), 95)) or 1e-4
    curv = 0.5 + 0.5 * np.tanh(raw_curv / (curv_scale * 1.5))
    curv_u8 = np.clip(np.round(curv * 255.0), 0, 255).astype(np.uint8)

    alpha = np.full((h, w), 255, dtype=np.uint8)
    return np.stack([slope_u8, ao_u8, curv_u8, alpha], axis=-1)


def export_pointcloud_ply(
    output_path: Path,
    depth_m: np.ndarray,
    albedo_rgb: np.ndarray,
    focallength_px: float,
    max_points: int = 300_000,
    binary: bool = True,
) -> int:
    """Unproject Depth Pro 2 metric depth (meters) + focal length into a colored 3D .ply point cloud.

    Uses pinhole camera unprojection:
        Z = depth_m
        X = (u - cx) * Z / f_px
        Y = -(v - cy) * Z / f_px

    Returns:
        Number of 3D vertices written to ``output_path``.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    h, w = depth.shape
    rgb = np.asarray(albedo_rgb, dtype=np.uint8)
    if rgb.shape[:2] != (h, w):
        rgb = np.asarray(
            Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

    f_px = float(focallength_px) if focallength_px and focallength_px > 1e-3 else float(max(w, h))

    stride = 1
    if max_points > 0 and (h * w) > max_points:
        stride = max(1, int(math.ceil(math.sqrt((h * w) / float(max_points)))))

    depth_s = depth[::stride, ::stride]
    rgb_s = rgb[::stride, ::stride]
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]

    valid = np.isfinite(depth_s) & (depth_s > 1e-4)
    if np.any(valid):
        p995 = float(np.percentile(depth_s[valid], 99.5))
        valid &= depth_s <= max(p995 * 2.5, 1.0)

    z = depth_s[valid].astype(np.float32)
    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    colors = rgb_s[valid]

    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    x = ((u - cx) * z / f_px).astype(np.float32)
    y = (-(v - cy) * z / f_px).astype(np.float32)

    count = int(z.size)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        "ply\n"
        f"format {fmt}\n"
        "comment Generated by Apple Depth Pro 2 -> Unity Terrain Pipeline\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    if binary:
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        verts = np.empty(count, dtype=vertex_dtype)
        verts["x"] = x
        verts["y"] = y
        verts["z"] = z
        verts["red"] = colors[:, 0]
        verts["green"] = colors[:, 1]
        verts["blue"] = colors[:, 2]
        with out.open("wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(verts.tobytes())
    else:
        with out.open("w", encoding="ascii") as fh:
            fh.write(header)
            for xi, yi, zi, r_c, g_c, b_c in zip(x, y, z, colors[:, 0], colors[:, 1], colors[:, 2]):
                fh.write(f"{xi:.4f} {yi:.4f} {zi:.4f} {int(r_c)} {int(g_c)} {int(b_c)}\n")

    return count


def depth_to_normalized_height(
    depth_m: np.ndarray,
    inverse_depth: np.ndarray,
    config: ExportConfig,
    guide_rgb: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Convert raw Depth Pro 2 predictions into a clean [0, 1] elevation map.

    For top-down imagery, objects/peaks closer to the camera have larger
    ``inverse_depth`` (smaller ``depth_m``), so ``use_inverse_depth=True`` maps
    higher terrain directly to brighter values (closer to 1.0).
    """
    finite_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if finite_depth.size == 0:
        finite_depth = np.array([1.0], dtype=np.float32)

    if config.use_inverse_depth:
        signal = np.nan_to_num(inverse_depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Linear depth inverted so smaller distance = higher elevation
        safe_d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, float(np.percentile(finite_depth, 99)))
        signal = -safe_d.astype(np.float32)

    lo = float(np.percentile(signal, config.percentile_low))
    hi = float(np.percentile(signal, config.percentile_high))
    if hi - lo < 1e-8:
        lo = float(signal.min())
        hi = float(signal.max())

    if hi - lo < 1e-8:
        norm = np.full_like(signal, 0.5, dtype=np.float32)
    else:
        norm = np.clip((signal - lo) / (hi - lo), 0.0, 1.0)

    if config.invert:
        norm = 1.0 - norm

    if config.guided_filter and guide_rgb is not None:
        norm = guided_filter_gray(
            guide_rgb,
            norm,
            radius=config.guided_radius,
            eps=config.guided_eps,
        )

    if config.smooth_sigma > 0.0:
        norm = gaussian_filter(norm, sigma=float(config.smooth_sigma)).astype(np.float32)
        norm = np.clip(norm, 0.0, 1.0)

    if abs(config.gamma - 1.0) > 1e-4:
        norm = np.power(np.clip(norm, 0.0, 1.0), float(config.gamma)).astype(np.float32)

    if config.water_clamp > 0.0:
        wc = float(np.clip(config.water_clamp, 0.0, 0.95))
        norm = np.where(norm <= wc, 0.0, (norm - wc) / (1.0 - wc)).astype(np.float32)

    stats = {
        "depthMinMeters": float(np.min(finite_depth)),
        "depthMaxMeters": float(np.max(finite_depth)),
        "depthP02Meters": float(np.percentile(finite_depth, 2.0)),
        "depthP50Meters": float(np.percentile(finite_depth, 50.0)),
        "depthP98Meters": float(np.percentile(finite_depth, 98.0)),
        "heightMin": float(norm.min()),
        "heightMax": float(norm.max()),
        "heightMean": float(norm.mean()),
    }
    return norm, stats


def colorize_height_turbo(norm_height: np.ndarray) -> np.ndarray:
    """Render a Turbo-colormapped RGB preview (red/warm = high/near, blue = low/far)."""
    idx = np.clip(np.round(norm_height * 255.0), 0, 255).astype(np.uint8)
    return TURBO_LUT[idx]


def resample_float_map(height_map: np.ndarray, target_res: int) -> np.ndarray:
    """Resample a float32 2D map to (target_res, target_res) with smooth bicubic filtering."""
    pil_f = Image.fromarray(height_map.astype(np.float32))
    resized = pil_f.resize((target_res, target_res), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def export_unity_terrain_bundle(
    output_dir: Path,
    albedo_rgb: np.ndarray,
    depth_m: np.ndarray,
    inverse_depth: np.ndarray,
    focallength_px: float,
    model_name: str,
    device: str,
    precision: str,
    inference_ms: float,
    source_type: str,
    source_label: str,
    config: Optional[ExportConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    """Write a complete Unity-ready terrain dataset bundle to ``output_dir``."""
    cfg = config or ExportConfig()
    res = nearest_unity_resolution(cfg.resolution)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    norm_full, stats = depth_to_normalized_height(depth_m, inverse_depth, cfg, guide_rgb=albedo_rgb)
    norm_res = resample_float_map(norm_full, res)
    u16 = np.clip(np.round(norm_res * 65535.0), 0, 65535).astype("<u2")

    albedo_path = out / "albedo.png"
    raw_path = out / "heightmap.raw"
    png16_path = out / "heightmap_16bit.png"
    preview_path = out / "depth_preview.png"
    meta_path = out / "metadata.json"
    npz_path: Optional[Path] = None
    masks_path: Optional[Path] = None
    ply_path: Optional[Path] = None
    ply_vertex_count = 0

    Image.fromarray(albedo_rgb.astype(np.uint8)).save(albedo_path)
    # 16-bit little-endian raw, row-major top-to-bottom (Unity flips Y to align North = +Z)
    raw_path.write_bytes(u16.tobytes())
    Image.fromarray(u16.astype(np.uint16)).save(png16_path)
    Image.fromarray(colorize_height_turbo(norm_full)).save(preview_path)

    extra = extra_metadata or {}
    ground_w = float(extra.get("groundWidthMeters", cfg.default_terrain_width_m))
    ground_h = float(extra.get("groundHeightMeters", cfg.default_terrain_width_m))
    recommended_size = float(extra.get("recommendedTerrainSize", cfg.default_terrain_width_m))
    recommended_height = float(extra.get("recommendedTerrainHeight", cfg.default_terrain_height_m))

    if cfg.export_masks:
        masks_path = out / "terrain_masks.png"
        masks_rgba = compute_terrain_masks(
            norm_res,
            terrain_size_m=recommended_size,
            terrain_height_m=recommended_height,
        )
        Image.fromarray(masks_rgba, mode="RGBA").save(masks_path)

    if cfg.export_ply:
        ply_path = out / "pointcloud.ply"
        ply_vertex_count = export_pointcloud_ply(
            output_path=ply_path,
            depth_m=depth_m,
            albedo_rgb=albedo_rgb,
            focallength_px=focallength_px,
            max_points=cfg.ply_max_points,
            binary=cfg.ply_binary,
        )

    if cfg.save_npz:
        npz_path = out / "depth_raw.npz"
        np.savez_compressed(
            npz_path,
            depth=depth_m.astype(np.float32),
            inverse_depth=inverse_depth.astype(np.float32),
            focallength_px=np.float32(focallength_px),
        )

    metadata: Dict[str, Any] = {
        "schemaVersion": 1,
        "sourceType": source_type,
        "sourceLabel": source_label,
        "modelName": model_name,
        "device": device,
        "precision": precision,
        "topDownMode": str(extra.get("topDownMode", "")),
        "guidedFilter": bool(cfg.guided_filter),
        "inferenceMs": round(float(inference_ms), 2),
        "focalLengthPx": round(float(focallength_px), 3),
        "imageWidth": int(albedo_rgb.shape[1]),
        "imageHeight": int(albedo_rgb.shape[0]),
        "rawResolution": int(res),
        "rawByteOrder": "little",
        "rawFlipY": True,
        "albedoFile": albedo_path.name,
        "heightmapRawFile": raw_path.name,
        "heightmapPng16File": png16_path.name,
        "depthPreviewFile": preview_path.name,
        "terrainMasksFile": masks_path.name if masks_path is not None else "",
        "pointCloudFile": ply_path.name if ply_path is not None else "",
        "pointCloudVertexCount": int(ply_vertex_count),
        "groundWidthMeters": round(ground_w, 2),
        "groundHeightMeters": round(ground_h, 2),
        "recommendedTerrainSize": round(recommended_size, 2),
        "recommendedTerrainHeight": round(recommended_height, 2),
        "recommendedWaterLevel": float(extra.get("recommendedWaterLevel", 0.08)),
        "latitude": float(extra.get("latitude", 0.0)),
        "longitude": float(extra.get("longitude", 0.0)),
        "zoom": int(extra.get("zoom", 0)),
        "mapProvider": str(extra.get("mapProvider", "")),
        "biome": str(extra.get("biome", "")),
        "seed": int(extra.get("seed", 0)),
        "videoFrameCount": int(extra.get("videoFrameCount", 0)),
        "keyframeIndex": int(extra.get("keyframeIndex", 0)),
        "exportConfig": asdict(cfg),
        **stats,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExportResult(
        output_dir=out,
        albedo_path=albedo_path,
        heightmap_raw_path=raw_path,
        heightmap_png16_path=png16_path,
        depth_preview_path=preview_path,
        metadata_path=meta_path,
        npz_path=npz_path,
        metadata=metadata,
        terrain_masks_path=masks_path,
        pointcloud_ply_path=ply_path,
    )
