"""Convert Depth Pro 2 metric/inverse depth arrays to Unity 6 Terrain assets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import gaussian_filter

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


def nearest_unity_resolution(requested: int) -> int:
    """Snap any requested resolution to a valid Unity TerrainData resolution (2^n + 1)."""
    if requested in VALID_UNITY_HEIGHTMAP_RESOLUTIONS:
        return requested
    return min(VALID_UNITY_HEIGHTMAP_RESOLUTIONS, key=lambda r: abs(r - requested))


def depth_to_normalized_height(
    depth_m: np.ndarray,
    inverse_depth: np.ndarray,
    config: ExportConfig,
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

    norm_full, stats = depth_to_normalized_height(depth_m, inverse_depth, cfg)
    norm_res = resample_float_map(norm_full, res)
    u16 = np.clip(np.round(norm_res * 65535.0), 0, 65535).astype("<u2")

    albedo_path = out / "albedo.png"
    raw_path = out / "heightmap.raw"
    png16_path = out / "heightmap_16bit.png"
    preview_path = out / "depth_preview.png"
    meta_path = out / "metadata.json"
    npz_path: Optional[Path] = None

    Image.fromarray(albedo_rgb.astype(np.uint8)).save(albedo_path)
    # 16-bit little-endian raw, row-major top-to-bottom (Unity flips Y to align North = +Z)
    raw_path.write_bytes(u16.tobytes())
    Image.fromarray(u16.astype(np.uint16)).save(png16_path)
    Image.fromarray(colorize_height_turbo(norm_full)).save(preview_path)

    if cfg.save_npz:
        npz_path = out / "depth_raw.npz"
        np.savez_compressed(
            npz_path,
            depth=depth_m.astype(np.float32),
            inverse_depth=inverse_depth.astype(np.float32),
            focallength_px=np.float32(focallength_px),
        )

    extra = extra_metadata or {}
    ground_w = float(extra.get("groundWidthMeters", cfg.default_terrain_width_m))
    ground_h = float(extra.get("groundHeightMeters", cfg.default_terrain_width_m))

    # Reasonable default Unity world footprint (kept around 500 units for comfortable camera inspection)
    recommended_size = float(extra.get("recommendedTerrainSize", cfg.default_terrain_width_m))
    recommended_height = float(extra.get("recommendedTerrainHeight", cfg.default_terrain_height_m))

    metadata: Dict[str, Any] = {
        "schemaVersion": 1,
        "sourceType": source_type,
        "sourceLabel": source_label,
        "modelName": model_name,
        "device": device,
        "precision": precision,
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
    )
