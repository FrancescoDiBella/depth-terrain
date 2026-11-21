"""Automated tests for the Apple Depth Pro 2 -> Unity Terrain pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth_terrain.depth_runner import debias_positive_radial_vignette, detrend_linear_plane
from depth_terrain.fictional_generator import FICTIONAL_BIOMES, generate_fictional_topdown
from depth_terrain.heightmap_export import (
    ExportConfig,
    depth_to_normalized_height,
    export_unity_terrain_bundle,
    nearest_unity_resolution,
)
from depth_terrain.map_fetcher import latlon_to_tile_float, meters_per_pixel


def test_web_mercator_tile_coordinates_vesuvius() -> None:
    fx, fy = latlon_to_tile_float(40.8218, 14.4260, zoom=14)
    assert 8840 < fx < 8860
    assert 6150 < fy < 6160
    mpp = meters_per_pixel(40.8218, zoom=14)
    # Expected ~7.23 m/px at 40.8N zoom 14
    assert math.isclose(mpp, 7.23, rel_tol=0.05)


@pytest.mark.parametrize("biome", FICTIONAL_BIOMES)
def test_fictional_generator_produces_valid_rgb(biome: str) -> None:
    res = generate_fictional_topdown(biome=biome, size=256, seed=123)
    assert res.image_rgb.shape == (256, 256, 3)
    assert res.image_rgb.dtype == np.uint8
    # Must have non-trivial visual contrast
    assert float(res.image_rgb.std()) > 15.0


def test_nearest_unity_resolution() -> None:
    assert nearest_unity_resolution(513) == 513
    assert nearest_unity_resolution(500) == 513
    assert nearest_unity_resolution(256) == 257
    assert nearest_unity_resolution(1024) == 1025


def test_heightmap_export_polarity_and_raw16_roundtrip(tmp_path: Path) -> None:
    # Center peak is closer to camera (smaller depth -> larger inverse depth)
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot((xx - 64.0) / 64.0, (yy - 64.0) / 64.0)
    depth_m = 100.0 + 50.0 * dist  # Center = 100m (peak), edges = ~170m (valley)
    inv_depth = 1000.0 / (w * depth_m)
    albedo = np.full((h, w, 3), 120, dtype=np.uint8)

    cfg = ExportConfig(resolution=129, use_inverse_depth=True, smooth_sigma=0.0, percentile_low=0.0, percentile_high=100.0)
    norm, stats = depth_to_normalized_height(depth_m, inv_depth, cfg)
    assert norm[64, 64] > 0.95  # Peak at center is highest!
    assert norm[0, 0] < 0.05  # Corners are lowest!
    assert stats["depthMinMeters"] == pytest.approx(100.0, abs=0.5)

    exported = export_unity_terrain_bundle(
        output_dir=tmp_path / "bundle",
        albedo_rgb=albedo,
        depth_m=depth_m,
        inverse_depth=inv_depth,
        focallength_px=1000.0,
        model_name="depthpro2-s",
        device="mps",
        precision="bf16",
        inference_ms=42.5,
        source_type="fictional_topdown",
        source_label="Unit Test Peak",
        config=cfg,
    )

    assert exported.albedo_path.exists()
    assert exported.heightmap_raw_path.exists()
    assert exported.heightmap_png16_path.exists()
    assert exported.depth_preview_path.exists()
    assert exported.metadata_path.exists()

    raw_bytes = exported.heightmap_raw_path.read_bytes()
    assert len(raw_bytes) == 129 * 129 * 2
    raw_u16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(129, 129)
    assert raw_u16[64, 64] > 60000
    assert raw_u16[0, 0] < 3000

    meta = json.loads(exported.metadata_path.read_text(encoding="utf-8"))
    assert meta["rawResolution"] == 129
    assert meta["rawByteOrder"] == "little"
    assert meta["rawFlipY"] is True
    assert Image.open(exported.heightmap_png16_path).size == (129, 129)


def test_linear_plane_detrend_and_radial_debias() -> None:
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / (w - 1)) * 2.0 - 1.0
    yn = (yy / (h - 1)) * 2.0 - 1.0

    # Pure tilted plane must collapse to ~0
    tilted = 2.5 * xn - 1.7 * yn + 4.0
    flat = detrend_linear_plane(tilted)
    assert float(np.max(np.abs(flat))) < 1e-4

    # Positive corner lift (r^2) is removed, while central mountain (-r^2) is preserved
    corner_lift = 1.5 * (xn**2 + yn**2)
    debiased = debias_positive_radial_vignette(corner_lift)
    assert float(np.max(np.abs(debiased))) < 1e-4

    central_peak = np.exp(-(xn**2 + yn**2) * 3.0)
    preserved = debias_positive_radial_vignette(central_peak)
    assert float(preserved[32, 32] - preserved[0, 0]) > 0.8

