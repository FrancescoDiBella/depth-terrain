"""Automated tests for the Apple Depth Pro 2 -> Unity Terrain pipeline."""

from __future__ import annotations

import json
import math
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

import depth_terrain.depth_runner as depth_runner_mod
import depth_terrain.server as server_mod
from depth_terrain.cli import main as cli_main
from depth_terrain.depth_runner import (
    DepthInferenceResult,
    clear_model_cache,
    debias_positive_radial_vignette,
    detrend_linear_plane,
    get_cached_model,
    get_model_info,
    infer_image_depth,
    list_cached_models,
    select_best_video_keyframe,
)
from depth_terrain.fictional_generator import FICTIONAL_BIOMES, generate_fictional_topdown
from depth_terrain.heightmap_export import (
    ExportConfig,
    compute_terrain_masks,
    depth_to_normalized_height,
    export_pointcloud_ply,
    export_unity_terrain_bundle,
    guided_filter_gray,
    nearest_unity_resolution,
)
from depth_terrain.map_fetcher import MapFetchResult, latlon_to_tile_float, meters_per_pixel
from depth_terrain.server import create_server


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
    assert exported.terrain_masks_path is not None and exported.terrain_masks_path.exists()
    assert exported.pointcloud_ply_path is None

    raw_bytes = exported.heightmap_raw_path.read_bytes()
    assert len(raw_bytes) == 129 * 129 * 2
    raw_u16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(129, 129)
    assert raw_u16[64, 64] > 60000
    assert raw_u16[0, 0] < 3000

    meta = json.loads(exported.metadata_path.read_text(encoding="utf-8"))
    assert meta["rawResolution"] == 129
    assert meta["rawByteOrder"] == "little"
    assert meta["rawFlipY"] is True
    assert meta["terrainMasksFile"] == "terrain_masks.png"
    assert meta["pointCloudFile"] == ""
    assert Image.open(exported.heightmap_png16_path).size == (129, 129)
    assert Image.open(exported.terrain_masks_path).size == (129, 129)


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


def test_guided_filter_preserves_edges_and_reduces_flat_noise() -> None:
    rng = np.random.default_rng(7)
    h, w = 128, 128
    # Sharp step edge in RGB guide (e.g. cliff / building roof boundary)
    guide_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    guide_rgb[:, w // 2 :] = 220
    guide_rgb[:, : w // 2] = 30

    clean = np.zeros((h, w), dtype=np.float32)
    clean[:, w // 2 :] = 0.85
    clean[:, : w // 2] = 0.15
    noisy = np.clip(clean + rng.normal(0.0, 0.08, size=(h, w)).astype(np.float32), 0.0, 1.0)

    filtered = guided_filter_gray(guide_rgb, noisy, radius=6, eps=1e-3)
    assert filtered.shape == (h, w)
    assert filtered.dtype == np.float32
    assert float(filtered.min()) >= 0.0
    assert float(filtered.max()) <= 1.0

    # Flat-region noise must drop substantially
    assert float(filtered[:, :40].std()) < float(noisy[:, :40].std()) * 0.45
    assert float(filtered[:, 88:].std()) < float(noisy[:, 88:].std()) * 0.45
    # Edge step across column w//2 must remain crisp (> 0.55 jump across 2 pixels)
    edge_jump = float(filtered[:, w // 2 + 1].mean() - filtered[:, w // 2 - 2].mean())
    assert edge_jump > 0.55


def test_terrain_masks_rgba_channels() -> None:
    h, w = 129, 129
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / (w - 1)) * 2.0 - 1.0
    yn = (yy / (h - 1)) * 2.0 - 1.0

    # Peak on left, deep pit on right, flat border
    peak = 0.45 * np.exp(-((xn + 0.4) ** 2 + yn**2) * 18.0)
    pit = -0.40 * np.exp(-((xn - 0.4) ** 2 + yn**2) * 18.0)
    height = np.clip(0.5 + peak + pit, 0.0, 1.0)

    masks = compute_terrain_masks(height, terrain_size_m=500.0, terrain_height_m=150.0)
    assert masks.shape == (h, w, 4)
    assert masks.dtype == np.uint8
    assert np.all(masks[..., 3] == 255)

    # Steep flank of peak has much higher slope (R) than flat corner
    assert int(masks[64, 30, 0]) > int(masks[2, 2, 0]) + 40
    # Pit center has darker ambient occlusion (G) than peak summit
    assert int(masks[64, 90, 1]) < int(masks[64, 38, 1]) - 30
    # Peak summit has convex curvature (B > 128), pit bottom has concave curvature (B < 128)
    assert int(masks[64, 38, 2]) > 150
    assert int(masks[64, 90, 2]) < 110


def test_export_pointcloud_ply_binary_and_ascii(tmp_path: Path) -> None:
    h, w = 64, 64
    depth_m = np.full((h, w), 10.0, dtype=np.float32)
    depth_m[0, 0] = np.nan  # Must be filtered out
    albedo = np.zeros((h, w, 3), dtype=np.uint8)
    albedo[..., 0] = 200
    albedo[..., 1] = 100
    albedo[..., 2] = 50

    bin_path = tmp_path / "cloud_bin.ply"
    count_bin = export_pointcloud_ply(bin_path, depth_m, albedo, focallength_px=500.0, max_points=0, binary=True)
    assert count_bin == h * w - 1
    raw = bin_path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header_text = raw[:header_end].decode("ascii")
    assert "format binary_little_endian 1.0" in header_text
    assert f"element vertex {count_bin}" in header_text
    payload = raw[header_end:]
    assert len(payload) == count_bin * 15  # 3*float32 (12) + 3*uint8 (3)
    x0, y0, z0, r0, g0, b0 = struct.unpack("<fffBBB", payload[:15])
    assert z0 == pytest.approx(10.0, abs=1e-4)
    assert (r0, g0, b0) == (200, 100, 50)

    asc_path = tmp_path / "cloud_asc.ply"
    count_asc = export_pointcloud_ply(asc_path, depth_m, albedo, focallength_px=500.0, max_points=1024, binary=False)
    assert 0 < count_asc <= 1024
    lines = asc_path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "ply"
    assert "format ascii 1.0" in lines[1]

    # Full bundle export with --export-ply + --guided-filter
    inv_depth = 500.0 / (w * np.nan_to_num(depth_m, nan=10.0))
    cfg = ExportConfig(resolution=65, guided_filter=True, export_ply=True, ply_max_points=1024)
    res = export_unity_terrain_bundle(
        output_dir=tmp_path / "bundle_ply",
        albedo_rgb=albedo,
        depth_m=np.nan_to_num(depth_m, nan=10.0),
        inverse_depth=inv_depth,
        focallength_px=500.0,
        model_name="depthpro2-b",
        device="mps",
        precision="bfloat16",
        inference_ms=15.0,
        source_type="fictional_topdown",
        source_label="PLY Test",
        config=cfg,
        extra_metadata={"topDownMode": "fast_multiscale"},
    )
    assert res.pointcloud_ply_path is not None and res.pointcloud_ply_path.exists()
    assert res.metadata["pointCloudFile"] == "pointcloud.ply"
    assert res.metadata["pointCloudVertexCount"] > 0
    assert res.metadata["guidedFilter"] is True
    assert res.metadata["topDownMode"] == "fast_multiscale"


class _SpyDepthPro2:
    """Minimal mock DepthPro2 tracking FOV (`infer`) vs FOV-free (`predict_inverse_depth`) passes."""

    precision = torch.bfloat16

    def __init__(self) -> None:
        self.infer_calls = 0
        self.predict_inv_calls = 0

    def predict_inverse_depth(self, x: torch.Tensor) -> torch.Tensor:
        self.predict_inv_calls += int(x.shape[0])
        # Deterministic structured depth signal from image luminance
        lum = x.mean(dim=1)
        return 0.01 + 0.005 * (lum - lum.min())

    def infer(self, x: torch.Tensor, f_px=None) -> dict:
        self.infer_calls += 1
        inv = self.predict_inverse_depth(x.unsqueeze(0))[0]
        # Adjust counter so infer_calls tracks FOV passes and predict_inv_calls tracks direct calls
        self.predict_inv_calls -= 1
        focal = torch.tensor(float(f_px if f_px is not None else x.shape[-1]), dtype=torch.float32)
        depth = focal / torch.clamp(x.shape[-1] * inv, min=1e-6)
        return {"inverse_depth": inv, "depth": depth, "focallength_px": focal}


def _dummy_transform(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


@pytest.mark.parametrize(
    ("mode", "f_px", "expected_fov_calls", "expected_direct_calls"),
    [
        ("raw", None, 1, 0),
        ("raw", 750.0, 0, 1),
        ("tta4", None, 1, 3),
        ("fast_multiscale", None, 1, 5),  # 2 global (1 primary + 1 rot180) + 2x2 patches = 6 total
        ("fast_multiscale", 750.0, 0, 6),
        ("multiscale_tta", None, 1, 21),  # 4 global + 9*2 patches = 22 total, only 1 FOV!
    ],
)
def test_fov_elimination_and_fast_multiscale_pass_counts(
    mode: str,
    f_px: float | None,
    expected_fov_calls: int,
    expected_direct_calls: int,
) -> None:
    spy = _SpyDepthPro2()
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)

    res = infer_image_depth(
        img,
        model_name="depthpro2-b",
        device_name="cpu",
        f_px=f_px,
        topdown_mode=mode,
        loaded_bundle=(spy, _dummy_transform, torch.device("cpu")),
    )
    assert spy.infer_calls == expected_fov_calls
    assert spy.predict_inv_calls == expected_direct_calls
    assert res.depth_m.shape == (512, 512)
    assert res.inverse_depth.shape == (512, 512)
    assert res.topdown_mode == mode
    assert np.all(np.isfinite(res.inverse_depth))


def test_model_lru_cache_hits_and_eviction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_model_cache()
    load_calls: list[tuple[str, str, str]] = []

    def fake_load(model_name, device_name="auto", precision_name=None, checkpoints_dir=None):
        load_calls.append((model_name, str(device_name), str(precision_name)))
        return object(), _dummy_transform, torch.device("cpu")

    monkeypatch.setattr(depth_runner_mod, "load_depth_pro_2", fake_load)

    b1 = get_cached_model("depthpro2-s", device="cpu", precision="fp32", checkpoints_dir=tmp_path, max_cache_size=2)
    b2 = get_cached_model("depthpro2-s", device="cpu", precision="fp32", checkpoints_dir=tmp_path, max_cache_size=2)
    assert b1 is b2
    assert len(load_calls) == 1
    assert list_cached_models() == ["depthpro2-s"]

    get_cached_model("depthpro2-b", device="cpu", precision="fp32", checkpoints_dir=tmp_path, max_cache_size=2)
    assert list_cached_models() == ["depthpro2-s", "depthpro2-b"]

    # Adding a 3rd model evicts the LRU ("depthpro2-s")
    get_cached_model("depthpro2-l", device="cpu", precision="fp32", checkpoints_dir=tmp_path, max_cache_size=2)
    assert list_cached_models() == ["depthpro2-b", "depthpro2-l"]
    clear_model_cache()
    assert list_cached_models() == []


def test_smart_video_keyframe_selection() -> None:
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Frame 0: sharp checkerboard + strong topographic relief -> MUST win over blurry midpoint!
    sharp_rgb = (((xx.astype(int) // 4 + yy.astype(int) // 4) % 2) * 240).astype(np.uint8)
    sharp_rgb = np.stack([sharp_rgb] * 3, axis=-1)
    rich_inv = np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / 180.0).astype(np.float32)

    # Frame 1 (midpoint) & Frame 2: blurry low-contrast flat frames
    blur_rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    flat_inv = np.full((h, w), 0.5, dtype=np.float32)

    def _mk(inv: np.ndarray) -> DepthInferenceResult:
        return DepthInferenceResult(
            depth_m=np.full((h, w), 50.0, dtype=np.float32),
            inverse_depth=inv,
            focallength_px=500.0,
            model_name="depthpro2-s",
            device="cpu",
            precision="float32",
            inference_ms=10.0,
        )

    results = [
        (0, sharp_rgb, _mk(rich_inv)),
        (5, blur_rgb, _mk(flat_inv)),
        (10, blur_rgb, _mk(flat_inv)),
    ]
    assert select_best_video_keyframe(results) == 0


def test_model_info_and_cli_commands() -> None:
    info = get_model_info()
    assert "platform" in info
    assert "acceleration" in info
    assert "checkpoints" in info
    assert info["coreml"]["repoId"] == "apple/DepthPro2-coreml"

    runner = CliRunner()
    res = runner.invoke(cli_main, ["model-info", "--json"])
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["coreml"]["repoId"] == "apple/DepthPro2-coreml"

    help_res = runner.invoke(cli_main, ["from-fictional", "--help"])
    assert help_res.exit_code == 0
    assert "fast_multiscale" in help_res.output
    assert "--guided-filter" in help_res.output
    assert "--export-ply" in help_res.output

    serve_help = runner.invoke(cli_main, ["serve", "--help"])
    assert serve_help.exit_code == 0
    assert "--port" in serve_help.output


def _http_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_http_server_health_and_infer_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spy = _SpyDepthPro2()

    def fake_infer_image(
        image_rgb: np.ndarray,
        model_name: str = "depthpro2-b",
        device_name: str = "cpu",
        precision_name: str | None = None,
        f_px: float | None = None,
        topdown_mode: str = "fast_multiscale",
        checkpoints_dir: Path | None = None,
        loaded_bundle=None,
    ) -> DepthInferenceResult:
        return infer_image_depth(
            image_rgb,
            model_name=model_name,
            device_name="cpu",
            f_px=f_px,
            topdown_mode=topdown_mode,
            loaded_bundle=(spy, _dummy_transform, torch.device("cpu")),
        )

    def fake_fetch_map(lat, lon, zoom, grid_size, provider, custom_url=None, place_label=""):
        img = generate_fictional_topdown("canyon_mesa", size=128, seed=9).image_rgb
        return MapFetchResult(
            image_rgb=img,
            lat=lat,
            lon=lon,
            zoom=zoom,
            grid_size=grid_size,
            provider=provider,
            ground_width_m=800.0,
            ground_height_m=800.0,
            meters_per_pixel=6.25,
            place_label=place_label or f"{lat:.2f},{lon:.2f}",
        )

    def fake_infer_video(**kwargs):
        img0 = generate_fictional_topdown("alpine_peaks", size=64, seed=1).image_rgb
        img1 = np.full((64, 64, 3), 120, dtype=np.uint8)
        res0 = fake_infer_image(img0, model_name=kwargs.get("model_name", "depthpro2-s"), topdown_mode="raw")
        res1 = fake_infer_image(img1, model_name=kwargs.get("model_name", "depthpro2-s"), topdown_mode="raw")
        return iter([(0, img0, res0), (5, img1, res1)])

    monkeypatch.setattr(server_mod, "fetch_map_region", fake_fetch_map)

    srv = create_server(
        host="127.0.0.1",
        port=0,
        model_name="depthpro2-s",
        device_name="cpu",
        preload=False,
        warmup=False,
        infer_image_fn=fake_infer_image,
        infer_video_fn=fake_infer_video,
    )
    host, port = srv.server_address
    base = f"http://{host}:{port}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. GET /health
        status, health = _http_json(f"{base}/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["model"] == "depthpro2-s"
        assert health["device"] == "cpu"
        assert health["requestsServed"] == 0
        assert isinstance(health["cachedModels"], list)
        assert health["uptimeSeconds"] >= 0.0

        # 2. POST /infer/fictional with guidedFilter + exportPly + fast_multiscale
        fict_out = tmp_path / "http_fictional"
        status, body = _http_json(
            f"{base}/infer/fictional",
            {
                "outputDir": str(fict_out),
                "biome": "volcanic_island",
                "seed": 42,
                "size": 128,
                "model": "depthpro2-s",
                "topDownMode": "fast_multiscale",
                "resolution": 129,
                "guidedFilter": True,
                "exportPly": True,
            },
        )
        assert status == 200
        assert body["status"] == "ok"
        assert Path(body["outputDir"]) == fict_out.resolve()
        assert body["elapsedMs"] >= 0.0
        assert body["metadata"]["guidedFilter"] is True
        assert body["metadata"]["terrainMasksFile"] == "terrain_masks.png"
        assert body["metadata"]["pointCloudFile"] == "pointcloud.ply"
        assert (fict_out / "albedo.png").exists()
        assert (fict_out / "heightmap.raw").exists()
        assert (fict_out / "terrain_masks.png").exists()
        assert (fict_out / "pointcloud.ply").exists()
        assert (fict_out / "metadata.json").exists()

        # 3. POST /infer/image
        img_file = tmp_path / "sample.png"
        Image.fromarray(generate_fictional_topdown("alpine_peaks", size=128, seed=3).image_rgb).save(img_file)
        img_out = tmp_path / "http_image"
        status, body = _http_json(
            f"{base}/infer/image",
            {
                "outputDir": str(img_out),
                "inputPath": str(img_file),
                "label": "Test Alpine",
                "topDownMode": "raw",
                "resolution": 65,
            },
        )
        assert status == 200
        assert body["metadata"]["sourceType"] == "local_image"
        assert body["metadata"]["sourceLabel"] == "Test Alpine"

        # 4. POST /infer/map
        map_out = tmp_path / "http_map"
        status, body = _http_json(
            f"{base}/infer/map",
            {
                "outputDir": str(map_out),
                "useCoordinates": True,
                "latitude": 36.1069,
                "longitude": -112.1129,
                "zoom": 13,
                "grid": 2,
                "provider": "esri_satellite",
                "topDownMode": "raw",
                "resolution": 65,
            },
        )
        assert status == 200
        assert body["metadata"]["sourceType"] == "real_map"
        assert body["metadata"]["latitude"] == pytest.approx(36.1069, abs=1e-3)

        # 5. POST /infer/video
        dummy_video = tmp_path / "drone.mp4"
        dummy_video.write_bytes(b"fake-mp4")
        vid_out = tmp_path / "http_video"
        status, body = _http_json(
            f"{base}/infer/video",
            {
                "outputDir": str(vid_out),
                "inputPath": str(dummy_video),
                "mode": "unbuffered",
                "maxFrames": 2,
                "stride": 1,
                "resolution": 65,
            },
        )
        assert status == 200
        assert body["metadata"]["sourceType"] == "aerial_video"
        assert body["metadata"]["videoFrameCount"] == 2
        assert body["metadata"]["keyframeIndex"] == 0
        assert (vid_out / "frames" / "frame_0000" / "metadata.json").exists()

        # 6. Error handling (missing inputPath -> 400, unknown route -> 404)
        status, err = _http_json(f"{base}/infer/image", {"outputDir": str(tmp_path / "bad")})
        assert status == 400
        assert err["status"] == "error"

        status, err = _http_json(f"{base}/infer/does_not_exist", {})
        assert status == 404
        assert err["status"] == "error"

        # 7. Health counter reflects 4 successful inference calls
        status, health2 = _http_json(f"{base}/health")
        assert status == 200
        assert health2["requestsServed"] == 4
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
