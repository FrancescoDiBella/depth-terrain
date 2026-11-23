"""Fast unit tests for the Apple Depth Pro 2 benchmark & efficiency suite.

Tests quantitative elevation accuracy metrics, synthetic ground-truth topography,
temporal stability metrics, benchmark JSON schema validation, and chart rendering
without invoking slow full-model inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from depth_terrain.benchmark import (
    SCHEMA_VERSION,
    collect_hardware_info,
    compute_accuracy_metrics,
    compute_temporal_stability_metrics,
    generate_charts,
    generate_synthetic_pan_video,
    generate_synthetic_topography,
    validate_benchmark_payload,
)


def test_synthetic_topography_determinism_and_shape() -> None:
    gt1, rgb1 = generate_synthetic_topography(size=128, seed=2025)
    gt2, rgb2 = generate_synthetic_topography(size=128, seed=2025)

    assert gt1.shape == (128, 128)
    assert gt1.dtype == np.float32
    assert rgb1.shape == (128, 128, 3)
    assert rgb1.dtype == np.uint8
    assert np.array_equal(gt1, gt2)
    assert np.array_equal(rgb1, rgb2)
    assert float(gt1.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(gt1.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(rgb1.std()) > 20.0

    # Rectangular strip for aerial panning video
    gt_strip, rgb_strip = generate_synthetic_topography(size=96, width=160, seed=7)
    assert gt_strip.shape == (96, 160)
    assert rgb_strip.shape == (96, 160, 3)


def test_accuracy_metrics_exact_affine_inverted_and_noisy() -> None:
    gt, _ = generate_synthetic_topography(size=128, seed=42)

    # 1. Exact identity match (slight p1/p99 clip can cause tiny numerical diff, still > 0.995)
    exact = compute_accuracy_metrics(gt, gt)
    assert exact["pearson_r"] > 0.998
    assert exact["spearman_rho"] > 0.998
    assert exact["nrmse"] < 0.01
    assert exact["sobel_edge_alignment"] > 0.995
    assert exact["composite_quality"] > 0.99

    # 2. Positive affine transformation (monocular depth scale/shift invariance)
    affine = compute_accuracy_metrics(3.7 * gt + 12.5, gt)
    assert affine["pearson_r"] == pytest.approx(exact["pearson_r"], abs=1e-3)
    assert affine["spearman_rho"] == pytest.approx(exact["spearman_rho"], abs=1e-3)
    assert affine["nrmse"] == pytest.approx(exact["nrmse"], abs=1e-3)
    assert affine["sobel_edge_alignment"] == pytest.approx(exact["sobel_edge_alignment"], abs=1e-3)

    # 3. Inverted elevation polarity
    inv = compute_accuracy_metrics(1.0 - gt, gt)
    assert inv["pearson_r"] < -0.99
    assert inv["spearman_rho"] < -0.99

    # 4. Degraded noisy prediction has worse metrics than exact
    rng = np.random.default_rng(0)
    noisy = gt + rng.normal(0.0, 0.25, size=gt.shape).astype(np.float32)
    deg = compute_accuracy_metrics(noisy, gt)
    assert deg["pearson_r"] < exact["pearson_r"] - 0.10
    assert deg["nrmse"] > exact["nrmse"] + 0.05
    assert deg["sobel_edge_alignment"] < exact["sobel_edge_alignment"] - 0.15


def test_accuracy_metrics_handles_constant_and_nans() -> None:
    gt, _ = generate_synthetic_topography(size=64, seed=1)
    const = np.full_like(gt, 0.5)
    const[0, 0] = np.nan
    const[0, 1] = np.inf
    res = compute_accuracy_metrics(const, gt)
    assert res["pearson_r"] == 0.0
    assert res["spearman_rho"] == 0.0
    assert np.isfinite(res["nrmse"])
    assert res["sobel_edge_alignment"] == 0.0


def test_temporal_stability_metrics_zero_jitter_vs_flicker() -> None:
    gt_strip, _ = generate_synthetic_topography(size=96, width=160, seed=99)
    dx = 8
    stable_frames = [gt_strip[:, i * dx : i * dx + 96] for i in range(6)]
    stab = compute_temporal_stability_metrics(stable_frames, pan_dx_px=dx)
    assert stab["num_frames"] == 6
    assert stab["overlap_mad"] < 0.03
    assert stab["overlap_pearson_r"] > 0.99

    # Inject alternating flicker noise across frames
    rng = np.random.default_rng(123)
    jittery_frames = [
        f + rng.normal(0.0, 0.18, size=f.shape).astype(np.float32) + (0.25 if i % 2 else 0.0)
        for i, f in enumerate(stable_frames)
    ]
    jit = compute_temporal_stability_metrics(jittery_frames, pan_dx_px=dx)
    assert jit["overlap_mad"] > stab["overlap_mad"] + 0.08
    assert jit["overlap_rmse"] > stab["overlap_rmse"] + 0.10
    assert jit["overlap_pearson_r"] < stab["overlap_pearson_r"] - 0.15


def test_generate_synthetic_pan_video_decodes_cleanly(tmp_path: Path) -> None:
    vid_path = tmp_path / "pan_test.mp4"
    out_path, gts = generate_synthetic_pan_video(vid_path, num_frames=5, frame_size=96, pan_dx_px=8, seed=11)
    assert out_path.exists()
    assert out_path.stat().st_size > 1024
    assert len(gts) == 5

    decoded = 0
    container = av.open(str(out_path))
    try:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            assert arr.shape == (96, 96, 3)
            decoded += 1
    finally:
        container.close()
    assert decoded == 5


def _make_mock_benchmark_payload() -> dict:
    acc_ok = {
        "pearson_r": 0.65,
        "spearman_rho": 0.72,
        "nrmse": 0.18,
        "sobel_edge_alignment": 0.35,
        "composite_quality": 0.6515,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": "2026-11-22T00:00:00Z",
        "hardware": collect_hardware_info(),
        "baseline_file": "baseline.json",
        "precision_matrix": [
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 768, "wall_ms": 220.0, "mps_allocated_mb": 180.0, "mps_driver_mb": 1200.0, "accuracy": acc_ok},
            {"model": "depthpro2-s", "device": "cpu", "precision": "fp32", "resolution": 768, "wall_ms": 2400.0, "mps_allocated_mb": 0.0, "mps_driver_mb": 0.0, "accuracy": acc_ok},
            {"model": "depthpro2-b", "device": "mps", "precision": "bf16", "resolution": 768, "wall_ms": 480.0, "mps_allocated_mb": 460.0, "mps_driver_mb": 1450.0, "accuracy": acc_ok},
        ],
        "resolution_scaling": [
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 384, "megapixels": 0.1475, "wall_ms": 110.0, "mp_per_sec": 1.34, "mps_allocated_mb": 180.0, "mps_driver_mb": 900.0, "accuracy": acc_ok},
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 768, "megapixels": 0.5898, "wall_ms": 220.0, "mp_per_sec": 2.68, "mps_allocated_mb": 185.0, "mps_driver_mb": 1200.0, "accuracy": acc_ok},
            {"model": "depthpro2-b", "device": "mps", "precision": "bf16", "resolution": 768, "megapixels": 0.5898, "wall_ms": 480.0, "mp_per_sec": 1.23, "mps_allocated_mb": 460.0, "mps_driver_mb": 1450.0, "accuracy": acc_ok},
        ],
        "topdown_modes": [
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 768, "mode": "raw", "guided_filter": False, "wall_ms": 220.0, "baseline_wall_ms": 247.0, "speedup_vs_baseline": 1.12, "accuracy": acc_ok},
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 768, "mode": "fast_multiscale", "guided_filter": False, "wall_ms": 720.0, "baseline_wall_ms": None, "speedup_vs_baseline": None, "accuracy": acc_ok},
            {"model": "depthpro2-s", "device": "mps", "precision": "bf16", "resolution": 768, "mode": "multiscale_tta", "guided_filter": False, "wall_ms": 1600.0, "baseline_wall_ms": 3038.0, "speedup_vs_baseline": 1.90, "accuracy": acc_ok},
            {"model": "depthpro2-b", "device": "mps", "precision": "bf16", "resolution": 768, "mode": "raw", "guided_filter": False, "wall_ms": 480.0, "baseline_wall_ms": 525.0, "speedup_vs_baseline": 1.09, "accuracy": acc_ok},
            {"model": "depthpro2-b", "device": "mps", "precision": "bf16", "resolution": 768, "mode": "multiscale_tta", "guided_filter": True, "wall_ms": 2650.0, "baseline_wall_ms": 4923.0, "speedup_vs_baseline": 1.86, "accuracy": acc_ok},
        ],
        "cold_vs_warm": {
            "resolution": 768,
            "topdown_mode": "fast_multiscale",
            "comparisons": [
                {"model": "depthpro2-s", "cold_cli_wall_ms": 3800.0, "warm_http_median_ms": 750.0, "warm_speedup_vs_cold": 5.07},
                {"model": "depthpro2-b", "cold_cli_wall_ms": 5400.0, "warm_http_median_ms": 1500.0, "warm_speedup_vs_cold": 3.60},
            ],
        },
        "video_stability": {
            "model": "depthpro2-s",
            "num_frames": 12,
            "strategies": [
                {"strategy": "infer_video_buffered", "per_frame_ms": 180.0, "stability": {"overlap_mad": 0.018}},
                {"strategy": "infer_video_unbuffered", "per_frame_ms": 155.0, "stability": {"overlap_mad": 0.024}},
                {"strategy": "single_image_raw", "per_frame_ms": 210.0, "stability": {"overlap_mad": 0.049}},
            ],
        },
    }


def test_benchmark_payload_schema_validation_and_chart_generation(tmp_path: Path) -> None:
    payload = _make_mock_benchmark_payload()
    errors = validate_benchmark_payload(payload)
    assert errors == []

    # Verify JSON serializability
    serialized = json.dumps(payload)
    assert len(serialized) > 500

    # Verify invalid payload catches errors
    bad = {"schema_version": 1}
    bad_errors = validate_benchmark_payload(bad)
    assert len(bad_errors) >= 4

    # Verify all 4 charts render cleanly and are non-empty PNGs
    charts = generate_charts(payload, tmp_path)
    assert len(charts) == 4
    expected_names = {
        "latency_before_after.png",
        "resolution_scaling.png",
        "speed_vs_quality_pareto.png",
        "memory_server_video.png",
    }
    assert {c.name for c in charts} == expected_names
    for chart in charts:
        assert chart.exists()
        assert chart.stat().st_size > 10_000
