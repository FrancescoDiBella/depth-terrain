"""Command-line interface for the Apple Depth Pro 2 -> Unity 6 Terrain pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click

from .depth_runner import (
    PRECISIONS,
    TOPDOWN_MODES,
    VALID_MODELS,
    download_model_weights,
    infer_image_depth,
    infer_video_depth,
    load_local_image,
)
from .fictional_generator import FICTIONAL_BIOMES, generate_fictional_topdown
from .heightmap_export import (
    VALID_UNITY_HEIGHTMAP_RESOLUTIONS,
    ExportConfig,
    export_unity_terrain_bundle,
)
from .map_fetcher import TILE_PROVIDERS, fetch_map_region, geocode_place


def _common_model_options(fn):
    options = [
        click.option(
            "-m",
            "--model",
            type=click.Choice(VALID_MODELS),
            default="depthpro2-b",
            show_default=True,
            help="Apple Depth Pro 2 model size (s=97MB fast, b=386MB balanced, l=1.3GB, h=3.5GB).",
        ),
        click.option(
            "--device",
            type=click.Choice(["auto", "mps", "cuda", "cpu"]),
            default="auto",
            show_default=True,
            help="Compute device for PyTorch inference.",
        ),
        click.option(
            "--precision",
            type=click.Choice(sorted(PRECISIONS.keys())),
            default=None,
            help="Autocast precision override (defaults to bf16 on GPU/MPS, fp32 on CPU).",
        ),
        click.option(
            "--topdown-mode",
            type=click.Choice(TOPDOWN_MODES),
            default="multiscale_tta",
            show_default=True,
            help="Top-down nadir adaptation: multiscale_tta (sharpest), tta4 (fast), or raw.",
        ),
        click.option(
            "--resolution",
            type=click.Choice([str(r) for r in VALID_UNITY_HEIGHTMAP_RESOLUTIONS]),
            default="513",
            show_default=True,
            help="Unity TerrainData heightmap resolution (2^n + 1).",
        ),
        click.option("--invert/--no-invert", default=False, help="Invert height polarity."),
        click.option("--smooth-sigma", type=float, default=1.0, show_default=True, help="Gaussian smoothing sigma."),
        click.option("--gamma", type=float, default=1.0, show_default=True, help="Height curve gamma."),
        click.option("--p-low", type=float, default=2.0, show_default=True, help="Lower percentile clip."),
        click.option("--p-high", type=float, default=98.0, show_default=True, help="Upper percentile clip."),
        click.option("--save-npz", is_flag=True, help="Also export raw float32 depth & focal length .npz."),
        click.option(
            "-o",
            "--output",
            type=click.Path(path_type=Path, file_okay=False),
            required=True,
            help="Output directory for Unity terrain bundle.",
        ),
    ]
    for opt in reversed(options):
        fn = opt(fn)
    return fn


def _build_export_config(
    resolution: str,
    invert: bool,
    smooth_sigma: float,
    gamma: float,
    p_low: float,
    p_high: float,
    save_npz: bool,
) -> ExportConfig:
    return ExportConfig(
        resolution=int(resolution),
        invert=invert,
        smooth_sigma=smooth_sigma,
        gamma=gamma,
        percentile_low=p_low,
        percentile_high=p_high,
        save_npz=save_npz,
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """Apple Depth Pro 2 -> Unity 6 Terrain Pipeline CLI."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )


@main.command("download")
@click.option(
    "-m",
    "--model",
    type=click.Choice(VALID_MODELS),
    default="depthpro2-b",
    show_default=True,
    help="Apple Depth Pro 2 model checkpoint to download.",
)
@click.option(
    "--all-laptop",
    is_flag=True,
    help="Download both depthpro2-s (97MB) and depthpro2-b (386MB) plus shared FOV estimator.",
)
def download_cmd(model: str, all_laptop: bool) -> None:
    """Pre-download Apple Depth Pro 2 model weights into pipeline/checkpoints/."""
    models = ["depthpro2-s", "depthpro2-b"] if all_laptop else [model]
    for name in models:
        click.echo(f"Downloading Apple Depth Pro 2 checkpoint '{name}' + FOV estimator...")
        paths = download_model_weights(name, include_fov=True)
        for k, p in paths.items():
            size_mb = p.stat().st_size / (1024 * 1024)
            click.echo(f"  [{k}] {p} ({size_mb:.1f} MB)")


@main.command("from-map")
@click.option("--place", type=str, default=None, help="Place name to geocode via OpenStreetMap Nominatim.")
@click.option("--lat", type=float, default=None, help="Center latitude (degrees).")
@click.option("--lon", type=float, default=None, help="Center longitude (degrees).")
@click.option("--zoom", type=int, default=14, show_default=True, help="Slippy map zoom level (1..19).")
@click.option("--grid", type=int, default=3, show_default=True, help="Tile grid size NxN (3=768px, 4=1024px).")
@click.option(
    "--provider",
    type=click.Choice(sorted(TILE_PROVIDERS.keys())),
    default="esri_satellite",
    show_default=True,
    help="Map/satellite tile provider.",
)
@click.option(
    "--custom-url",
    type=str,
    default=None,
    help="Optional custom XYZ tile template with {z}, {x}, {y} placeholders.",
)
@_common_model_options
def from_map_cmd(
    place: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    zoom: int,
    grid: int,
    provider: str,
    custom_url: Optional[str],
    model: str,
    device: str,
    precision: Optional[str],
    topdown_mode: str,
    resolution: str,
    invert: bool,
    smooth_sigma: float,
    gamma: float,
    p_low: float,
    p_high: float,
    save_npz: bool,
    output: Path,
) -> None:
    """Fetch real-world satellite/map tiles and estimate 3D terrain via Apple Depth Pro 2."""
    place_label = place or ""
    if place and (lat is None or lon is None):
        click.echo(f"Geocoding '{place}' via OpenStreetMap Nominatim...")
        lat, lon, resolved_name = geocode_place(place)
        place_label = resolved_name
        click.echo(f"  Resolved -> ({lat:.5f}, {lon:.5f}) {resolved_name}")
    elif lat is None or lon is None:
        raise click.UsageError("Provide either --place or both --lat and --lon.")

    click.echo(f"Fetching {grid}x{grid} tiles from '{custom_url or provider}' at zoom {zoom}...")
    map_res = fetch_map_region(
        lat=lat,
        lon=lon,
        zoom=zoom,
        grid_size=grid,
        provider=provider,
        custom_url=custom_url,
        place_label=place_label,
    )
    click.echo(
        f"  Stitched {map_res.image_rgb.shape[1]}x{map_res.image_rgb.shape[0]} px "
        f"(~{map_res.ground_width_m:.0f}m x {map_res.ground_height_m:.0f}m)"
    )

    click.echo(f"Running Apple Depth Pro 2 ({model}, mode={topdown_mode}) on device '{device}'...")
    depth_res = infer_image_depth(
        map_res.image_rgb,
        model_name=model,
        device_name=device,
        precision_name=precision,
        topdown_mode=topdown_mode,
    )
    click.echo(
        f"  Depth inferred in {depth_res.inference_ms:.1f} ms on {depth_res.device} ({depth_res.precision})"
    )

    cfg = _build_export_config(resolution, invert, smooth_sigma, gamma, p_low, p_high, save_npz)
    exported = export_unity_terrain_bundle(
        output_dir=output,
        albedo_rgb=map_res.image_rgb,
        depth_m=depth_res.depth_m,
        inverse_depth=depth_res.inverse_depth,
        focallength_px=depth_res.focallength_px,
        model_name=depth_res.model_name,
        device=depth_res.device,
        precision=depth_res.precision,
        inference_ms=depth_res.inference_ms,
        source_type="real_map",
        source_label=map_res.place_label,
        config=cfg,
        extra_metadata={
            "latitude": map_res.lat,
            "longitude": map_res.lon,
            "zoom": map_res.zoom,
            "mapProvider": map_res.provider,
            "groundWidthMeters": map_res.ground_width_m,
            "groundHeightMeters": map_res.ground_height_m,
            "recommendedTerrainSize": 500.0,
            "recommendedTerrainHeight": 125.0,
            "recommendedWaterLevel": 0.04,
        },
    )
    click.echo(json.dumps({"status": "ok", "outputDir": str(exported.output_dir.resolve()), "metadata": exported.metadata}))


@main.command("from-image")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to a top-down image (real aerial/satellite photo or fictional top-down artwork).",
)
@click.option("--label", type=str, default=None, help="Optional display label for this terrain.")
@_common_model_options
def from_image_cmd(
    input_path: Path,
    label: Optional[str],
    model: str,
    device: str,
    precision: Optional[str],
    topdown_mode: str,
    resolution: str,
    invert: bool,
    smooth_sigma: float,
    gamma: float,
    p_low: float,
    p_high: float,
    save_npz: bool,
    output: Path,
) -> None:
    """Estimate 3D terrain from any local top-down image (real or fictional) via Apple Depth Pro 2."""
    click.echo(f"Loading image {input_path}...")
    rgb, f_px = load_local_image(input_path)
    click.echo(f"  Loaded {rgb.shape[1]}x{rgb.shape[0]} px (EXIF f_px={f_px})")

    click.echo(f"Running Apple Depth Pro 2 ({model}, mode={topdown_mode}) on device '{device}'...")
    depth_res = infer_image_depth(
        rgb,
        model_name=model,
        device_name=device,
        precision_name=precision,
        f_px=f_px,
        topdown_mode=topdown_mode,
    )
    click.echo(f"  Depth inferred in {depth_res.inference_ms:.1f} ms on {depth_res.device}")

    cfg = _build_export_config(resolution, invert, smooth_sigma, gamma, p_low, p_high, save_npz)
    exported = export_unity_terrain_bundle(
        output_dir=output,
        albedo_rgb=rgb,
        depth_m=depth_res.depth_m,
        inverse_depth=depth_res.inverse_depth,
        focallength_px=depth_res.focallength_px,
        model_name=depth_res.model_name,
        device=depth_res.device,
        precision=depth_res.precision,
        inference_ms=depth_res.inference_ms,
        source_type="local_image",
        source_label=label or input_path.stem,
        config=cfg,
        extra_metadata={
            "recommendedTerrainSize": 500.0,
            "recommendedTerrainHeight": 120.0,
            "recommendedWaterLevel": 0.06,
        },
    )
    click.echo(json.dumps({"status": "ok", "outputDir": str(exported.output_dir.resolve()), "metadata": exported.metadata}))


@main.command("from-fictional")
@click.option(
    "--biome",
    type=click.Choice(FICTIONAL_BIOMES),
    default="volcanic_island",
    show_default=True,
    help="Procedural fictional top-down aerial biome to synthesize before Depth Pro 2 inference.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for fictional map.")
@click.option("--size", type=int, default=768, show_default=True, help="Generated top-down RGB image size (px).")
@_common_model_options
def from_fictional_cmd(
    biome: str,
    seed: int,
    size: int,
    model: str,
    device: str,
    precision: Optional[str],
    topdown_mode: str,
    resolution: str,
    invert: bool,
    smooth_sigma: float,
    gamma: float,
    p_low: float,
    p_high: float,
    save_npz: bool,
    output: Path,
) -> None:
    """Generate a fictional top-down landscape RGB image and reconstruct its 3D heightmap with Apple Depth Pro 2."""
    click.echo(f"Generating fictional top-down aerial world '{biome}' (seed={seed}, {size}x{size})...")
    fictional = generate_fictional_topdown(biome=biome, size=size, seed=seed)

    click.echo(f"Running Apple Depth Pro 2 ({model}, mode={topdown_mode}) on fictional aerial RGB...")
    depth_res = infer_image_depth(
        fictional.image_rgb,
        model_name=model,
        device_name=device,
        precision_name=precision,
        topdown_mode=topdown_mode,
    )
    click.echo(f"  Depth inferred in {depth_res.inference_ms:.1f} ms on {depth_res.device}")

    water_defaults = {
        "volcanic_island": 0.16,
        "fantasy_archipelago": 0.22,
        "alpine_peaks": 0.06,
        "canyon_mesa": 0.08,
    }
    cfg = _build_export_config(resolution, invert, smooth_sigma, gamma, p_low, p_high, save_npz)
    exported = export_unity_terrain_bundle(
        output_dir=output,
        albedo_rgb=fictional.image_rgb,
        depth_m=depth_res.depth_m,
        inverse_depth=depth_res.inverse_depth,
        focallength_px=depth_res.focallength_px,
        model_name=depth_res.model_name,
        device=depth_res.device,
        precision=depth_res.precision,
        inference_ms=depth_res.inference_ms,
        source_type="fictional_topdown",
        source_label=f"Fictional {biome.replace('_', ' ').title()} (seed {seed})",
        config=cfg,
        extra_metadata={
            "biome": biome,
            "seed": seed,
            "recommendedTerrainSize": 500.0,
            "recommendedTerrainHeight": 135.0,
            "recommendedWaterLevel": water_defaults.get(biome, 0.10),
        },
    )
    click.echo(json.dumps({"status": "ok", "outputDir": str(exported.output_dir.resolve()), "metadata": exported.metadata}))


@main.command("from-video")
@click.option(
    "-i",
    "--input",
    "video_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to a top-down / aerial drone video (.mp4, .mov, .webm, etc.).",
)
@click.option("--mode", type=click.Choice(["buffered", "unbuffered"]), default="buffered", show_default=True)
@click.option("--window", type=int, default=8, show_default=True, help="Temporal attention window size.")
@click.option("--overlap", type=int, default=2, show_default=True, help="Temporal window overlap (buffered mode).")
@click.option("--max-frames", type=int, default=16, show_default=True, help="Maximum video frames to process.")
@click.option("--stride", type=int, default=5, show_default=True, help="Decode every N-th video frame.")
@_common_model_options
def from_video_cmd(
    video_path: Path,
    mode: str,
    window: int,
    overlap: int,
    max_frames: int,
    stride: int,
    model: str,
    device: str,
    precision: Optional[str],
    topdown_mode: str,
    resolution: str,
    invert: bool,
    smooth_sigma: float,
    gamma: float,
    p_low: float,
    p_high: float,
    save_npz: bool,
    output: Path,
) -> None:
    """Run temporally consistent Apple Depth Pro 2 video depth estimation and export frame sequence for Unity."""
    click.echo(f"Running Apple Depth Pro 2 ({model}, {mode}) on video {video_path}...")
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_export_config(resolution, invert, smooth_sigma, gamma, p_low, p_high, save_npz)

    results = list(
        infer_video_depth(
            video_path=video_path,
            model_name=model,
            device_name=device,
            precision_name=precision,
            mode=mode,
            window=window,
            overlap=overlap,
            max_frames=max_frames,
            frame_stride=stride,
            detrend_frames=(topdown_mode != "raw"),
        )
    )
    click.echo(f"  Inferred {len(results)} temporally consistent video frames.")

    for seq_idx, (src_idx, rgb, depth_res) in enumerate(results):
        frame_out = frames_dir / f"frame_{seq_idx:04d}"
        export_unity_terrain_bundle(
            output_dir=frame_out,
            albedo_rgb=rgb,
            depth_m=depth_res.depth_m,
            inverse_depth=depth_res.inverse_depth,
            focallength_px=depth_res.focallength_px,
            model_name=depth_res.model_name,
            device=depth_res.device,
            precision=depth_res.precision,
            inference_ms=depth_res.inference_ms,
            source_type="aerial_video_frame",
            source_label=f"{video_path.stem} [frame {src_idx}]",
            config=cfg,
        )

    # Export middle keyframe as root bundle so Unity loads it immediately
    mid = len(results) // 2
    _, mid_rgb, mid_depth = results[mid]
    exported = export_unity_terrain_bundle(
        output_dir=output,
        albedo_rgb=mid_rgb,
        depth_m=mid_depth.depth_m,
        inverse_depth=mid_depth.inverse_depth,
        focallength_px=mid_depth.focallength_px,
        model_name=mid_depth.model_name,
        device=mid_depth.device,
        precision=mid_depth.precision,
        inference_ms=mid_depth.inference_ms,
        source_type="aerial_video",
        source_label=f"{video_path.stem} ({len(results)} frames)",
        config=cfg,
        extra_metadata={
            "videoFrameCount": len(results),
            "recommendedTerrainSize": 500.0,
            "recommendedTerrainHeight": 115.0,
            "recommendedWaterLevel": 0.05,
        },
    )
    click.echo(json.dumps({"status": "ok", "outputDir": str(exported.output_dir.resolve()), "metadata": exported.metadata}))


if __name__ == "__main__":
    main()
