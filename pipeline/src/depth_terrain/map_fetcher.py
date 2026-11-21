"""Real-world Slippy Map tile fetcher and OpenStreetMap Nominatim geocoder."""

from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image

USER_AGENT = "DepthTerrain-UnityPipeline/1.0 (+https://github.com/FrancescoDiBella/depth-terrain)"
TILE_SIZE = 256
EARTH_EQUATOR_CIRCUMFERENCE_M = 40075016.68557849

TILE_PROVIDERS: Dict[str, str] = {
    # High-resolution aerial/satellite imagery (note ArcGIS z/y/x order)
    "esri_satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    # Shaded topographic relief from Esri
    "esri_shaded_relief": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}"
    ),
    # OpenStreetMap standard cartography
    "osm_standard": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    # OpenTopoMap topographic contours & terrain shading
    "opentopomap": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
    # CartoDB Voyager basemap
    "carto_voyager": "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
}


@dataclass(frozen=True)
class MapFetchResult:
    """Stitched top-down map imagery and Web Mercator geographic metadata."""

    image_rgb: np.ndarray  # (H, W, 3) uint8
    lat: float
    lon: float
    zoom: int
    grid_size: int
    provider: str
    ground_width_m: float
    ground_height_m: float
    meters_per_pixel: float
    place_label: str = ""


def latlon_to_tile_float(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Convert WGS84 latitude/longitude to fractional Web Mercator Slippy tile coordinates."""
    lat_clamped = max(-85.05112878, min(85.05112878, lat))
    lat_rad = math.radians(lat_clamped)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Return Web Mercator ground resolution (meters per pixel) at latitude and zoom."""
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, lat)))
    return (EARTH_EQUATOR_CIRCUMFERENCE_M * math.cos(lat_rad)) / (TILE_SIZE * (2.0**zoom))


def geocode_place(query: str, timeout: float = 15.0) -> Tuple[float, float, str]:
    """Resolve a human-readable place name via OpenStreetMap Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode place '{query}' via Nominatim.")
    top = results[0]
    return float(top["lat"]), float(top["lon"]), str(top.get("display_name", query))


def _download_tile(
    session: requests.Session,
    template: str,
    z: int,
    x: int,
    y: int,
    timeout: float = 20.0,
    retries: int = 3,
) -> Image.Image:
    n = 1 << z
    wrapped_x = x % n
    clamped_y = max(0, min(n - 1, y))
    url = template.format(z=z, x=wrapped_x, y=clamped_y)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
            return img
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Failed to download map tile z={z} x={x} y={y} from {url}: {last_err}")


def fetch_map_region(
    lat: float,
    lon: float,
    zoom: int = 14,
    grid_size: int = 3,
    provider: str = "esri_satellite",
    custom_url: Optional[str] = None,
    place_label: str = "",
) -> MapFetchResult:
    """Download and stitch an NxN grid of Slippy map tiles centered around (lat, lon).

    Args:
        lat: Center latitude in degrees.
        lon: Center longitude in degrees.
        zoom: Slippy map zoom level (1..19).
        grid_size: Number of tiles per side (e.g., 3 -> 768x768, 4 -> 1024x1024).
        provider: Provider key in ``TILE_PROVIDERS`` or ``"custom"``.
        custom_url: Custom tile template containing ``{z}``, ``{x}``, ``{y}``.
        place_label: Optional human-readable name for metadata.
    """
    if not (1 <= zoom <= 19):
        raise ValueError(f"Zoom must be in [1, 19], got {zoom}.")
    if not (1 <= grid_size <= 8):
        raise ValueError(f"grid_size must be in [1, 8], got {grid_size}.")

    if custom_url:
        template = custom_url
        provider_name = "custom"
    else:
        if provider not in TILE_PROVIDERS:
            valid = ", ".join(sorted(TILE_PROVIDERS))
            raise ValueError(f"Unknown provider '{provider}'. Choose one of: {valid}")
        template = TILE_PROVIDERS[provider]
        provider_name = provider

    fx, fy = latlon_to_tile_float(lat, lon, zoom)
    center_x = int(math.floor(fx))
    center_y = int(math.floor(fy))

    # Fetch a slightly padded tile window if even grid_size so center stays centered
    half_neg = grid_size // 2
    start_x = center_x - half_neg
    start_y = center_y - half_neg

    canvas_px = grid_size * TILE_SIZE
    canvas = Image.new("RGB", (canvas_px, canvas_px))

    coords = [
        (col, row, start_x + col, start_y + row)
        for row in range(grid_size)
        for col in range(grid_size)
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    def _fetch_one(item: Tuple[int, int, int, int]) -> Tuple[int, int, Image.Image]:
        col, row, tx, ty = item
        return col, row, _download_tile(session, template, zoom, tx, ty)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for col, row, tile_img in pool.map(_fetch_one, coords):
            canvas.paste(tile_img, (col * TILE_SIZE, row * TILE_SIZE))

    mpp = meters_per_pixel(lat, zoom)
    ground_m = mpp * canvas_px

    return MapFetchResult(
        image_rgb=np.asarray(canvas, dtype=np.uint8),
        lat=lat,
        lon=lon,
        zoom=zoom,
        grid_size=grid_size,
        provider=provider_name,
        ground_width_m=ground_m,
        ground_height_m=ground_m,
        meters_per_pixel=mpp,
        place_label=place_label or f"{lat:.4f}, {lon:.4f}",
    )
