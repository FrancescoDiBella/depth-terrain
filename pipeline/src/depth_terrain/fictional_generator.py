"""Procedural fictional top-down aerial landscape RGB generator.

Generates photorealistic, sunlit top-down RGB imagery for fictional worlds using
vectorized 2D gradient (Perlin) FBM noise so Apple Depth Pro 2 can infer 3D
monocular depth directly from invented landscapes without grid artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

FICTIONAL_BIOMES = (
    "volcanic_island",
    "alpine_peaks",
    "canyon_mesa",
    "fantasy_archipelago",
)


@dataclass(frozen=True)
class FictionalMapResult:
    """Generated fictional top-down aerial RGB image and metadata."""

    image_rgb: np.ndarray  # (H, W, 3) uint8
    biome: str
    seed: int
    size: int


def _perlin_2d(h: int, w: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized 2D gradient Perlin noise in approximately [-1, 1] with C2 quintic fade."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(grid + 1, grid + 1)).astype(np.float32)
    gx = np.cos(angles)
    gy = np.sin(angles)

    y = np.linspace(0.0, grid - 1e-4, h, dtype=np.float32)
    x = np.linspace(0.0, grid - 1e-4, w, dtype=np.float32)
    yi = np.floor(y).astype(np.int32)
    xi = np.floor(x).astype(np.int32)
    yf = (y - yi)[:, None]
    xf = (x - xi)[None, :]

    # Quintic smoothstep: 6t^5 - 15t^4 + 10t^3
    u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

    def dot(dy: int, dx: int) -> np.ndarray:
        g_x = gx[yi[:, None] + dy, xi[None, :] + dx]
        g_y = gy[yi[:, None] + dy, xi[None, :] + dx]
        return g_x * (xf - dx) + g_y * (yf - dy)

    n00 = dot(0, 0)
    n10 = dot(0, 1)
    n01 = dot(1, 0)
    n11 = dot(1, 1)

    nx0 = n00 * (1.0 - u) + n10 * u
    nx1 = n01 * (1.0 - u) + n11 * u
    return (nx0 * (1.0 - v) + nx1 * v).astype(np.float32)


def _fbm(
    size: int,
    rng: np.random.Generator,
    octaves: int = 6,
    base_grid: int = 3,
    persistence: float = 0.52,
    ridged: bool = False,
) -> np.ndarray:
    acc = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    grid = base_grid
    for _ in range(octaves):
        n = _perlin_2d(size, size, grid, rng)
        if ridged:
            n = 1.0 - np.abs(n * 1.4)
            n = np.clip(n, 0.0, 1.0) ** 2.0
        else:
            n = n * 0.5 + 0.5
        acc += amp * n
        norm += amp
        amp *= persistence
        grid = min(size // 2, grid * 2)
    out = acc / max(norm, 1e-6)
    return (out - out.min()) / max(float(out.max() - out.min()), 1e-6)


def _compute_hillshade(
    elevation: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 34.0, z_factor: float = 28.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute directional sunlight illumination and slope magnitude for satellite-like realism."""
    dy, dx = np.gradient(elevation * z_factor)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dy, dx)

    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    smooth = gaussian_filter(elevation, sigma=max(2.0, elevation.shape[0] / 80.0))
    ao = np.clip(0.86 + 2.1 * (elevation - smooth), 0.50, 1.18)
    light = np.clip(0.26 + 0.92 * np.clip(shaded, 0.0, 1.0), 0.22, 1.28) * ao
    slope_norm = np.clip(slope / (np.pi * 0.4), 0.0, 1.0)
    return light.astype(np.float32), slope_norm.astype(np.float32)


def _lerp_ramp(value: np.ndarray, stops: list[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    """Map scalar array in [0, 1] to RGB float32 array in [0, 255] via piecewise linear stops."""
    h, w = value.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    v = np.clip(value, 0.0, 1.0)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        mask = (v >= p0) & (v <= p1 if i == len(stops) - 2 else v < p1)
        t = np.clip((v[mask] - p0) / max(p1 - p0, 1e-6), 0.0, 1.0)[:, None]
        col = np.array(c0, dtype=np.float32) * (1.0 - t) + np.array(c1, dtype=np.float32) * t
        rgb[mask] = col
    return rgb


def generate_fictional_topdown(
    biome: str = "volcanic_island",
    size: int = 768,
    seed: int = 42,
) -> FictionalMapResult:
    """Synthesize a photorealistic fictional top-down aerial RGB photograph.

    The internal elevation synthesis is used strictly to render a realistic sunlit
    top-down RGB photograph — Apple Depth Pro 2 then infers the 3D depth map
    purely from the resulting RGB pixels.
    """
    if biome not in FICTIONAL_BIOMES:
        raise ValueError(f"Unknown fictional biome '{biome}'. Choose one of: {', '.join(FICTIONAL_BIOMES)}")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = (xx / (size - 1)) * 2.0 - 1.0
    ny = (yy / (size - 1)) * 2.0 - 1.0
    radius = np.hypot(nx, ny)

    base = _fbm(size, rng, octaves=6, base_grid=3, persistence=0.52, ridged=False)
    ridges = _fbm(size, rng, octaves=6, base_grid=4, persistence=0.54, ridged=True)
    micro = _fbm(size, rng, octaves=4, base_grid=20, persistence=0.55, ridged=False)

    if biome == "volcanic_island":
        island_mask = np.clip(1.0 - (radius / 0.88) ** 2.3, 0.0, 1.0)
        cone = np.exp(-((radius / 0.42) ** 2.0)) * 0.82
        caldera = np.exp(-((radius / 0.12) ** 2.6)) * 0.36
        elev = island_mask * (0.28 * base + 0.44 * ridges) + cone - caldera
        elev = np.clip(elev, 0.0, 1.0)
        elev = (elev - elev.min()) / max(float(elev.max() - elev.min()), 1e-6)

        light, slope = _compute_hillshade(elev, azimuth_deg=315.0, altitude_deg=34.0, z_factor=32.0)
        albedo = _lerp_ramp(
            elev,
            [
                (0.00, (12, 42, 74)),
                (0.14, (20, 82, 118)),
                (0.21, (48, 142, 156)),
                (0.25, (198, 182, 138)),
                (0.30, (58, 102, 48)),
                (0.50, (40, 78, 38)),
                (0.66, (94, 86, 78)),
                (0.82, (132, 122, 114)),
                (0.93, (176, 162, 152)),
                (1.00, (228, 226, 222)),
            ],
        )
        crater_center = np.exp(-((radius / 0.05) ** 2.4))
        albedo[..., 0] = np.clip(albedo[..., 0] + crater_center * 90.0, 0, 255)
        albedo[..., 1] = np.clip(albedo[..., 1] - crater_center * 28.0, 0, 255)
        albedo[..., 2] = np.clip(albedo[..., 2] - crater_center * 45.0, 0, 255)
        water_blend = np.clip((0.23 - elev) / 0.06, 0.0, 1.0)
        light = light * (1.0 - water_blend) + (0.92 + 0.08 * micro) * water_blend

    elif biome == "alpine_peaks":
        valley = np.exp(-(((nx * 0.85 + ny * 0.42 + 0.22 * (base - 0.5)) / 0.24) ** 2.0))
        elev = (0.32 * base + 0.78 * ridges) * (1.0 - 0.68 * valley)
        elev = (elev - elev.min()) / max(float(elev.max() - elev.min()), 1e-6)
        light, slope = _compute_hillshade(elev, azimuth_deg=305.0, altitude_deg=30.0, z_factor=34.0)
        albedo = _lerp_ramp(
            elev,
            [
                (0.00, (32, 76, 106)),
                (0.10, (54, 96, 54)),
                (0.28, (38, 70, 40)),
                (0.46, (90, 92, 84)),
                (0.60, (128, 130, 134)),
                (0.73, (206, 214, 224)),
                (1.00, (248, 250, 254)),
            ],
        )
        rock = np.clip((slope - 0.32) * 2.5, 0.0, 0.78)[..., None]
        albedo = albedo * (1.0 - rock) + np.array([92, 96, 102], dtype=np.float32) * rock

    elif biome == "canyon_mesa":
        river_x = 0.28 * np.sin(ny * 2.6 + 1.1 * (base - 0.5))
        gorge = np.exp(-(((nx - river_x) / 0.20) ** 2.0))
        raw_elev = 0.62 * base + 0.38 * ridges
        terraced = np.floor(raw_elev * 6.0) / 6.0
        elev = (0.55 * raw_elev + 0.45 * terraced) * (1.0 - 0.80 * gorge)
        elev = (elev - elev.min()) / max(float(elev.max() - elev.min()), 1e-6)
        light, _ = _compute_hillshade(elev, azimuth_deg=290.0, altitude_deg=30.0, z_factor=36.0)
        albedo = _lerp_ramp(
            elev,
            [
                (0.00, (36, 94, 112)),
                (0.09, (124, 88, 62)),
                (0.25, (162, 92, 56)),
                (0.48, (188, 108, 66)),
                (0.70, (206, 136, 88)),
                (0.88, (220, 170, 118)),
                (1.00, (228, 196, 152)),
            ],
        )

    else:  # fantasy_archipelago
        c1 = np.exp(-(((nx + 0.16) ** 2 + (ny + 0.12) ** 2) / 0.24))
        c2 = np.exp(-(((nx - 0.42) ** 2 + (ny - 0.28) ** 2) / 0.14))
        c3 = np.exp(-(((nx - 0.36) ** 2 + (ny + 0.42) ** 2) / 0.11))
        c4 = np.exp(-(((nx + 0.45) ** 2 + (ny - 0.40) ** 2) / 0.10))
        clusters = np.clip(c1 + 0.88 * c2 + 0.74 * c3 + 0.70 * c4, 0.0, 1.25)
        elev = clusters * (0.32 + 0.48 * ridges + 0.36 * base)
        elev = (elev - elev.min()) / max(float(elev.max() - elev.min()), 1e-6)
        light, _ = _compute_hillshade(elev, azimuth_deg=320.0, altitude_deg=36.0, z_factor=28.0)
        albedo = _lerp_ramp(
            elev,
            [
                (0.00, (12, 42, 82)),
                (0.20, (18, 92, 134)),
                (0.28, (54, 166, 178)),
                (0.32, (216, 200, 150)),
                (0.38, (60, 126, 60)),
                (0.58, (36, 90, 46)),
                (0.76, (108, 104, 96)),
                (0.90, (186, 190, 196)),
                (1.00, (244, 248, 252)),
            ],
        )
        water_blend = np.clip((0.30 - elev) / 0.06, 0.0, 1.0)
        light = light * (1.0 - water_blend) + (0.94 + 0.06 * micro) * water_blend

    grain = (0.93 + 0.14 * micro)[..., None]
    shaded_rgb = np.clip(albedo * light[..., None] * grain, 0.0, 255.0).astype(np.uint8)
    return FictionalMapResult(image_rgb=shaded_rgb, biome=biome, seed=seed, size=size)
