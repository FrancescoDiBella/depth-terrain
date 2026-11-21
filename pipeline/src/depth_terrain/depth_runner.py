"""Apple Depth Pro 2 model management and image/video inference runner.

Includes top-down nadir adaptation strategies (4-way rotational TTA, linear plane
detrending, perspective-tunnel radial debiasing, and overlapping multi-scale patch
fusion) so monocular perspective depth from Apple Depth Pro 2 converts cleanly into
rotation-invariant topographic heightmaps for Unity Terrains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import av
import numpy as np
import torch

from depth_pro_2 import PRESETS, create_model_and_transforms, load_rgb, upright
from depth_pro_2 import model as depth_pro_2_model
from depth_pro_2.utils import default_device

LOGGER = logging.getLogger(__name__)

VALID_MODELS = tuple(sorted(PRESETS.keys()))
TOPDOWN_MODES = ("multiscale_tta", "tta4", "raw")
PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


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
    if device_name == "auto":
        return default_device()
    return torch.device(device_name)


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
    precision = PRECISIONS[precision_name] if precision_name else None
    model, transform = create_model_and_transforms(model_name, device=device, precision=precision)
    model.eval()
    return model, transform, device


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def detrend_linear_plane(field: np.ndarray) -> np.ndarray:
    """Fit and subtract a best-fit 2D plane ax + by + c to cancel perspective camera tilt."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xx / max(w - 1, 1)) * 2.0 - 1.0
    yn = (yy / max(h - 1, 1)) * 2.0 - 1.0
    design = np.column_stack([xn.ravel(), yn.ravel(), np.ones(h * w, dtype=np.float64)])
    coef, _, _, _ = np.linalg.lstsq(design, field.ravel().astype(np.float64), rcond=None)
    plane = (coef[0] * xn + coef[1] * yn + coef[2]).astype(np.float32)
    return (field - plane).astype(np.float32)


def debias_positive_radial_vignette(field: np.ndarray) -> np.ndarray:
    """Remove perspective tunnel bias (corners artificially closer than center) when present.

    Only subtracts positive r^2 curvature so genuine central peaks/islands are preserved.
    """
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx / max(w - 1, 1)) * 2.0 - 1.0) ** 2 + ((yy / max(h - 1, 1)) * 2.0 - 1.0) ** 2
    design = np.column_stack([r2.ravel(), np.ones(h * w, dtype=np.float64)])
    coef, _, _, _ = np.linalg.lstsq(design, field.ravel().astype(np.float64), rcond=None)
    if coef[0] > 0.0:
        return (field - (coef[0] * r2 + coef[1]).astype(np.float32)).astype(np.float32)
    return field.astype(np.float32)


def _infer_raw_once(
    image_rgb: np.ndarray,
    model,
    transform,
    device: torch.device,
    f_px: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    writable = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")
    tensor = transform(writable)
    prediction = model.infer(tensor, f_px=f_px)
    inv_depth = prediction["inverse_depth"].detach().float().cpu().numpy()
    depth_tensor = prediction["depth"]
    focal_tensor = prediction["focallength_px"]
    focal_val = float(focal_tensor.detach().cpu().item()) if focal_tensor is not None else float(writable.shape[1])
    if depth_tensor is not None:
        depth_m = depth_tensor.detach().float().cpu().numpy()
    else:
        depth_m = focal_val / np.maximum(writable.shape[1] * inv_depth, 1e-6)
    return depth_m, inv_depth, focal_val


def infer_image_depth(
    image_rgb: np.ndarray,
    model_name: str = "depthpro2-b",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    f_px: Optional[float] = None,
    topdown_mode: str = "multiscale_tta",
    checkpoints_dir: Optional[Path] = None,
    loaded_bundle: Optional[Tuple] = None,
) -> DepthInferenceResult:
    """Run Apple Depth Pro 2 on a single (H, W, 3) uint8 RGB image.

    Args:
        topdown_mode:
            - ``raw``: Direct single-pass Depth Pro 2 inference (fastest; ideal for oblique views).
            - ``tta4``: 4-way rotational TTA (0/90/180/270) + plane detrending + radial debias.
            - ``multiscale_tta``: Global 4-way TTA fused with 3x3 overlapping local Depth Pro 2
              patches (sharpest buildings, canyons, ridges, and craters on nadir top-down maps).
    """
    if topdown_mode not in TOPDOWN_MODES:
        raise ValueError(f"Unknown topdown_mode '{topdown_mode}'. Choose one of: {', '.join(TOPDOWN_MODES)}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {image_rgb.shape}.")

    writable = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")

    if loaded_bundle is not None:
        model, transform, device = loaded_bundle
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
        depth_m, inv_depth, focal_val = _infer_raw_once(writable, model, transform, device, f_px=f_px)
    else:
        h, w = writable.shape[:2]
        g_acc = np.zeros((h, w), dtype=np.float32)
        base_depth_m: Optional[np.ndarray] = None
        focal_val = float(w)

        for k in range(4):
            rot = np.array(np.rot90(writable, k), dtype=np.uint8, copy=True, order="C")
            d_k, inv_k, f_k = _infer_raw_once(rot, model, transform, device, f_px=f_px if k == 0 else None)
            if k == 0:
                base_depth_m = d_k
                focal_val = f_k
            unrot = np.rot90(inv_k, -k)
            detrended = detrend_linear_plane(unrot)
            std = float(np.std(detrended)) or 1.0
            g_acc += detrended / std

        g_map = debias_positive_radial_vignette(g_acc / 4.0)
        g_map = g_map / (float(np.std(g_map)) or 1.0)

        if topdown_mode == "multiscale_tta" and min(h, w) >= 384:
            patch_h = max(256, h // 2)
            patch_w = max(256, w // 2)
            ys = sorted({0, (h - patch_h) // 2, h - patch_h})
            xs = sorted({0, (w - patch_w) // 2, w - patch_w})
            win = np.outer(np.hanning(patch_h + 2)[1:-1], np.hanning(patch_w + 2)[1:-1]).astype(np.float32)

            p_acc = np.zeros((h, w), dtype=np.float32)
            w_acc = np.zeros((h, w), dtype=np.float32)
            for y0 in ys:
                for x0 in xs:
                    patch = np.array(writable[y0 : y0 + patch_h, x0 : x0 + patch_w], dtype=np.uint8, copy=True)
                    _, inv0, _ = _infer_raw_once(patch, model, transform, device)
                    patch180 = np.array(np.rot90(patch, 2), dtype=np.uint8, copy=True)
                    _, inv2_rot, _ = _infer_raw_once(patch180, model, transform, device)
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

        assert base_depth_m is not None
        depth_m = base_depth_m

    _sync_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    prec_label = str(model.precision).removeprefix("torch.")

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
) -> Iterator[Tuple[int, np.ndarray, DepthInferenceResult]]:
    """Run temporally consistent Apple Depth Pro 2 inference across aerial video frames."""
    from PIL import Image

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

    eff_window = min(window, len(raw_frames))
    eff_overlap = min(overlap, max(0, eff_window - 1)) if mode == "buffered" else 0

    tensors = (transform(rgb) for _, rgb in raw_frames)
    prec_label = str(model.precision).removeprefix("torch.")
    t0 = time.perf_counter()
    preds = list(model.infer_video(tensors, window=eff_window, overlap=eff_overlap, mode=mode))
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
