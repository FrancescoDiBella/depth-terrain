"""Persistent warm HTTP inference server for Apple Depth Pro 2 -> Unity 6 Terrain.

Uses Python's standard library ``http.server.ThreadingHTTPServer`` so no extra
web framework dependencies are required. Keeps the Depth Pro 2 model warm in
Apple Silicon MPS / CUDA / CPU memory via ``get_cached_model()`` for near-instant
subsequent requests from Unity Editor and Runtime HUD.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .depth_runner import (
    PRECISIONS,
    TOPDOWN_MODES,
    VALID_MODELS,
    get_cached_model,
    infer_image_depth,
    infer_video_depth,
    list_cached_models,
    load_local_image,
    resolve_device,
    resolve_precision_name,
    select_best_video_keyframe,
)
from .fictional_generator import FICTIONAL_BIOMES, generate_fictional_topdown
from .heightmap_export import ExportConfig, export_unity_terrain_bundle
from .map_fetcher import TILE_PROVIDERS, fetch_map_region, geocode_place

LOGGER = logging.getLogger(__name__)

FICTIONAL_WATER_DEFAULTS = {
    "volcanic_island": 0.16,
    "fantasy_archipelago": 0.22,
    "alpine_peaks": 0.06,
    "canyon_mesa": 0.08,
}


def _pick(payload: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present non-None value across camelCase/snake_case aliases."""
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


class DepthTerrainHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server holding warm Depth Pro 2 server state."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        default_model: str = "depthpro2-b",
        default_device: str = "auto",
        default_precision: Optional[str] = None,
        checkpoints_dir: Optional[Path] = None,
        preload: bool = True,
        warmup: bool = True,
        infer_image_fn: Optional[Callable[..., Any]] = None,
        infer_video_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if default_model not in VALID_MODELS:
            raise ValueError(
                f"Unknown model '{default_model}'. Choose one of: {', '.join(VALID_MODELS)}"
            )
        resolved_dev = resolve_device(default_device)
        resolved_prec = resolve_precision_name(resolved_dev, default_precision)

        self.default_model = default_model
        self.default_device = str(resolved_dev)
        self.default_precision_flag = resolved_prec
        self.default_precision_label = str(PRECISIONS[resolved_prec]).removeprefix("torch.")
        self.checkpoints_dir = checkpoints_dir
        self.start_time = time.monotonic()
        self.requests_served = 0
        self._stats_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._infer_image_fn = infer_image_fn or infer_image_depth
        self._infer_video_fn = infer_video_fn or infer_video_depth

        if preload:
            LOGGER.info(
                "Preloading Apple Depth Pro 2 model '%s' on %s (%s)...",
                self.default_model,
                self.default_device,
                self.default_precision_flag,
            )
            bundle = get_cached_model(
                model_name=self.default_model,
                device=self.default_device,
                precision=self.default_precision_flag,
                checkpoints_dir=self.checkpoints_dir,
            )
            self.default_precision_label = str(
                getattr(bundle[0], "precision", PRECISIONS[resolved_prec])
            ).removeprefix("torch.")
            if warmup:
                LOGGER.info("Running quick MPS/GPU shader warmup pass (256x256)...")
                dummy = np.zeros((256, 256, 3), dtype=np.uint8)
                self._infer_image_fn(
                    dummy,
                    model_name=self.default_model,
                    device_name=self.default_device,
                    precision_name=self.default_precision_flag,
                    topdown_mode="raw",
                    loaded_bundle=bundle,
                )

        super().__init__(server_address, DepthTerrainRequestHandler)

    def increment_requests_served(self) -> None:
        with self._stats_lock:
            self.requests_served += 1

    def get_health_payload(self) -> Dict[str, Any]:
        with self._stats_lock:
            served = self.requests_served
        cached = list_cached_models()
        return {
            "status": "ok",
            "model": self.default_model,
            "device": self.default_device,
            "precision": self.default_precision_label,
            "uptimeSeconds": round(time.monotonic() - self.start_time, 2),
            "cachedModels": cached,
            "requestsServed": served,
        }


class DepthTerrainRequestHandler(BaseHTTPRequestHandler):
    """JSON HTTP request handler for `/health` and `/infer/*` endpoints."""

    server: DepthTerrainHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if not raw_len:
            return {}
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON request body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be a JSON object.")
        return data

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(int(HTTPStatus.NO_CONTENT))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self._send_json(HTTPStatus.OK, self.server.get_health_payload())
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"status": "error", "error": f"Unknown GET endpoint '{self.path}'."},
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        routes = {
            "/infer/map": self._handle_infer_map,
            "/infer/image": self._handle_infer_image,
            "/infer/fictional": self._handle_infer_fictional,
            "/infer/video": self._handle_infer_video,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "error": f"Unknown POST endpoint '{self.path}'."},
            )
            return

        t0 = time.perf_counter()
        try:
            payload = self._read_json_body()
            output_dir, metadata = handler(payload)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.server.increment_requests_served()
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "outputDir": str(output_dir.resolve()),
                    "elapsedMs": elapsed_ms,
                    "metadata": metadata,
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": str(exc)})
        except FileNotFoundError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("Unhandled error serving %s", path)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": str(exc)},
            )

    def _parse_common(self, payload: Dict[str, Any]) -> tuple[Path, str, str, Optional[str], str, ExportConfig]:
        out_str = _as_optional_str(_pick(payload, "outputDir", "output_dir"))
        if out_str:
            output_dir = Path(out_str).expanduser()
        else:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            output_dir = Path(tempfile.gettempdir()) / "depth_terrain_server" / run_name

        model_name = _as_optional_str(_pick(payload, "model")) or self.server.default_model
        if model_name not in VALID_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(VALID_MODELS)}")

        device_name = _as_optional_str(_pick(payload, "device")) or self.server.default_device
        precision_name = _as_optional_str(_pick(payload, "precision"))
        if precision_name is None and device_name == self.server.default_device:
            precision_name = self.server.default_precision_flag

        topdown_mode = _as_optional_str(_pick(payload, "topDownMode", "topdown_mode")) or "fast_multiscale"
        if topdown_mode not in TOPDOWN_MODES:
            raise ValueError(
                f"Unknown topDownMode '{topdown_mode}'. Choose one of: {', '.join(TOPDOWN_MODES)}"
            )

        cfg = ExportConfig(
            resolution=int(_pick(payload, "resolution", default=513)),
            invert=_as_bool(_pick(payload, "invert", default=False)),
            smooth_sigma=float(_pick(payload, "smoothSigma", "smooth_sigma", default=1.0)),
            gamma=float(_pick(payload, "gamma", default=1.0)),
            percentile_low=float(_pick(payload, "pLow", "p_low", default=2.0)),
            percentile_high=float(_pick(payload, "pHigh", "p_high", default=98.0)),
            save_npz=_as_bool(_pick(payload, "saveNpz", "save_npz", default=False)),
            guided_filter=_as_bool(_pick(payload, "guidedFilter", "guided_filter", default=False)),
            export_ply=_as_bool(_pick(payload, "exportPly", "export_ply", default=False)),
        )
        return output_dir, model_name, device_name, precision_name, topdown_mode, cfg

    def _handle_infer_map(self, payload: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
        output_dir, model_name, device_name, precision_name, topdown_mode, cfg = self._parse_common(payload)

        place = _as_optional_str(_pick(payload, "place", "placeQuery"))
        use_coords = _as_bool(_pick(payload, "useCoordinates", "use_coordinates", default=False))
        lat_raw = _pick(payload, "latitude", "lat")
        lon_raw = _pick(payload, "longitude", "lon")
        lat: Optional[float] = float(lat_raw) if lat_raw is not None else None
        lon: Optional[float] = float(lon_raw) if lon_raw is not None else None

        place_label = place or ""
        if (not use_coords) and place:
            lat, lon, resolved_name = geocode_place(place)
            place_label = resolved_name
        elif lat is None or lon is None:
            if place:
                lat, lon, resolved_name = geocode_place(place)
                place_label = resolved_name
            else:
                raise ValueError("Provide either 'place' or both 'latitude' and 'longitude'.")

        zoom = int(_pick(payload, "zoom", default=14))
        grid = int(_pick(payload, "grid", "tileGrid", default=3))
        provider = _as_optional_str(_pick(payload, "provider", "mapProvider")) or "esri_satellite"
        custom_url = _as_optional_str(_pick(payload, "customUrl", "custom_url"))
        if not custom_url and provider not in TILE_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose one of: {', '.join(sorted(TILE_PROVIDERS))}"
            )

        map_res = fetch_map_region(
            lat=lat,
            lon=lon,
            zoom=zoom,
            grid_size=grid,
            provider=provider,
            custom_url=custom_url,
            place_label=place_label,
        )

        with self.server._infer_lock:
            depth_res = self.server._infer_image_fn(
                map_res.image_rgb,
                model_name=model_name,
                device_name=device_name,
                precision_name=precision_name,
                topdown_mode=topdown_mode,
                checkpoints_dir=self.server.checkpoints_dir,
            )

        exported = export_unity_terrain_bundle(
            output_dir=output_dir,
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
                "topDownMode": depth_res.topdown_mode,
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
        return exported.output_dir, exported.metadata

    def _handle_infer_image(self, payload: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
        output_dir, model_name, device_name, precision_name, topdown_mode, cfg = self._parse_common(payload)

        input_str = _as_optional_str(_pick(payload, "inputPath", "input_path", "input"))
        if not input_str:
            raise ValueError("Missing required field 'inputPath'.")
        input_path = Path(input_str).expanduser()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        label = _as_optional_str(_pick(payload, "label"))
        rgb, f_px = load_local_image(input_path)

        with self.server._infer_lock:
            depth_res = self.server._infer_image_fn(
                rgb,
                model_name=model_name,
                device_name=device_name,
                precision_name=precision_name,
                f_px=f_px,
                topdown_mode=topdown_mode,
                checkpoints_dir=self.server.checkpoints_dir,
            )

        exported = export_unity_terrain_bundle(
            output_dir=output_dir,
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
                "topDownMode": depth_res.topdown_mode,
                "recommendedTerrainSize": 500.0,
                "recommendedTerrainHeight": 120.0,
                "recommendedWaterLevel": 0.06,
            },
        )
        return exported.output_dir, exported.metadata

    def _handle_infer_fictional(self, payload: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
        output_dir, model_name, device_name, precision_name, topdown_mode, cfg = self._parse_common(payload)

        biome = _as_optional_str(_pick(payload, "biome")) or "volcanic_island"
        if biome not in FICTIONAL_BIOMES:
            raise ValueError(f"Unknown biome '{biome}'. Choose one of: {', '.join(FICTIONAL_BIOMES)}")
        seed = int(_pick(payload, "seed", default=42))
        size = int(_pick(payload, "size", default=768))
        if size < 64 or size > 2048:
            raise ValueError(f"Invalid size {size}; expected between 64 and 2048.")

        fictional = generate_fictional_topdown(biome=biome, size=size, seed=seed)

        with self.server._infer_lock:
            depth_res = self.server._infer_image_fn(
                fictional.image_rgb,
                model_name=model_name,
                device_name=device_name,
                precision_name=precision_name,
                topdown_mode=topdown_mode,
                checkpoints_dir=self.server.checkpoints_dir,
            )

        exported = export_unity_terrain_bundle(
            output_dir=output_dir,
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
                "topDownMode": depth_res.topdown_mode,
                "biome": biome,
                "seed": seed,
                "recommendedTerrainSize": 500.0,
                "recommendedTerrainHeight": 135.0,
                "recommendedWaterLevel": FICTIONAL_WATER_DEFAULTS.get(biome, 0.10),
            },
        )
        return exported.output_dir, exported.metadata

    def _handle_infer_video(self, payload: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
        output_dir, model_name, device_name, precision_name, topdown_mode, cfg = self._parse_common(payload)

        input_str = _as_optional_str(_pick(payload, "inputPath", "input_path", "input"))
        if not input_str:
            raise ValueError("Missing required field 'inputPath'.")
        video_path = Path(input_str).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        mode = _as_optional_str(_pick(payload, "mode")) or "buffered"
        window = int(_pick(payload, "window", default=8))
        overlap = int(_pick(payload, "overlap", default=2))
        max_frames = int(_pick(payload, "maxFrames", "max_frames", default=16))
        stride = int(_pick(payload, "stride", default=5))

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        with self.server._infer_lock:
            results = list(
                self.server._infer_video_fn(
                    video_path=video_path,
                    model_name=model_name,
                    device_name=device_name,
                    precision_name=precision_name,
                    mode=mode,
                    window=window,
                    overlap=overlap,
                    max_frames=max_frames,
                    frame_stride=stride,
                    detrend_frames=(topdown_mode != "raw"),
                    checkpoints_dir=self.server.checkpoints_dir,
                )
            )

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
                extra_metadata={"topDownMode": depth_res.topdown_mode},
            )

        best_idx = select_best_video_keyframe(results)
        _, key_rgb, key_depth = results[best_idx]
        exported = export_unity_terrain_bundle(
            output_dir=output_dir,
            albedo_rgb=key_rgb,
            depth_m=key_depth.depth_m,
            inverse_depth=key_depth.inverse_depth,
            focallength_px=key_depth.focallength_px,
            model_name=key_depth.model_name,
            device=key_depth.device,
            precision=key_depth.precision,
            inference_ms=key_depth.inference_ms,
            source_type="aerial_video",
            source_label=f"{video_path.stem} ({len(results)} frames)",
            config=cfg,
            extra_metadata={
                "topDownMode": key_depth.topdown_mode,
                "videoFrameCount": len(results),
                "keyframeIndex": best_idx,
                "recommendedTerrainSize": 500.0,
                "recommendedTerrainHeight": 115.0,
                "recommendedWaterLevel": 0.05,
            },
        )
        return exported.output_dir, exported.metadata


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    model_name: str = "depthpro2-b",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    preload: bool = True,
    warmup: bool = True,
    infer_image_fn: Optional[Callable[..., Any]] = None,
    infer_video_fn: Optional[Callable[..., Any]] = None,
) -> DepthTerrainHTTPServer:
    """Create a configured ``DepthTerrainHTTPServer`` ready to serve requests."""
    return DepthTerrainHTTPServer(
        (host, port),
        default_model=model_name,
        default_device=device_name,
        default_precision=precision_name,
        checkpoints_dir=checkpoints_dir,
        preload=preload,
        warmup=warmup,
        infer_image_fn=infer_image_fn,
        infer_video_fn=infer_video_fn,
    )


def serve_forever(
    host: str = "127.0.0.1",
    port: int = 8765,
    model_name: str = "depthpro2-b",
    device_name: str = "auto",
    precision_name: Optional[str] = None,
    checkpoints_dir: Optional[Path] = None,
    warmup: bool = True,
) -> None:
    """Start the persistent warm Depth Pro 2 HTTP inference server and block until interrupted."""
    server = create_server(
        host=host,
        port=port,
        model_name=model_name,
        device_name=device_name,
        precision_name=precision_name,
        checkpoints_dir=checkpoints_dir,
        preload=True,
        warmup=warmup,
    )
    actual_host, actual_port = server.server_address
    print(
        f"Apple Depth Pro 2 warm server listening on http://{actual_host}:{actual_port} "
        f"(model={server.default_model}, device={server.default_device}, "
        f"precision={server.default_precision_label})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
