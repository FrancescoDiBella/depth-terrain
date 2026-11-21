# Research Report: Apple Depth Pro 2 for Image & Video Monocular Depth Estimation

## Executive Summary

**Apple Depth Pro 2** (`apple-aiml-research/ml-depth-pro-2`, [arXiv:2611.26603](https://arxiv.org/abs/2611.26603)) is Apple Machine Learning Research's state-of-the-art foundation model for **fast, sharp, and temporally consistent metric monocular depth estimation** across **single images, full videos, and live camera streams**.

Unlike first-generation monocular depth networks that operate strictly frame-by-frame (causing temporal flicker on video) or output affine-invariant relative depth without metric scale, Depth Pro 2 predicts **metric depth (in meters)** at **full input resolution** with sharp high-frequency boundary fidelity and cross-frame attention for video stability.

---

## Key Technical Capabilities

1. **Unified Image + Video + Live Stream Architecture**
   - Built on **DINOv3 Vision Transformer** backbones (`timm`) coupled with a **MultiViewEncoder** (cross-frame attention blocks) and **DPTDecoder** inverse-depth prediction head.
   - Single-image inference (`model.infer`) predicts metric depth with zero camera calibration required.
   - Video inference supports both:
     - **Buffered mode** (`--mode buffered`, overlapping sliding windows e.g. `--window 16 --overlap 4`) for offline video clips with bidirectional temporal context.
     - **Unbuffered streaming mode** (`--mode unbuffered`, KV-cache causal window e.g. `-w 4`) for low-latency live feeds and real-time Core ML execution.
2. **Automatic Camera Intrinsics (Focal Length Estimator)**
   - Ships with a shared auxiliary **Field-of-View (FOV) Estimator** (`depthpro2-fov`, 23.5M params / 94 MB) that predicts horizontal/vertical focal length in pixels (`f_px`) when EXIF metadata is absent (such as stitched map tiles, fictional artwork, or drone videos).
3. **Four Model Scales**

| Variant | CLI Flag (`-m`) | Backbone | Parameters | Checkpoint Size | Ideal Target |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Small (S)** | `depthpro2-s` | `vit_small_patch16_dinov3` | 24M | 97 MB | Real-time laptops, edge, fast iteration |
| **Base (B)** | `depthpro2-b` | `vit_base_patch16_dinov3` | 97M | 386 MB | Best speed/detail balance on Apple Silicon (M1–M4) |
| **Large (L)** | `depthpro2-l` | `vit_large_patch16_dinov3` | 334M | 1.3 GB | High-detail offline terrain generation |
| **Huge+ (H+)** | `depthpro2-h` | `vit_huge_plus_patch16_dinov3` | 872M | 3.5 GB | Maximum sharpness & temporal stability |

---

## Apple Silicon Performance & Deployment Options

- **PyTorch Metal (MPS)**: Runs natively on Apple Silicon GPUs (`mps`) using `bfloat16` / `float16` autocast out of the box via `uv` (`torch>=2.7`).
- **Core ML (`.mlpackage`)**: Apple provides official Core ML export tooling (`python -m demo.export stream`) and pre-exported packages on Hugging Face (`apple/DepthPro2-coreml`) targeting macOS 15+.
  - Reference Core ML GPU speeds (Apple M4 Max):
    - `depthpro2-s` single-frame (`-w 1`): **10 ms (102 fps)** at 640×360, **19 ms (53 fps)** at 896×512.
    - `depthpro2-s` streaming (`-w 4`): **13 ms (76 fps)** at 640×360, **30 ms (33 fps)** at 896×512.
    - `depthpro2-b` single-frame (`-w 1`): **26 ms (39 fps)** at 640×360, **48 ms (20 fps)** at 896×512.

---

## Adapting Depth Pro 2 to Top-Down Terrain Reconstruction in Unity

When estimating elevation from **top-down (nadir / aerial / satellite / fictional map)** imagery:

1. **Depth vs. Elevation Polarity**
   - Depth Pro 2 predicts `depth` $z(x, y)$ as distance along the camera viewing axis (meters) and normalized inverse depth:
     $$\text{inv\_depth}(x, y) = \frac{f_{\text{px}}}{W \cdot z(x, y)}$$
   - For a downward-looking camera, **peaks, ridges, and buildings are closer to the camera** (smaller $z$, **larger** `inv_depth`), while **valleys, craters, and water are farther** (larger $z$, **smaller** `inv_depth`).
   - Therefore, normalized `inverse_depth` maps directly to **elevation** (white = high, black = low), with an optional `invert` flag for oblique or artistic inputs where depth cues reverse.
2. **Robust Range Normalization & 16-Bit Precision**
   - Raw monocular depth maps can contain extreme outlier pixels (water reflections, cloud edges, distant borders). Clipping to robust percentiles (default `p2`–`p98`) preserves fine topographic relief across the full 16-bit (`0..65535`) dynamic range required by Unity `TerrainData.SetHeights`.
3. **Physical Scale Mapping**
   - For real-world Slippy Map tiles (Web Mercator EPSG:3857), ground resolution in meters/pixel at latitude $\phi$ and zoom $Z$ is:
     $$\text{m/px} = \frac{156543.03392 \cdot \cos(\phi)}{2^Z}$$
   - Multiplying by stitched tile pixel dimensions provides real-world terrain footprint dimensions saved into `metadata.json` for Unity.

---

## References & Licensing

- **Paper**: Hu et al., *Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth*, arXiv:2611.26603 (2026).
- **Official Code**: [apple-aiml-research/ml-depth-pro-2](https://github.com/apple-aiml-research/ml-depth-pro-2)
- **Hugging Face Models**: [apple/DepthPro2](https://huggingface.co/apple/DepthPro2) & [apple/DepthPro2-coreml](https://huggingface.co/apple/DepthPro2-coreml)
- **License**: Apple Machine Learning Research Model License — strictly for **non-commercial scientific research and academic development**.
