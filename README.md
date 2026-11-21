# Depth Terrain — Apple Depth Pro 2 → Unity 6 Terrain Pipeline

An end-to-end pipeline and modular **Unity 6 (URP)** toolkit that turns **top-down imagery and aerial video** — from **real-world satellite/map tiles** (Esri World Imagery, OpenStreetMap, OpenTopoMap, CartoDB, or custom XYZ tile URLs), **local aerial photos**, or **fictional top-down worlds** — into playable, lit **3D Unity Terrains** using **Apple Depth Pro 2** (`apple-aiml-research/ml-depth-pro-2`).

> See [docs/DEPTH_PRO_2_RESEARCH.md](docs/DEPTH_PRO_2_RESEARCH.md) for the full technical research report on Apple Depth Pro 2.

## Project Structure

- `docs/` — Research notes and architecture documentation.
- `pipeline/` — Python (`uv`) Apple Depth Pro 2 inference CLI, Slippy Map tile fetcher & geocoder, fictional top-down generator, and Unity 16-bit RAW heightmap exporter.
- `DepthTerrainUnity/` — Unity 6 URP project containing the modular `Assets/DepthTerrain` package (Runtime + Editor Studio + CLI Commands + EditMode Tests) and the `DepthTerrainShowcase` scene.
