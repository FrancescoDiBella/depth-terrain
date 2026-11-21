# Depth Terrain Pipeline (Apple Depth Pro 2)

Python (`uv`) inference pipeline powered by [Apple Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2) for generating Unity 6 Terrain assets from real-world maps, local top-down images, aerial video, and fictional worlds.

## Quick Start

```bash
uv sync --python 3.12 --extra dev
uv run depth-terrain download --all-laptop
uv run depth-terrain from-map --place "Mount Vesuvius" --zoom 14 --grid 3 -m depthpro2-b -o out/vesuvius
uv run depth-terrain from-fictional --biome volcanic_island --seed 42 -m depthpro2-b -o out/volcanic
```
