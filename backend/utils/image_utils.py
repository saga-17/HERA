"""Image processing utilities for HERA-VLM."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    """Load image as RGB PIL Image."""
    return Image.open(path).convert("RGB")


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    """Crop image region from bounding box [x1, y1, x2, y2]."""
    if len(bbox) != 4:
        return image
    x1, y1, x2, y2 = bbox
    w, h = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def generate_grid_regions(image: Image.Image, grid: int = 3) -> list[tuple[list[int], Image.Image]]:
    """
    Split image into a grid of regions for visual evidence retrieval.
    Returns list of (bbox, cropped_image) tuples.
    """
    w, h = image.size
    cell_w, cell_h = w // grid, h // grid
    regions = []

    for row in range(grid):
        for col in range(grid):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w if col < grid - 1 else w
            y2 = y1 + cell_h if row < grid - 1 else h
            bbox = [x1, y1, x2, y2]
            regions.append((bbox, crop_region(image, bbox)))

    return regions
