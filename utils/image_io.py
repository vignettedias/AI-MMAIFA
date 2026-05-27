from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(source: str | Path | bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load an image from a path, bytes, file object, or PIL image."""

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    if hasattr(source, "read"):
        return Image.open(source).convert("RGB")
    return Image.open(Path(source)).convert("RGB")


def image_to_array(
    image: Image.Image,
    size: tuple[int, int] | None = None,
    grayscale: bool = False,
) -> np.ndarray:
    if size is not None:
        image = image.resize(size, Image.Resampling.BICUBIC)
    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def create_demo_image(path: str | Path, mode: str = "real", size: int = 128) -> Path:
    """Create small deterministic demo images for smoke tests and examples."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode.lower() == "fake":
        grid = np.indices((size, size)).sum(axis=0)
        r = ((np.sin(grid / 4.0) + 1.0) * 127.5).astype(np.uint8)
        g = ((np.cos(np.indices((size, size))[0] / 5.0) + 1.0) * 127.5).astype(
            np.uint8
        )
        b = ((np.sin(np.indices((size, size))[1] / 3.0) + 1.0) * 127.5).astype(
            np.uint8
        )
        image = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
        draw = ImageDraw.Draw(image)
        for offset in range(0, size, 16):
            draw.line((0, offset, size, size - offset), fill=(255, 64, 128), width=1)
        image = image.filter(ImageFilter.SMOOTH_MORE)
    else:
        x = np.linspace(0, 1, size, dtype=np.float32)
        y = np.linspace(0, 1, size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        r = (95 + 90 * xx + 20 * yy).clip(0, 255).astype(np.uint8)
        g = (120 + 80 * yy).clip(0, 255).astype(np.uint8)
        b = (100 + 35 * (1 - xx) + 25 * yy).clip(0, 255).astype(np.uint8)
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 8, (size, size, 3)).astype(np.int16)
        arr = np.stack([r, g, b], axis=-1).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse((size * 0.2, size * 0.25, size * 0.55, size * 0.65), outline=(60, 80, 70), width=3)
        draw.rectangle((size * 0.58, size * 0.45, size * 0.84, size * 0.75), outline=(80, 70, 60), width=2)

    image.save(path)
    return path
