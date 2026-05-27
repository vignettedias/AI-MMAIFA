from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops, ImageStat

from utils.math_utils import clip01
from utils.types import StreamScore


class ELAAnalyzer:
    """Error Level Analysis using recompression residuals."""

    name = "forensic_ela"

    def __init__(self, jpeg_quality: int = 90):
        self.jpeg_quality = jpeg_quality

    def analyze(self, image: Image.Image) -> StreamScore:
        original = image.convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(original, compressed)
        stat = ImageStat.Stat(diff)
        mean_diff = float(sum(stat.mean) / (3.0 * 255.0))
        extrema = diff.getextrema()
        max_diff = max(channel[1] for channel in extrema) / 255.0
        arr = np.asarray(diff, dtype=np.float32) / 255.0
        p95 = float(np.percentile(arr, 95))

        score = clip01((mean_diff * 4.0) + (p95 * 1.8) + (max_diff * 0.15))
        return StreamScore(
            name=self.name,
            score=score,
            details={
                "jpeg_quality": self.jpeg_quality,
                "mean_residual": mean_diff,
                "p95_residual": p95,
                "max_residual": max_diff,
            },
        )
