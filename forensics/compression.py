from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, normalize_minmax
from utils.types import StreamScore


class CompressionAnalyzer:
    """JPEG/block artifact analysis for compression and resampling traces."""

    name = "forensic_compression"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> StreamScore:
        gray = image_to_array(image, size=(self.size, self.size), grayscale=True)
        vertical = np.abs(np.diff(gray, axis=1))
        horizontal = np.abs(np.diff(gray, axis=0))

        grid_v = vertical[:, 7::8]
        grid_h = horizontal[7::8, :]
        non_grid_v = np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1)
        non_grid_h = np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0)

        boundary_energy = float((np.mean(grid_v) + np.mean(grid_h)) / 2.0)
        interior_energy = float((np.mean(non_grid_v) + np.mean(non_grid_h)) / 2.0 + 1e-8)
        block_ratio = boundary_energy / interior_energy

        laplacian = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        ringing = float(np.percentile(np.abs(laplacian), 95))
        gradient = np.zeros_like(gray)
        gradient[:, :-1] += vertical
        gradient[:-1, :] += horizontal
        flat_regions = float(np.mean(gradient < 0.012))

        score = clip01(
            0.45 * normalize_minmax(block_ratio, 1.08, 1.55)
            + 0.35 * normalize_minmax(ringing, 0.035, 0.16)
            + 0.20 * normalize_minmax(flat_regions, 0.18, 0.55)
        )
        return StreamScore(
            name=self.name,
            score=score,
            details={
                "block_boundary_ratio": block_ratio,
                "ringing_p95": ringing,
                "flat_region_ratio": flat_regions,
            },
        )
