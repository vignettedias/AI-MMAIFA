from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, normalize_minmax
from utils.types import StreamScore


class NoiseAnalyzer:
    """Detects block-wise inconsistency in residual sensor-like noise."""

    name = "forensic_noise"

    def __init__(self, size: int = 256, block_size: int = 32):
        self.size = size
        self.block_size = block_size

    def analyze(self, image: Image.Image) -> StreamScore:
        gray = image_to_array(image, size=(self.size, self.size), grayscale=True)
        smoothed = self._box_blur(gray, kernel_size=5)
        residual = gray - smoothed
        block_stds = self._block_stds(residual)

        mean_noise = float(np.mean(np.abs(residual)))
        std_variation = float(np.std(block_stds))
        coeff_var = float(std_variation / (float(np.mean(block_stds)) + 1e-6))
        grid_score = self._grid_discontinuity(gray)

        variation_score = normalize_minmax(coeff_var, 0.15, 0.85)
        residual_score = normalize_minmax(mean_noise, 0.015, 0.090)
        score = clip01(0.55 * variation_score + 0.25 * residual_score + 0.20 * grid_score)

        return StreamScore(
            name=self.name,
            score=score,
            details={
                "mean_abs_residual": mean_noise,
                "block_noise_coeff_var": coeff_var,
                "grid_discontinuity": grid_score,
            },
        )

    @staticmethod
    def _box_blur(arr: np.ndarray, kernel_size: int) -> np.ndarray:
        pad = kernel_size // 2
        padded = np.pad(arr, pad_width=pad, mode="reflect")
        out = np.zeros_like(arr, dtype=np.float32)
        for y in range(kernel_size):
            for x in range(kernel_size):
                out += padded[y : y + arr.shape[0], x : x + arr.shape[1]]
        return out / float(kernel_size * kernel_size)

    def _block_stds(self, residual: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for y in range(0, residual.shape[0] - self.block_size + 1, self.block_size):
            for x in range(0, residual.shape[1] - self.block_size + 1, self.block_size):
                block = residual[y : y + self.block_size, x : x + self.block_size]
                values.append(float(np.std(block)))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _grid_discontinuity(gray: np.ndarray) -> float:
        vertical = np.abs(np.diff(gray, axis=1))
        horizontal = np.abs(np.diff(gray, axis=0))
        grid_v = float(np.mean(vertical[:, 7::8])) if vertical.shape[1] > 8 else 0.0
        grid_h = float(np.mean(horizontal[7::8, :])) if horizontal.shape[0] > 8 else 0.0
        base = float((np.mean(vertical) + np.mean(horizontal)) / 2.0 + 1e-6)
        return clip01(((grid_v + grid_h) / 2.0) / (base * 2.5))
