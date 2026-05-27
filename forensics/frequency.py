from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, normalize_minmax
from utils.types import StreamScore


class FrequencyAnalyzer:
    """FFT-based detector for unusual high-frequency distributions."""

    name = "forensic_fft"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> StreamScore:
        gray = image_to_array(image, size=(self.size, self.size), grayscale=True)
        gray = gray - float(np.mean(gray))
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(spectrum))

        h, w = magnitude.shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius_norm = radius / radius.max()

        total_energy = float(np.sum(magnitude) + 1e-8)
        high_energy = float(np.sum(magnitude[radius_norm > 0.55]) / total_energy)
        mid_energy = float(
            np.sum(magnitude[(radius_norm > 0.22) & (radius_norm <= 0.55)])
            / total_energy
        )
        ring_variance = float(np.var(magnitude[radius_norm > 0.55]))

        high_score = normalize_minmax(high_energy, 0.20, 0.48)
        mid_score = normalize_minmax(mid_energy, 0.28, 0.55)
        variance_score = normalize_minmax(ring_variance, 0.015, 0.16)
        score = clip01(0.55 * high_score + 0.25 * mid_score + 0.20 * variance_score)

        return StreamScore(
            name=self.name,
            score=score,
            details={
                "high_frequency_energy": high_energy,
                "mid_frequency_energy": mid_energy,
                "high_ring_variance": ring_variance,
            },
        )
