from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, entropy01, normalize_minmax
from utils.types import StreamScore


class SpectralIntelligenceAnalyzer:
    """Multi-statistic Fourier forensic analyzer."""

    name = "forensic_spectral_intelligence"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> StreamScore:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        spectrum = spectrum / (float(np.max(spectrum)) + 1e-8)

        yy, xx = np.indices(spectrum.shape)
        center = (np.asarray(spectrum.shape) - 1.0) / 2.0
        radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        radius_norm = radius / (float(np.max(radius)) + 1e-8)

        high_energy = float(np.mean(spectrum[radius_norm > 0.62]))
        mid_energy = float(np.mean(spectrum[(radius_norm > 0.24) & (radius_norm <= 0.62)]))
        low_energy = float(np.mean(spectrum[radius_norm <= 0.18]))
        high_to_mid = high_energy / (mid_energy + 1e-8)

        profile = _radial_profile(spectrum, radius_norm, bins=48)
        ring_artifact = float(np.std(np.diff(profile)))
        spectral_entropy = entropy01(spectrum, bins=64)

        vertical_band = float(np.mean(spectrum[:, self.size // 2 - 2 : self.size // 2 + 3]))
        horizontal_band = float(np.mean(spectrum[self.size // 2 - 2 : self.size // 2 + 3, :]))
        anisotropy = abs(vertical_band - horizontal_band) / (vertical_band + horizontal_band + 1e-8)

        channel_spectra = []
        for channel in range(3):
            channel_fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr[:, :, channel]))))
            channel_spectra.append(channel_fft / (float(np.max(channel_fft)) + 1e-8))
        channel_stack = np.stack(channel_spectra, axis=0)
        cross_channel_variance = float(np.mean(np.var(channel_stack, axis=0)))

        score = clip01(
            0.26 * normalize_minmax(high_to_mid, 0.72, 1.20)
            + 0.20 * normalize_minmax(ring_artifact, 0.018, 0.055)
            + 0.18 * normalize_minmax(anisotropy, 0.035, 0.18)
            + 0.18 * normalize_minmax(cross_channel_variance, 0.006, 0.028)
            + 0.12 * normalize_minmax(spectral_entropy, 0.78, 0.95)
            + 0.06 * normalize_minmax(low_energy, 0.34, 0.58)
        )
        return StreamScore(
            self.name,
            score,
            {
                "high_energy": high_energy,
                "mid_energy": mid_energy,
                "low_energy": low_energy,
                "high_to_mid": high_to_mid,
                "ring_artifact": ring_artifact,
                "spectral_entropy": spectral_entropy,
                "anisotropy": anisotropy,
                "cross_channel_variance": cross_channel_variance,
            },
        )


def _radial_profile(spectrum: np.ndarray, radius_norm: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (radius_norm >= low) & (radius_norm < high)
        values.append(float(np.mean(spectrum[mask])) if np.any(mask) else 0.0)
    return np.asarray(values, dtype=np.float64)
