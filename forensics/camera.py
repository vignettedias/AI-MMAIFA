from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, normalize_minmax
from utils.types import StreamScore


class CameraProvenanceAnalyzer:
    """Estimates whether image statistics look compatible with a camera pipeline."""

    name = "forensic_camera_provenance"

    def __init__(self, size: int = 256):
        self.size = size

    def analyze(self, image: Image.Image) -> StreamScore:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)

        residual = _high_pass(gray)
        block = 2
        bayer_cells = [
            residual[y::block, x::block]
            for y in range(block)
            for x in range(block)
        ]
        cell_energy = np.asarray([float(np.mean(np.abs(cell))) for cell in bayer_cells])
        cfa_coherence = float(np.std(cell_energy) / (np.mean(cell_energy) + 1e-8))

        grad_r = _gradient_magnitude(arr[:, :, 0])
        grad_g = _gradient_magnitude(arr[:, :, 1])
        grad_b = _gradient_magnitude(arr[:, :, 2])
        edge_mask = _gradient_magnitude(gray) > np.percentile(_gradient_magnitude(gray), 72)
        if np.any(edge_mask):
            rg_corr = _corr(grad_r[edge_mask], grad_g[edge_mask])
            gb_corr = _corr(grad_g[edge_mask], grad_b[edge_mask])
        else:
            rg_corr = 1.0
            gb_corr = 1.0
        over_aligned_edges = float((rg_corr + gb_corr) / 2.0)

        intensity_bins = np.linspace(0.0, 1.0, 9)
        noise_curve = []
        for low, high in zip(intensity_bins[:-1], intensity_bins[1:], strict=True):
            mask = (gray >= low) & (gray < high)
            if np.mean(mask) > 0.02:
                noise_curve.append(float(np.mean(np.abs(residual[mask]))))
        if len(noise_curve) > 2:
            sensor_noise_monotonicity = _corr(
                np.arange(len(noise_curve), dtype=np.float64),
                np.asarray(noise_curve, dtype=np.float64),
            )
        else:
            sensor_noise_monotonicity = 0.0

        flat_mask = _gradient_magnitude(gray) < np.percentile(_gradient_magnitude(gray), 35)
        flat_noise = float(np.mean(np.abs(residual[flat_mask]))) if np.any(flat_mask) else 0.0
        edge_noise = float(np.mean(np.abs(residual[edge_mask]))) if np.any(edge_mask) else 0.0
        residual_edge_coupling = edge_noise / (flat_noise + 1e-8)

        missing_cfa = clip01(1.0 - normalize_minmax(cfa_coherence, 0.035, 0.18))
        too_aligned = normalize_minmax(over_aligned_edges, 0.86, 0.985)
        weak_shot_noise = clip01(1.0 - normalize_minmax(sensor_noise_monotonicity, 0.12, 0.62))
        residual_overcoupled = normalize_minmax(residual_edge_coupling, 2.2, 5.0)

        score = clip01(
            0.30 * missing_cfa
            + 0.24 * too_aligned
            + 0.24 * weak_shot_noise
            + 0.22 * residual_overcoupled
        )
        return StreamScore(
            self.name,
            score,
            {
                "cfa_coherence": cfa_coherence,
                "missing_cfa": missing_cfa,
                "edge_channel_alignment": over_aligned_edges,
                "sensor_noise_monotonicity": sensor_noise_monotonicity,
                "residual_edge_coupling": residual_edge_coupling,
            },
        )


def _high_pass(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    smooth = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return gray - smooth


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(values, axis=0, append=values[-1:, :]))
    gx = np.abs(np.diff(values, axis=1, append=values[:, -1:]))
    return gx + gy


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
