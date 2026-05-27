from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.math_utils import clip01


@dataclass
class CalibrationReport:
    ece: float
    brier: float
    temperature: float
    samples: int


class TemperatureScaler:
    """Small binary temperature-scaling calibrator."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.05, float(temperature))

    def fit(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
        grid: Sequence[float] | None = None,
    ) -> CalibrationReport:
        probs = np.asarray([clip01(value) for value in probabilities], dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        candidates = np.asarray(list(grid or np.linspace(0.55, 3.0, 40)), dtype=np.float64)
        losses = [self._nll(self.transform(probs, temp), y) for temp in candidates]
        self.temperature = float(candidates[int(np.argmin(losses))])
        calibrated = self.transform(probs)
        return CalibrationReport(
            ece=expected_calibration_error(calibrated, y),
            brier=float(np.mean((calibrated - y) ** 2)) if y.size else 0.0,
            temperature=self.temperature,
            samples=int(y.size),
        )

    def transform(self, probabilities: Sequence[float] | np.ndarray, temperature: float | None = None) -> np.ndarray:
        probs = np.asarray([clip01(value) for value in probabilities], dtype=np.float64)
        temp = self.temperature if temperature is None else max(0.05, float(temperature))
        logits = np.log((probs + 1e-8) / (1.0 - probs + 1e-8))
        return 1.0 / (1.0 + np.exp(-logits / temp))

    @staticmethod
    def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        probs = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


class DirichletCalibrator:
    """Lightweight evidence shrinkage calibrator for binary Dirichlet evidence."""

    def __init__(self, strength: float = 2.0):
        self.strength = max(0.1, float(strength))

    def transform(self, probability: float, uncertainty: float) -> float:
        probability = clip01(probability)
        uncertainty = clip01(uncertainty)
        alpha_fake = 1.0 + probability * self.strength * (1.0 - uncertainty)
        alpha_real = 1.0 + (1.0 - probability) * self.strength * (1.0 - uncertainty)
        return clip01(alpha_fake / (alpha_fake + alpha_real))


def expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    bins: int = 15,
) -> float:
    probs = np.asarray([clip01(value) for value in probabilities], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs >= low) & (probs < high if high < 1.0 else probs <= high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(np.maximum(probs[mask], 1.0 - probs[mask])))
        accuracy = float(np.mean((probs[mask] >= 0.5) == (y[mask] >= 0.5)))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return clip01(ece)
