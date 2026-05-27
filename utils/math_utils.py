from __future__ import annotations

import math

import numpy as np


def clip01(value: float) -> float:
    if math.isnan(float(value)) or math.isinf(float(value)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def normalize_minmax(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clip01((float(value) - low) / (high - low))


def safe_mean(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return clip01(float(np.mean(values)))


def entropy01(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return clip01(float(entropy / math.log2(bins)))
