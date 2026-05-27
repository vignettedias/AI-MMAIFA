from __future__ import annotations

from utils.math_utils import clip01
from utils.types import FeatureVector, StreamScore


class FeatureAggregator:
    """Builds the ordered meta-classifier feature vector."""

    feature_names = [
        "pixel_efficientnet",
        "pixel_alexnet",
        "pixel_googlenet",
        "pixel_ensemble",
        "forensic_ela",
        "forensic_fft",
        "forensic_noise",
        "forensic_ensemble",
        "semantic_anomaly",
    ]

    def transform(
        self,
        pixel_scores: dict[str, StreamScore],
        forensic_scores: dict[str, StreamScore],
        semantic_scores: dict[str, StreamScore],
    ) -> FeatureVector:
        merged = {}
        merged.update(pixel_scores)
        merged.update(forensic_scores)
        merged.update(semantic_scores)
        values = [clip01(merged.get(name, StreamScore(name, 0.0)).score) for name in self.feature_names]
        return FeatureVector(names=list(self.feature_names), values=values)
