from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamScore:
    """Normalized score from an individual stream.

    By convention, `score` is clipped to [0, 1]. For fused features, higher
    values indicate stronger evidence that an image is AI-generated.
    """

    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    names: list[str]
    values: list[float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


@dataclass
class PredictionResult:
    label: str
    confidence: float
    fake_probability: float
    features: dict[str, float]
    stream_scores: dict[str, float]
    image_hash: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_details: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "fake_probability": round(self.fake_probability, 6),
            "image_hash": self.image_hash,
            "features": {key: round(value, 6) for key, value in self.features.items()},
            "stream_scores": {
                key: round(value, 6) for key, value in self.stream_scores.items()
            },
        }
        if include_details:
            payload["details"] = self.details
        return payload
