from __future__ import annotations

from pathlib import Path

from PIL import Image

from pixel_models.alexnet import AlexNetPipeline
from pixel_models.base import PixelTrainingResult
from pixel_models.efficientnet import EfficientNetPipeline
from pixel_models.experts import ModernPixelExpertEnsemble
from pixel_models.googlenet import GoogLeNetPipeline
from utils.dataset import ImageSample
from utils.math_utils import safe_mean
from utils.types import StreamScore


class PixelEnsemble:
    """Pixel-level ensemble with legacy compatibility and modern expert streams."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        prefer_torch: bool = True,
        enable_modern_experts: bool = True,
    ):
        self.models = {
            "efficientnet": EfficientNetPipeline(model_dir, prefer_torch=prefer_torch),
            "alexnet": AlexNetPipeline(model_dir, prefer_torch=prefer_torch),
            "googlenet": GoogLeNetPipeline(model_dir, prefer_torch=prefer_torch),
        }
        self.modern_experts = ModernPixelExpertEnsemble() if enable_modern_experts else None

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        scores = {model.stream_name: model.predict(image) for model in self.models.values()}
        legacy_ensemble = safe_mean([score.score for score in scores.values()])
        modern_ensemble = legacy_ensemble
        if self.modern_experts is not None:
            modern_scores = self.modern_experts.analyze(image)
            scores.update(modern_scores)
            modern_ensemble = modern_scores["pixel_modern_ensemble"].score
        ensemble = safe_mean([legacy_ensemble * 0.65, modern_ensemble * 1.35])
        scores["pixel_ensemble"] = StreamScore(
            name="pixel_ensemble",
            score=ensemble,
            details={
                "legacy_ensemble": legacy_ensemble,
                "modern_ensemble": modern_ensemble,
                "members": list(scores.keys()),
                "policy": "legacy CNN compatibility plus modern expert evidence",
            },
        )
        return scores

    def train_model(
        self,
        model_name: str,
        samples: list[ImageSample],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ) -> PixelTrainingResult:
        if model_name not in self.models:
            raise ValueError(f"Unknown pixel model '{model_name}'. Use one of {list(self.models)}")
        return self.models[model_name].train(samples, epochs, batch_size, learning_rate)

    def train_all(
        self,
        samples: list[ImageSample],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ) -> list[PixelTrainingResult]:
        return [
            model.train(samples, epochs, batch_size, learning_rate)
            for model in self.models.values()
        ]
