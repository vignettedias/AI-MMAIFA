from __future__ import annotations

from pathlib import Path

from PIL import Image

from pixel_models.alexnet import AlexNetPipeline
from pixel_models.base import PixelTrainingResult
from pixel_models.efficientnet import EfficientNetPipeline
from pixel_models.googlenet import GoogLeNetPipeline
from utils.dataset import ImageSample
from utils.math_utils import safe_mean
from utils.types import StreamScore


class PixelEnsemble:
    """Three-stream pixel-level ensemble: EfficientNet, AlexNet, GoogLeNet."""

    def __init__(self, model_dir: str | Path = "models", prefer_torch: bool = True):
        self.models = {
            "efficientnet": EfficientNetPipeline(model_dir, prefer_torch=prefer_torch),
            "alexnet": AlexNetPipeline(model_dir, prefer_torch=prefer_torch),
            "googlenet": GoogLeNetPipeline(model_dir, prefer_torch=prefer_torch),
        }

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        scores = {model.stream_name: model.predict(image) for model in self.models.values()}
        ensemble = safe_mean([score.score for score in scores.values()])
        scores["pixel_ensemble"] = StreamScore(
            name="pixel_ensemble",
            score=ensemble,
            details={"members": list(scores.keys())},
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
