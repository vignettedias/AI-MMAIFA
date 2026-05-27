from __future__ import annotations

from pathlib import Path

from pixel_models.base import PixelModel


class EfficientNetPipeline(PixelModel):
    def __init__(self, model_dir: str | Path = "models", prefer_torch: bool = True):
        super().__init__("efficientnet", model_dir=model_dir, prefer_torch=prefer_torch)
