from __future__ import annotations

from pixel_models import PixelEnsemble
from utils.image_io import create_demo_image, load_image


def test_pixel_ensemble_returns_three_models_and_ensemble(tmp_path):
    image = load_image(create_demo_image(tmp_path / "sample.png", "fake"))
    scores = PixelEnsemble(model_dir=tmp_path / "models", prefer_torch=False).analyze(image)
    assert {"pixel_efficientnet", "pixel_alexnet", "pixel_googlenet", "pixel_ensemble"} <= set(scores)
    assert all(0.0 <= score.score <= 1.0 for score in scores.values())
