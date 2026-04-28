from __future__ import annotations

from pipeline import AuthenticityPipeline
from utils.image_io import create_demo_image


def test_full_pipeline_predicts_end_to_end(tmp_path):
    image_path = create_demo_image(tmp_path / "image.png", "real")
    pipeline = AuthenticityPipeline(model_dir=tmp_path / "models", prefer_torch=False)
    result = pipeline.predict(image_path)
    assert result.label in {"REAL", "FAKE"}
    assert 0.0 <= result.fake_probability <= 1.0
    assert 50.0 <= result.confidence <= 100.0
    assert "semantic_anomaly" in result.features
    assert "pixel_ensemble" in result.features
    assert "forensic_ensemble" in result.features


def test_pipeline_auto_learns_high_confidence_predictions(tmp_path):
    image_path = create_demo_image(tmp_path / "image.png", "real")
    pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
        auto_learn_confidence=50.0,
    )
    result = pipeline.predict(image_path)

    assert result.details["auto_memory"]["last_action"] in {"created", "updated"}
    assert pipeline.memory.stats()["automatic_examples"] >= 1
