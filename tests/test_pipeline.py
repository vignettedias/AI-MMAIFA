from __future__ import annotations

from pipeline import AuthenticityPipeline
from utils.image_io import create_demo_image


def test_full_pipeline_predicts_end_to_end(tmp_path):
    image_path = create_demo_image(tmp_path / "image.png", "real")
    pipeline = AuthenticityPipeline(model_dir=tmp_path / "models", prefer_torch=False)
    result = pipeline.predict(image_path)
    assert result.label in {"REAL", "AI_GENERATED", "MANIPULATED", "HUMAN_REVIEW_REQUIRED"}
    assert 0.0 <= result.fake_probability <= 1.0
    assert 50.0 <= result.confidence <= 100.0
    assert "semantic_anomaly" in result.features
    assert "semantic_portrait_synthetic" in result.features
    assert "semantic_stylized_composite" in result.features
    assert "semantic_low_resolution_composite" in result.features
    assert "semantic_nature_render" in result.features
    assert "semantic_architecture_render" in result.features
    assert "pixel_ensemble" in result.features
    assert "forensic_ensemble" in result.features
    assert "forensic_compression" in result.features
    assert "forensic_spectral_intelligence" in result.features
    assert "forensic_camera_provenance" in result.features
    assert "forensic_manipulation" in result.features
    assert "forensic_metadata_provenance" in result.features
    assert "decision_band" in result.details
    assert "decision_message" in result.details
    assert "uncertainty" in result.details
    assert "review_decision" in result.details
    assert result.details["feature_cache"] == "miss"

    cached = pipeline.predict(image_path)
    assert cached.details["feature_cache"] == "hit"


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


def test_pipeline_learns_from_stream_consensus_below_confidence_threshold(tmp_path):
    pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
        auto_learn_confidence=90.0,
    )
    features = {name: 0.0 for name in pipeline.fusion.feature_names}
    features.update(
        {
            "pixel_ensemble": 0.58,
            "forensic_ensemble": 0.47,
            "forensic_fft": 0.77,
            "forensic_ela": 0.10,
            "semantic_anomaly": 0.56,
            "semantic_architecture_render": 0.70,
        }
    )
    result = pipeline._auto_learn_if_confident(
        label="FAKE",
        confidence=60.0,
        features=features,
        image_hash="consensus-image",
        memory_strategy="memory_no_close_match",
    )

    assert result["status"] == "learned_stream_consensus"
    assert pipeline.memory.stats()["automatic_examples"] == 1
