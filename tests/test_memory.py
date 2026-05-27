from __future__ import annotations

from meta import DynamicMemoryLearner
from fusion import FeatureAggregator


def test_dynamic_memory_exact_image_override(tmp_path):
    feature_names = FeatureAggregator.feature_names
    features = {name: 0.2 for name in feature_names}
    memory = DynamicMemoryLearner(tmp_path / "models", feature_names=feature_names)

    memory.learn(label="FAKE", features=features, image_hash="abc123")
    probability, details = memory.adjust_probability(
        base_probability=0.1,
        features=features,
        image_hash="abc123",
    )

    assert probability == 1.0
    assert details["strategy"] == "exact_image_memory"


def test_auto_memory_does_not_exact_override(tmp_path):
    feature_names = FeatureAggregator.feature_names
    features = {name: 0.2 for name in feature_names}
    memory = DynamicMemoryLearner(tmp_path / "models", feature_names=feature_names)

    memory.learn(
        label="FAKE",
        features=features,
        image_hash="abc123",
        source="auto_confident_prediction",
        trust=0.4,
        allow_exact_override=False,
    )
    probability, details = memory.adjust_probability(
        base_probability=0.1,
        features=features,
        image_hash="abc123",
    )

    assert probability < 1.0
    assert details["strategy"] == "memory_no_close_match"


def test_memory_guard_keeps_strong_artifact_evidence(tmp_path):
    feature_names = FeatureAggregator.feature_names
    features = {name: 0.0 for name in feature_names}
    features.update(
        {
            "pixel_ensemble": 0.18,
            "forensic_ela": 0.04,
            "forensic_fft": 0.82,
            "forensic_noise": 0.42,
            "forensic_ensemble": 0.42,
            "semantic_nature_render": 0.86,
            "semantic_architecture_render": 0.0,
        }
    )
    memory = DynamicMemoryLearner(tmp_path / "models", feature_names=feature_names)
    neighbor = dict(features)
    neighbor["semantic_nature_render"] = 0.78
    memory.learn(
        label="REAL",
        features=neighbor,
        image_hash="nearby",
        source="auto_confident_prediction",
        trust=0.6,
        allow_exact_override=False,
    )

    probability, details = memory.adjust_probability(
        base_probability=0.58,
        features=features,
        image_hash="current",
    )

    assert probability >= 0.55
    assert details["strategy"] == "nearest_feature_memory_guarded"


def test_admin_memory_replaces_lower_trust_auto_memory(tmp_path):
    feature_names = FeatureAggregator.feature_names
    features = {name: 0.2 for name in feature_names}
    memory = DynamicMemoryLearner(tmp_path / "models", feature_names=feature_names)

    memory.learn(
        label="FAKE",
        features=features,
        image_hash="abc123",
        source="auto_confident_prediction",
        trust=0.4,
        allow_exact_override=False,
    )
    memory.learn(label="REAL", features=features, image_hash="abc123")
    probability, details = memory.adjust_probability(
        base_probability=0.9,
        features=features,
        image_hash="abc123",
    )

    assert probability == 0.0
    assert details["strategy"] == "exact_image_memory"
