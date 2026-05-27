from __future__ import annotations

from fusion import FeatureAggregator
from meta import MetaClassifier
from utils.training import generate_mock_feature_matrix


def test_meta_classifier_trains_and_predicts_probability(tmp_path):
    feature_names = FeatureAggregator.feature_names
    x, y = generate_mock_feature_matrix(feature_names)
    classifier = MetaClassifier(model_dir=tmp_path / "models", feature_names=feature_names)
    result = classifier.fit(x, y)
    probability = classifier.predict_proba(dict(zip(feature_names, x[-1], strict=True)))
    assert result["samples"] == len(x)
    assert 0.0 <= probability <= 1.0


def test_meta_classifier_uses_moderate_multi_stream_consensus(tmp_path):
    feature_names = FeatureAggregator.feature_names
    features = {name: 0.0 for name in feature_names}
    features.update(
        {
            "pixel_efficientnet": 0.58,
            "pixel_alexnet": 0.58,
            "pixel_googlenet": 0.58,
            "pixel_ensemble": 0.58,
            "forensic_ela": 0.10,
            "forensic_fft": 0.77,
            "forensic_noise": 0.54,
            "forensic_compression": 0.46,
            "forensic_ensemble": 0.47,
            "semantic_anomaly": 0.90,
            "semantic_architecture_render": 0.90,
        }
    )
    classifier = MetaClassifier(model_dir=tmp_path / "models", feature_names=feature_names)
    probability = classifier.predict_proba(features)

    assert probability > 0.8
    assert classifier.last_profile_details["adjustments"]
