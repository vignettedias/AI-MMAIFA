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
