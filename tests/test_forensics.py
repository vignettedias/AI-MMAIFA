from __future__ import annotations

from utils.image_io import create_demo_image, load_image
from forensics import (
    CameraProvenanceAnalyzer,
    CompressionAnalyzer,
    ELAAnalyzer,
    ForensicAnalyzer,
    FrequencyAnalyzer,
    ManipulationAnalyzer,
    MetadataProvenanceAnalyzer,
    NoiseAnalyzer,
    SpectralIntelligenceAnalyzer,
)


def test_forensic_modules_return_normalized_scores(tmp_path):
    image_path = create_demo_image(tmp_path / "real.png", "real")
    image = load_image(image_path)
    for analyzer in [
        ELAAnalyzer(),
        FrequencyAnalyzer(),
        NoiseAnalyzer(),
        CompressionAnalyzer(),
        SpectralIntelligenceAnalyzer(),
        CameraProvenanceAnalyzer(),
        ManipulationAnalyzer(),
        MetadataProvenanceAnalyzer(),
    ]:
        score = analyzer.analyze(image)
        assert 0.0 <= score.score <= 1.0


def test_forensic_ensemble_contains_all_scores(tmp_path):
    image = load_image(create_demo_image(tmp_path / "fake.png", "fake"))
    scores = ForensicAnalyzer().analyze(image)
    assert {
        "forensic_ela",
        "forensic_fft",
        "forensic_noise",
        "forensic_compression",
        "forensic_spectral_intelligence",
        "forensic_camera_provenance",
        "forensic_manipulation",
        "forensic_metadata_provenance",
        "forensic_ensemble",
    } <= set(scores)
    assert all(0.0 <= score.score <= 1.0 for score in scores.values())
