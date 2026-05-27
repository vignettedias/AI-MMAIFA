from __future__ import annotations

from PIL import Image

from forensics.advanced import (
    AdvancedFrequencyAnalyzer,
    AdvancedNoiseForensicsAnalyzer,
    DiffusionTraceAnalyzer,
    ProvenanceConsistencyAnalyzer,
)
from forensics.camera import CameraProvenanceAnalyzer
from forensics.compression import CompressionAnalyzer
from forensics.ela import ELAAnalyzer
from forensics.frequency import FrequencyAnalyzer
from forensics.manipulation import ManipulationAnalyzer
from forensics.metadata import MetadataProvenanceAnalyzer
from forensics.noise import NoiseAnalyzer
from forensics.spectral import SpectralIntelligenceAnalyzer
from utils.math_utils import safe_mean
from utils.types import StreamScore


class ForensicAnalyzer:
    """Runs all forensic submodules and returns normalized fake-evidence scores."""

    def __init__(self):
        self.analyzers = [
            ELAAnalyzer(),
            FrequencyAnalyzer(),
            NoiseAnalyzer(),
            CompressionAnalyzer(),
            SpectralIntelligenceAnalyzer(),
            CameraProvenanceAnalyzer(),
            ManipulationAnalyzer(),
            MetadataProvenanceAnalyzer(),
            AdvancedNoiseForensicsAnalyzer(),
            DiffusionTraceAnalyzer(),
            AdvancedFrequencyAnalyzer(),
            ProvenanceConsistencyAnalyzer(),
        ]

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        scores: dict[str, StreamScore] = {}
        for analyzer in self.analyzers:
            result = analyzer.analyze(image)
            if isinstance(result, dict):
                scores.update(result)
            else:
                scores[analyzer.name] = result
        ensemble = safe_mean([score.score for score in scores.values()])
        scores["forensic_ensemble"] = StreamScore(
            name="forensic_ensemble",
            score=ensemble,
            details={
                "members": list(scores.keys()),
                "policy": "camera provenance, noise, diffusion, frequency, manipulation, and metadata evidence",
            },
        )
        return scores
