from __future__ import annotations

from PIL import Image

from forensics.ela import ELAAnalyzer
from forensics.frequency import FrequencyAnalyzer
from forensics.noise import NoiseAnalyzer
from utils.math_utils import safe_mean
from utils.types import StreamScore


class ForensicAnalyzer:
    """Runs all forensic submodules and returns normalized fake-evidence scores."""

    def __init__(self):
        self.analyzers = [ELAAnalyzer(), FrequencyAnalyzer(), NoiseAnalyzer()]

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        scores = {analyzer.name: analyzer.analyze(image) for analyzer in self.analyzers}
        ensemble = safe_mean([score.score for score in scores.values()])
        scores["forensic_ensemble"] = StreamScore(
            name="forensic_ensemble",
            score=ensemble,
            details={"members": list(scores.keys())},
        )
        return scores
