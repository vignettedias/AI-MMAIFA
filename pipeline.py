from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import hashlib

from forensics import ForensicAnalyzer
from fusion import EvidenceFusionEngine, FeatureAggregator
from meta import DynamicMemoryLearner, MetaClassifier
from pixel_models import PixelEnsemble
from semantic import CLIPSemanticAnalyzer, MultiAgentSemanticAnalyzer
from uncertainty import HumanReviewPolicy, UncertaintyEstimator
from utils.image_io import image_to_png_bytes, load_image
from utils.math_utils import clip01
from utils.types import PredictionResult, StreamScore


class AuthenticityPipeline:
    """End-to-end multi-stream authenticity verification pipeline."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        prefer_torch: bool = True,
        use_clip: bool = False,
        auto_learn: bool = True,
        auto_learn_confidence: float = 70.0,
        feature_cache_size: int = 128,
    ):
        self.model_dir = Path(model_dir)
        self.auto_learn = auto_learn
        self.auto_learn_confidence = auto_learn_confidence
        self.feature_cache_size = feature_cache_size
        self._feature_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.pixel = PixelEnsemble(model_dir=self.model_dir, prefer_torch=prefer_torch)
        self.forensics = ForensicAnalyzer()
        self.semantic = CLIPSemanticAnalyzer(use_clip=use_clip)
        self.semantic_agents = MultiAgentSemanticAnalyzer()
        self.fusion = FeatureAggregator()
        self.evidence_fusion = EvidenceFusionEngine()
        self.uncertainty = UncertaintyEstimator()
        self.review_policy = HumanReviewPolicy()
        self.meta = MetaClassifier(
            model_dir=self.model_dir,
            feature_names=self.fusion.feature_names,
        )
        self.memory = DynamicMemoryLearner(
            model_dir=self.model_dir,
            feature_names=self.fusion.feature_names,
        )

    def predict(self, image_input: Any) -> PredictionResult:
        image = load_image(image_input)
        image_hash = _image_hash(image)
        cached = self._get_cached_features(image_hash)
        if cached is None:
            pixel_scores = self.pixel.analyze(image)
            forensic_scores = self.forensics.analyze(image)
            semantic_scores = self.semantic.analyze(image)
            semantic_scores.update(self.semantic_agents.analyze(image))
            feature_vector = self.fusion.transform(pixel_scores, forensic_scores, semantic_scores)
            features = feature_vector.as_dict()
            stream_scores = {}
            stream_scores.update(_scores_to_float(pixel_scores))
            stream_scores.update(_scores_to_float(forensic_scores))
            stream_scores.update(_scores_to_float(semantic_scores))
            stream_details = {}
            stream_details.update(_score_details(pixel_scores))
            stream_details.update(_score_details(forensic_scores))
            stream_details.update(_score_details(semantic_scores))
            self._store_cached_features(image_hash, features, stream_scores, stream_details)
            cache_status = "miss"
        else:
            features = dict(cached["features"])
            stream_scores = dict(cached["stream_scores"])
            stream_details = dict(cached.get("stream_details", {}))
            feature_vector = self.fusion.from_dict(features)
            cache_status = "hit"

        meta_probability = clip01(self.meta.predict_proba(features))
        evidence_fusion = self.evidence_fusion.fuse(
            features=features,
            prior_probability=meta_probability,
        )
        meta_probability = evidence_fusion.probability
        fake_probability, memory_decision = self.memory.adjust_probability(
            base_probability=meta_probability,
            features=features,
            image_hash=image_hash,
        )
        model_label = "FAKE" if fake_probability >= 0.5 else "REAL"
        uncertainty = self.uncertainty.estimate(
            features=features,
            fake_probability=fake_probability,
            meta_decision=self.meta.last_decision,
            memory_decision=memory_decision,
        )
        review_decision = self.review_policy.decide(
            fake_probability=fake_probability,
            features=features,
            uncertainty=uncertainty,
        )
        label = review_decision["decision_label"]
        confidence = float(review_decision["confidence"])
        decision_band = _decision_band(fake_probability)
        auto_memory = self._auto_learn_if_confident(
            label=model_label,
            confidence=confidence,
            features=features,
            image_hash=image_hash,
            memory_strategy=memory_decision.get("strategy"),
        )

        details = {
            "meta_backend": self.meta.backend,
            "meta_decision": self.meta.last_decision,
            "evidence_fusion": evidence_fusion.details,
            "dynamic_memory": memory_decision,
            "auto_memory": auto_memory,
            "model_label": model_label,
            "decision_label": label,
            "review_decision": review_decision,
            "uncertainty": uncertainty,
            "decision_band": decision_band,
            "decision_message": _decision_message(decision_band),
            "feature_cache": cache_status,
            "feature_order": feature_vector.names,
            "stream_details": stream_details,
            "pixel_backends": {
                name: model.backend for name, model in self.pixel.models.items()
            },
        }
        return PredictionResult(
            label=label,
            confidence=confidence,
            fake_probability=fake_probability,
            features=features,
            stream_scores=stream_scores,
            image_hash=image_hash,
            details=details,
        )

    def learn_from_feedback(
        self,
        *,
        label: str | int,
        features: dict[str, float],
        image_hash: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        return self.memory.learn(
            label=label,
            features=features,
            image_hash=image_hash,
            note=note,
            trust=1.0,
            allow_exact_override=True,
        )

    def _auto_learn_if_confident(
        self,
        *,
        label: str,
        confidence: float,
        features: dict[str, float],
        image_hash: str,
        memory_strategy: str | None,
    ) -> dict[str, Any]:
        if not self.auto_learn:
            return {"status": "disabled"}
        if memory_strategy == "exact_image_memory":
            return {"status": "skipped_exact_memory"}
        consensus = self._auto_consensus_label(features)
        if confidence < self.auto_learn_confidence:
            if consensus is not None and consensus["label"] == label:
                trust = min(0.55, max(0.30, float(consensus["trust"])))
                result = self.memory.learn(
                    label=label,
                    features=features,
                    image_hash=image_hash,
                    source="auto_stream_consensus",
                    note=(
                        "Auto-learned from multi-stream consensus "
                        f"({consensus['reason']}, strength {consensus['strength']:.3f})."
                    ),
                    trust=trust,
                    allow_exact_override=False,
                )
                return {
                    **result,
                    "status": "learned_stream_consensus",
                    "confidence": round(confidence, 2),
                    "consensus": consensus,
                }
            return {
                "status": "skipped_low_confidence",
                "confidence": round(confidence, 2),
                "threshold": self.auto_learn_confidence,
            }
        trust = min(0.60, max(0.25, (confidence - 50.0) / 100.0))
        return self.memory.learn(
            label=label,
            features=features,
            image_hash=image_hash,
            source="auto_confident_prediction",
            note=f"Auto-learned from high-confidence prediction ({confidence:.2f}%).",
            trust=trust,
            allow_exact_override=False,
        )

    @staticmethod
    def _auto_consensus_label(features: dict[str, float]) -> dict[str, Any] | None:
        pixel = features.get("pixel_ensemble", 0.0)
        forensic = features.get("forensic_ensemble", 0.0)
        fft = features.get("forensic_fft", 0.0)
        ela = features.get("forensic_ela", 0.0)
        semantic = max(
            features.get("semantic_anomaly", 0.0),
            features.get("semantic_portrait_synthetic", 0.0),
            features.get("semantic_stylized_composite", 0.0),
            features.get("semantic_low_resolution_composite", 0.0),
            features.get("semantic_nature_render", 0.0),
            features.get("semantic_architecture_render", 0.0),
            features.get("semantic_multi_agent", 0.0),
            features.get("semantic_physical_realism", 0.0),
            features.get("semantic_typography", 0.0),
        )
        fake_strength = min(
            clip01((pixel - 0.52) / 0.22),
            clip01((forensic - 0.40) / 0.18),
            clip01((semantic - 0.50) / 0.24),
        )
        if fake_strength >= 0.22:
            return {
                "label": "FAKE",
                "strength": round(fake_strength, 6),
                "trust": round(0.30 + 0.35 * fake_strength, 6),
                "reason": "moderate pixel/forensic/semantic fake agreement",
            }

        real_strength = min(
            clip01((0.34 - pixel) / 0.22),
            clip01((0.38 - forensic) / 0.18),
            clip01((0.40 - semantic) / 0.24),
            clip01((0.62 - fft) / 0.24),
            clip01((0.16 - ela) / 0.16),
        )
        if real_strength >= 0.45:
            return {
                "label": "REAL",
                "strength": round(real_strength, 6),
                "trust": round(0.25 + 0.30 * real_strength, 6),
                "reason": "low artifact consensus",
            }
        return None

    def memory_stats(self) -> dict[str, Any]:
        return self.memory.stats()

    def _get_cached_features(self, image_hash: str) -> dict[str, Any] | None:
        cached = self._feature_cache.get(image_hash)
        if cached is None:
            return None
        self._feature_cache.move_to_end(image_hash)
        return cached

    def _store_cached_features(
        self,
        image_hash: str,
        features: dict[str, float],
        stream_scores: dict[str, float],
        stream_details: dict[str, Any],
    ) -> None:
        if self.feature_cache_size <= 0:
            return
        self._feature_cache[image_hash] = {
            "features": dict(features),
            "stream_scores": dict(stream_scores),
            "stream_details": dict(stream_details),
        }
        self._feature_cache.move_to_end(image_hash)
        while len(self._feature_cache) > self.feature_cache_size:
            self._feature_cache.popitem(last=False)


def _scores_to_float(scores: dict[str, StreamScore]) -> dict[str, float]:
    return {name: clip01(score.score) for name, score in scores.items()}


def _score_details(scores: dict[str, StreamScore]) -> dict[str, dict[str, Any]]:
    return {name: score.details for name, score in scores.items()}


def _image_hash(image) -> str:
    return hashlib.sha256(image_to_png_bytes(image)).hexdigest()


def _decision_band(fake_probability: float) -> str:
    if fake_probability >= 0.80:
        return "decisive_ai_generated"
    if fake_probability >= 0.65:
        return "likely_ai_generated"
    if fake_probability >= 0.50:
        return "ai_signals_present"
    if fake_probability <= 0.20:
        return "decisive_real"
    if fake_probability <= 0.35:
        return "likely_real"
    return "uncertain_real"


def _decision_message(decision_band: str) -> str:
    messages = {
        "decisive_ai_generated": "Strong multi-stream evidence indicates an AI-generated image.",
        "likely_ai_generated": "Multiple streams indicate this is likely AI-generated.",
        "ai_signals_present": "AI-generation signals are present, but confidence is moderate.",
        "decisive_real": "Streams show low artifact evidence; image is likely real.",
        "likely_real": "Evidence leans real, with limited synthetic artifacts.",
        "uncertain_real": "Evidence is mixed; the current threshold leans real.",
    }
    return messages[decision_band]
