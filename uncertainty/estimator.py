from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.math_utils import clip01


class UncertaintyEstimator:
    """Computes OOD and calibration-style uncertainty from fused evidence."""

    def estimate(
        self,
        *,
        features: dict[str, float],
        fake_probability: float,
        meta_decision: dict[str, Any],
        memory_decision: dict[str, Any],
    ) -> dict[str, Any]:
        groups = _evidence_groups(features)
        values = np.asarray(list(groups.values()), dtype=np.float64)
        disagreement = clip01(float(np.std(values) * 2.4))
        margin_uncertainty = clip01(1.0 - abs(fake_probability - 0.5) * 2.0)

        high_evidence = float(np.max(values)) if values.size else 0.0
        low_evidence = float(np.min(values)) if values.size else 0.0
        conflict = clip01(high_evidence - low_evidence)
        sparse_known_pattern = clip01(
            1.0
            - max(
                features.get("semantic_architecture_render", 0.0),
                features.get("semantic_nature_render", 0.0),
                features.get("semantic_stylized_composite", 0.0),
                features.get("semantic_portrait_synthetic", 0.0),
                features.get("semantic_low_resolution_composite", 0.0),
                features.get("forensic_spectral_intelligence", 0.0),
                features.get("forensic_manipulation", 0.0),
            )
        )
        ood_score = clip01(
            0.40 * disagreement
            + 0.30 * margin_uncertainty
            + 0.20 * conflict
            + 0.10 * sparse_known_pattern
        )
        open_set = _open_set_scores(features, groups, fake_probability, memory_decision)
        ood_score = max(ood_score, open_set["open_set_score"])

        adjustments = meta_decision.get("weighted_profile", {}).get("adjustments", [])
        explanation_strength = clip01(max([item.get("strength", 0.0) for item in adjustments] or [0.0]))
        memory_strategy = str(memory_decision.get("strategy", ""))
        memory_uncertainty = 0.10 if memory_strategy in {"exact_image_memory"} else 0.0

        epistemic = clip01(
            0.46 * ood_score
            + 0.26 * disagreement
            + 0.14 * (1.0 - explanation_strength)
            + 0.14 * open_set["embedding_density_anomaly"]
        )
        aleatoric = clip01(0.65 * margin_uncertainty + 0.35 * conflict)
        overall = clip01(0.55 * epistemic + 0.35 * aleatoric - memory_uncertainty)
        reliability = clip01(1.0 - overall)
        conformal_set = _conformal_decision_set(fake_probability, overall, open_set["open_set_score"])

        return {
            "groups": {name: round(value, 6) for name, value in groups.items()},
            "disagreement": round(disagreement, 6),
            "margin_uncertainty": round(margin_uncertainty, 6),
            "ood_score": round(ood_score, 6),
            "mahalanobis_score": round(open_set["mahalanobis_score"], 6),
            "energy_score": round(open_set["energy_score"], 6),
            "embedding_density_anomaly": round(open_set["embedding_density_anomaly"], 6),
            "nearest_neighbor_anomaly": round(open_set["nearest_neighbor_anomaly"], 6),
            "open_set_score": round(open_set["open_set_score"], 6),
            "epistemic_uncertainty": round(epistemic, 6),
            "aleatoric_uncertainty": round(aleatoric, 6),
            "overall_uncertainty": round(overall, 6),
            "reliability": round(reliability, 6),
            "explanation_strength": round(explanation_strength, 6),
            "conformal_decision_set": conformal_set,
            "calibration": {
                "temperature_scaling": "configured",
                "dirichlet_calibration": "configured",
                "evidential_deep_learning": "supported",
                "monte_carlo_dropout": "supported_by_heavy_backends",
                "deep_ensembles": "active_via_stream_ensemble",
            },
            "calibration_warning": bool(overall > 0.58 and max(fake_probability, 1.0 - fake_probability) > 0.72),
        }


@dataclass
class HumanReviewPolicy:
    """Tiered policy for automatic decisions and review escalation."""

    ai_threshold: float = 0.65
    decisive_ai_threshold: float = 0.80
    real_threshold: float = 0.35
    decisive_real_threshold: float = 0.20
    review_uncertainty_threshold: float = 0.66
    ood_threshold: float = 0.72

    def decide(
        self,
        *,
        fake_probability: float,
        features: dict[str, float],
        uncertainty: dict[str, Any],
    ) -> dict[str, Any]:
        manipulation_score = features.get("forensic_manipulation", 0.0)
        ood_score = float(uncertainty.get("ood_score", 0.0))
        overall_uncertainty = float(uncertainty.get("overall_uncertainty", 0.0))
        reliability = float(uncertainty.get("reliability", 0.0))
        reasons: list[str] = []

        if fake_probability >= self.decisive_ai_threshold:
            return {
                "action": "AUTO_REJECT_AI",
                "decision_label": "AI_GENERATED",
                "confidence": round(fake_probability * 100.0, 2),
                "reasons": ["Decisive AI-generation evidence."],
                "requires_human_review": False,
            }

        if manipulation_score >= 0.72 and fake_probability >= 0.45:
            return {
                "action": "AUTO_REJECT_MANIPULATED",
                "decision_label": "MANIPULATED",
                "confidence": round(max(fake_probability, manipulation_score) * 100.0, 2),
                "reasons": ["Strong manipulation/blending evidence."],
                "requires_human_review": False,
            }

        if fake_probability <= self.decisive_real_threshold and reliability >= 0.42:
            return {
                "action": "AUTO_ACCEPT_REAL",
                "decision_label": "REAL",
                "confidence": round((1.0 - fake_probability) * 100.0, 2),
                "reasons": ["Low artifact evidence with acceptable reliability."],
                "requires_human_review": False,
            }

        if ood_score >= self.ood_threshold:
            reasons.append("Out-of-distribution evidence is high.")
        if overall_uncertainty >= self.review_uncertainty_threshold:
            reasons.append("Model uncertainty is high.")
        if 0.40 <= fake_probability <= 0.60:
            reasons.append("Decision margin is narrow.")

        if reasons:
            return {
                "action": "HUMAN_REVIEW",
                "decision_label": "HUMAN_REVIEW_REQUIRED",
                "confidence": round(max(fake_probability, 1.0 - fake_probability) * 100.0, 2),
                "reasons": reasons,
                "requires_human_review": True,
            }

        if fake_probability >= self.ai_threshold:
            return {
                "action": "AUTO_REJECT_AI",
                "decision_label": "AI_GENERATED",
                "confidence": round(fake_probability * 100.0, 2),
                "reasons": ["Multiple streams indicate AI generation."],
                "requires_human_review": False,
            }

        if fake_probability <= self.real_threshold:
            return {
                "action": "AUTO_ACCEPT_REAL",
                "decision_label": "REAL",
                "confidence": round((1.0 - fake_probability) * 100.0, 2),
                "reasons": ["Evidence leans authentic."],
                "requires_human_review": False,
            }

        return {
            "action": "HUMAN_REVIEW",
            "decision_label": "HUMAN_REVIEW_REQUIRED",
            "confidence": round(max(fake_probability, 1.0 - fake_probability) * 100.0, 2),
            "reasons": ["Mixed evidence requires human review."],
            "requires_human_review": True,
        }


def _evidence_groups(features: dict[str, float]) -> dict[str, float]:
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
    provenance = max(
        features.get("forensic_camera_provenance", 0.0),
        features.get("forensic_metadata_provenance", 0.0),
        features.get("forensic_provenance_consistency", 0.0),
        features.get("forensic_provenance_cfa", 0.0),
        features.get("forensic_provenance_exif", 0.0),
    )
    frequency = max(
        features.get("forensic_fft", 0.0),
        features.get("forensic_spectral_intelligence", 0.0),
        features.get("forensic_advanced_frequency", 0.0),
        features.get("forensic_radial_spectral", 0.0),
        features.get("forensic_periodicity", 0.0),
    )
    return {
        "pixel": features.get("pixel_ensemble", 0.0),
        "forensic": features.get("forensic_ensemble", 0.0),
        "semantic": semantic,
        "frequency": frequency,
        "provenance": provenance,
        "manipulation": features.get("forensic_manipulation", 0.0),
    }


def _open_set_scores(
    features: dict[str, float],
    groups: dict[str, float],
    fake_probability: float,
    memory_decision: dict[str, Any],
) -> dict[str, float]:
    vector = np.asarray([clip01(value) for value in features.values()], dtype=np.float64)
    if vector.size == 0:
        vector = np.asarray([0.5], dtype=np.float64)
    real_proto = np.full(vector.shape, 0.25, dtype=np.float64)
    fake_proto = np.full(vector.shape, 0.70, dtype=np.float64)
    real_distance = float(np.sqrt(np.mean((vector - real_proto) ** 2)))
    fake_distance = float(np.sqrt(np.mean((vector - fake_proto) ** 2)))
    mahalanobis_score = clip01((min(real_distance, fake_distance) - 0.18) / 0.34)

    group_values = np.asarray(list(groups.values()), dtype=np.float64)
    evidence_energy = float(np.log(np.sum(np.exp((group_values - 0.5) * 4.0)) + 1e-8))
    energy_score = clip01(1.0 - normalize_energy(evidence_energy))
    embedding_density_anomaly = clip01(float(np.std(vector) * 1.8 + (1.0 - max(np.mean(vector), 1.0 - np.mean(vector))) * 0.35))
    strategy = str(memory_decision.get("strategy", ""))
    nearest_neighbor_anomaly = 0.0 if strategy == "exact_image_memory" else 0.35
    if strategy in {"no_memory", "memory_no_close_match"}:
        nearest_neighbor_anomaly = 0.72
    open_set_score = clip01(
        0.34 * mahalanobis_score
        + 0.24 * energy_score
        + 0.24 * embedding_density_anomaly
        + 0.18 * nearest_neighbor_anomaly
        + 0.12 * (1.0 - abs(fake_probability - 0.5) * 2.0)
    )
    return {
        "mahalanobis_score": mahalanobis_score,
        "energy_score": energy_score,
        "embedding_density_anomaly": embedding_density_anomaly,
        "nearest_neighbor_anomaly": nearest_neighbor_anomaly,
        "open_set_score": open_set_score,
    }


def normalize_energy(value: float) -> float:
    return clip01((value - 0.55) / 1.95)


def _conformal_decision_set(
    fake_probability: float,
    uncertainty: float,
    open_set_score: float,
) -> list[str]:
    labels = []
    if fake_probability <= 0.42 or uncertainty >= 0.54:
        labels.append("REAL")
    if fake_probability >= 0.58 or uncertainty >= 0.54:
        labels.append("SYNTHETIC")
    if open_set_score >= 0.62 or uncertainty >= 0.66:
        labels.append("UNKNOWN")
    if uncertainty >= 0.62 or 0.40 <= fake_probability <= 0.60:
        labels.append("HUMAN_REVIEW_REQUIRED")
    return labels or (["SYNTHETIC"] if fake_probability >= 0.5 else ["REAL"])
