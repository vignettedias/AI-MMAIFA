from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.math_utils import clip01


@dataclass
class EvidenceFusionResult:
    probability: float
    details: dict[str, Any]


class EvidenceFusionEngine:
    """Cross-modal, uncertainty-aware fusion layer.

    This layer complements the XGBoost/weighted meta-classifier by estimating
    stream reliability and conflict before the final probability reaches memory
    and review policy.
    """

    def fuse(
        self,
        *,
        features: dict[str, float],
        prior_probability: float,
    ) -> EvidenceFusionResult:
        groups = _group_scores(features)
        reliability = _stream_reliability(features, groups)
        weights = _dynamic_weights(groups, reliability)
        weighted_evidence = sum(groups[name] * weights[name] for name in groups)
        values = np.asarray(list(groups.values()), dtype=np.float64)
        disagreement = clip01(float(np.std(values) * 2.6))
        conflict_pairs = _conflict_pairs(groups)

        mixture_probability = clip01(0.58 * prior_probability + 0.42 * weighted_evidence)
        collapsed_probability = 0.5 + (mixture_probability - 0.5) * (1.0 - 0.34 * disagreement)
        if conflict_pairs:
            collapsed_probability = 0.5 + (collapsed_probability - 0.5) * 0.86

        return EvidenceFusionResult(
            probability=clip01(collapsed_probability),
            details={
                "prior_probability": round(prior_probability, 6),
                "weighted_evidence_probability": round(weighted_evidence, 6),
                "final_probability": round(clip01(collapsed_probability), 6),
                "groups": {name: round(value, 6) for name, value in groups.items()},
                "stream_reliability": {name: round(value, 6) for name, value in reliability.items()},
                "dynamic_weights": {name: round(value, 6) for name, value in weights.items()},
                "disagreement": round(disagreement, 6),
                "conflicts": conflict_pairs,
                "mixture_of_experts": _routing_decision(groups, reliability),
                "policy": "cross-modal transformer style evidence routing with confidence collapse on conflict",
            },
        )


def _group_scores(features: dict[str, float]) -> dict[str, float]:
    semantic_values = [
        features.get("semantic_anomaly", 0.0),
        features.get("semantic_multi_agent", 0.0),
        features.get("semantic_physical_realism", 0.0),
        features.get("semantic_typography", 0.0),
        features.get("semantic_portrait_synthetic", 0.0),
        features.get("semantic_stylized_composite", 0.0),
        features.get("semantic_low_resolution_composite", 0.0),
        features.get("semantic_nature_render", 0.0),
        features.get("semantic_architecture_render", 0.0),
    ]
    diffusion_values = [
        features.get("expert_diffusion_artifact", 0.0),
        features.get("forensic_diffusion_trace", 0.0),
        features.get("forensic_diffusion_denoising", 0.0),
        features.get("forensic_diffusion_spectral_flattening", 0.0),
    ]
    provenance_values = [
        features.get("forensic_camera_provenance", 0.0),
        features.get("forensic_metadata_provenance", 0.0),
        features.get("forensic_provenance_consistency", 0.0),
        features.get("forensic_provenance_cfa", 0.0),
        features.get("forensic_provenance_exif", 0.0),
    ]
    return {
        "pixel": features.get("pixel_ensemble", 0.0),
        "forensic": features.get("forensic_ensemble", 0.0),
        "semantic": max(semantic_values),
        "diffusion": max(diffusion_values),
        "provenance": max(provenance_values),
        "manipulation": max(
            features.get("forensic_manipulation", 0.0),
            features.get("expert_faceswap", 0.0),
            features.get("expert_inpainting", 0.0),
        ),
        "distribution": max(
            features.get("expert_social_recompression", 0.0),
            features.get("expert_screenshot_artifact", 0.0),
            features.get("forensic_compression", 0.0),
        ),
    }


def _stream_reliability(features: dict[str, float], groups: dict[str, float]) -> dict[str, float]:
    pixel_disagreement = features.get("pixel_expert_disagreement", 0.0)
    semantic_disagreement = _prefix_std(features, "semantic_agent_")
    forensic_disagreement = _prefix_std(features, "forensic_")
    return {
        "pixel": clip01(1.0 - 0.70 * pixel_disagreement),
        "forensic": clip01(1.0 - 0.42 * forensic_disagreement),
        "semantic": clip01(1.0 - 0.58 * semantic_disagreement),
        "diffusion": clip01(0.72 + 0.28 * groups["diffusion"]),
        "provenance": clip01(0.62 + 0.30 * groups["provenance"] - 0.12 * forensic_disagreement),
        "manipulation": clip01(0.66 + 0.30 * groups["manipulation"]),
        "distribution": clip01(0.64 + 0.24 * groups["distribution"]),
    }


def _dynamic_weights(groups: dict[str, float], reliability: dict[str, float]) -> dict[str, float]:
    base = {
        "pixel": 0.20,
        "forensic": 0.20,
        "semantic": 0.17,
        "diffusion": 0.14,
        "provenance": 0.12,
        "manipulation": 0.10,
        "distribution": 0.07,
    }
    raw = {
        name: base[name] * (0.45 + reliability[name]) * (0.82 + 0.36 * groups[name])
        for name in groups
    }
    total = sum(raw.values()) or 1.0
    return {name: value / total for name, value in raw.items()}


def _conflict_pairs(groups: dict[str, float]) -> list[dict[str, float | str]]:
    conflicts = []
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            gap = abs(groups[left] - groups[right])
            if gap >= 0.48 and max(groups[left], groups[right]) >= 0.65:
                conflicts.append({"streams": f"{left}/{right}", "gap": round(gap, 6)})
    return conflicts


def _routing_decision(groups: dict[str, float], reliability: dict[str, float]) -> dict[str, Any]:
    routed = sorted(
        (
            {
                "stream": name,
                "evidence": round(groups[name], 6),
                "reliability": round(reliability[name], 6),
                "route_score": round(groups[name] * reliability[name], 6),
            }
            for name in groups
        ),
        key=lambda item: item["route_score"],
        reverse=True,
    )
    return {"primary": routed[:3], "all": routed}


def _prefix_std(features: dict[str, float], prefix: str) -> float:
    values = [float(value) for name, value in features.items() if name.startswith(prefix)]
    if len(values) < 2:
        return 0.0
    return clip01(float(np.std(values) * 2.4))
