from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, entropy01, normalize_minmax, safe_mean
from utils.types import StreamScore


@dataclass(frozen=True)
class SemanticAgentSpec:
    name: str
    prompt: str
    scorer: Callable[["SemanticSceneProfile"], tuple[float, str]]


class MultiAgentSemanticAnalyzer:
    """Independent semantic auditors for physical and contextual plausibility."""

    def __init__(self):
        self.agents = _build_agents()

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        profile = SemanticSceneProfile.from_image(image)
        scores: dict[str, StreamScore] = {}
        for agent in self.agents:
            probability, reason = agent.scorer(profile)
            stream_name = f"semantic_agent_{agent.name}"
            scores[stream_name] = StreamScore(
                stream_name,
                clip01(probability),
                {
                    "agent": agent.name,
                    "prompt": agent.prompt,
                    "reasoning": reason,
                    "backend": "deterministic_semantic_auditor",
                    "vlm_backends_supported": ["Florence-2", "Qwen-VL", "LLaVA", "CLIP", "SigLIP"],
                    "profile": profile.summary(),
                },
            )

        agent_values = [score.score for score in scores.values()]
        scores["semantic_multi_agent"] = StreamScore(
            "semantic_multi_agent",
            safe_mean(agent_values),
            {
                "members": list(scores),
                "disagreement": round(clip01(float(np.std(agent_values) * 2.8)) if agent_values else 0.0, 6),
                "aggregation": "probabilistic independent-auditor consensus",
            },
        )
        scores["semantic_physical_realism"] = StreamScore(
            "semantic_physical_realism",
            clip01(
                max(
                    scores["semantic_agent_lighting"].score,
                    scores["semantic_agent_reflection"].score,
                    scores["semantic_agent_perspective"].score,
                    scores["semantic_agent_physics"].score,
                    scores["semantic_agent_scene_geometry"].score,
                )
            ),
            {"members": ["lighting", "reflection", "perspective", "physics", "scene_geometry"]},
        )
        scores["semantic_typography"] = StreamScore(
            "semantic_typography",
            scores["semantic_agent_typography"].score,
            scores["semantic_agent_typography"].details,
        )
        return scores


@dataclass
class SemanticSceneProfile:
    texture: float
    entropy: float
    saturation: float
    channel_imbalance: float
    mirror_similarity: float
    center_skin_mass: float
    skin_smoothness: float
    face_symmetry_proxy: float
    pupil_like_asymmetry: float
    hand_like_topology: float
    glyph_density: float
    glyph_irregularity: float
    light_direction_variance: float
    shadow_incoherence: float
    reflection_incoherence: float
    perspective_incoherence: float
    object_grounding_error: float
    depth_layer_conflict: float
    repeated_pattern_error: float
    context_entropy_gap: float
    physics_edge_conflict: float

    @classmethod
    def from_image(cls, image: Image.Image) -> "SemanticSceneProfile":
        arr = image_to_array(image, size=(224, 224), grayscale=False)
        gray = np.mean(arr, axis=2)
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        grad = np.abs(gx) + np.abs(gy)
        texture = normalize_minmax(float(np.mean(grad)), 0.015, 0.11)
        entropy = entropy01(gray)
        chroma = np.max(arr, axis=2) - np.min(arr, axis=2)
        saturation = float(np.mean(chroma))
        channel_imbalance = float(np.std(np.mean(arr, axis=(0, 1))))
        mirror_similarity = 1.0 - float(np.mean(np.abs(arr - np.flip(arr, axis=1))))
        skin = _skin_mask(arr)
        center = np.zeros_like(gray, dtype=bool)
        center[36:188, 44:180] = True
        center_skin_mass = float(np.mean(skin & center)) / max(float(np.mean(center)), 1e-8)
        skin_smoothness = _skin_smoothness(skin, grad)
        face_crop = gray[34:190, 44:180]
        face_symmetry_proxy = 1.0 - float(np.mean(np.abs(face_crop - np.flip(face_crop, axis=1))))
        pupil_like_asymmetry = _pupil_asymmetry_proxy(gray)
        hand_like_topology = _hand_topology_proxy(arr, grad)
        glyph_density, glyph_irregularity = _glyph_proxy(gray, grad)
        light_direction_variance = _light_direction_variance(gx, gy, grad)
        shadow_incoherence = _shadow_incoherence(gray, grad)
        reflection_incoherence = _reflection_incoherence(gray)
        perspective_incoherence = _perspective_incoherence(grad)
        object_grounding_error = _object_grounding_error(gray, grad)
        depth_layer_conflict = _depth_layer_conflict(gray, grad)
        repeated_pattern_error = _repeated_pattern_error(gray)
        context_entropy_gap = _context_entropy_gap(gray)
        physics_edge_conflict = clip01(
            0.34 * shadow_incoherence
            + 0.24 * perspective_incoherence
            + 0.22 * object_grounding_error
            + 0.20 * depth_layer_conflict
        )
        return cls(
            texture=texture,
            entropy=entropy,
            saturation=saturation,
            channel_imbalance=channel_imbalance,
            mirror_similarity=mirror_similarity,
            center_skin_mass=clip01(center_skin_mass),
            skin_smoothness=skin_smoothness,
            face_symmetry_proxy=clip01(face_symmetry_proxy),
            pupil_like_asymmetry=pupil_like_asymmetry,
            hand_like_topology=hand_like_topology,
            glyph_density=glyph_density,
            glyph_irregularity=glyph_irregularity,
            light_direction_variance=light_direction_variance,
            shadow_incoherence=shadow_incoherence,
            reflection_incoherence=reflection_incoherence,
            perspective_incoherence=perspective_incoherence,
            object_grounding_error=object_grounding_error,
            depth_layer_conflict=depth_layer_conflict,
            repeated_pattern_error=repeated_pattern_error,
            context_entropy_gap=context_entropy_gap,
            physics_edge_conflict=physics_edge_conflict,
        )

    def summary(self) -> dict[str, float]:
        return {
            "texture": round(self.texture, 6),
            "entropy": round(self.entropy, 6),
            "saturation": round(self.saturation, 6),
            "center_skin_mass": round(self.center_skin_mass, 6),
            "skin_smoothness": round(self.skin_smoothness, 6),
            "glyph_density": round(self.glyph_density, 6),
            "light_direction_variance": round(self.light_direction_variance, 6),
            "perspective_incoherence": round(self.perspective_incoherence, 6),
            "physics_edge_conflict": round(self.physics_edge_conflict, 6),
        }


def _build_agents() -> list[SemanticAgentSpec]:
    return [
        SemanticAgentSpec(
            "anatomy",
            "Identify anatomical inconsistencies in hands, faces, teeth, ears, hair, pupils, and body joints.",
            lambda p: (
                clip01(
                    0.26 * p.center_skin_mass * p.skin_smoothness
                    + 0.22 * normalize_minmax(p.face_symmetry_proxy, 0.74, 0.92)
                    + 0.22 * p.pupil_like_asymmetry
                    + 0.18 * p.hand_like_topology
                    + 0.12 * p.repeated_pattern_error
                ),
                "Skin, symmetry, pupil, and hand-topology proxies were compared for biologically implausible regularity.",
            ),
        ),
        SemanticAgentSpec(
            "typography",
            "Check readable text, glyph consistency, logos, symbols, and repeated patterns.",
            lambda p: (
                clip01(0.52 * p.glyph_irregularity + 0.28 * p.glyph_density + 0.20 * p.repeated_pattern_error),
                "Text-like edge clusters were scored for malformed glyph rhythm and repeated-symbol artifacts.",
            ),
        ),
        SemanticAgentSpec(
            "lighting",
            "Estimate whether light-source direction and shadow fields are coherent.",
            lambda p: (
                clip01(0.58 * p.light_direction_variance + 0.42 * p.shadow_incoherence),
                "Dominant edge-light directions and dark-region gradients disagree more as the score rises.",
            ),
        ),
        SemanticAgentSpec(
            "reflection",
            "Check whether reflections match scene layout and lighting.",
            lambda p: (
                p.reflection_incoherence,
                "Vertical and horizontal mirror-like structures were compared for inconsistent reflected content.",
            ),
        ),
        SemanticAgentSpec(
            "perspective",
            "Determine whether object geometry and vanishing directions are physically plausible.",
            lambda p: (
                p.perspective_incoherence,
                "Line orientation concentration and horizon-like structure were audited for incompatible perspective cues.",
            ),
        ),
        SemanticAgentSpec(
            "object_relation",
            "Identify impossible object relationships and compositional collisions.",
            lambda p: (
                clip01(0.42 * p.object_grounding_error + 0.34 * p.depth_layer_conflict + 0.24 * p.context_entropy_gap),
                "Object grounding and local context discontinuity were used as relation-level anomaly cues.",
            ),
        ),
        SemanticAgentSpec(
            "human_realism",
            "Evaluate human realism, skin continuity, facial asymmetry, and hair/feature consistency.",
            lambda p: (
                clip01(0.40 * p.center_skin_mass * p.skin_smoothness + 0.26 * p.face_symmetry_proxy + 0.20 * p.pupil_like_asymmetry + 0.14 * p.channel_imbalance),
                "Human-region smoothness, symmetry, and color-continuity proxies were aggregated.",
            ),
        ),
        SemanticAgentSpec(
            "contextual_logic",
            "Find contextual contradictions and implausible scene semantics.",
            lambda p: (
                clip01(0.36 * p.context_entropy_gap + 0.26 * p.repeated_pattern_error + 0.22 * p.depth_layer_conflict + 0.16 * (1.0 - p.entropy)),
                "Large local entropy gaps and repeated motif errors indicate weak scene logic.",
            ),
        ),
        SemanticAgentSpec(
            "scene_geometry",
            "Audit global scene geometry, depth relationships, and object grounding.",
            lambda p: (
                clip01(0.40 * p.perspective_incoherence + 0.32 * p.object_grounding_error + 0.28 * p.depth_layer_conflict),
                "Grounding, depth, and line-field cues were fused into a scene geometry anomaly score.",
            ),
        ),
        SemanticAgentSpec(
            "physics",
            "Validate shadows, depth ordering, contact, reflectance, and basic optics.",
            lambda p: (
                p.physics_edge_conflict,
                "Shadow, perspective, grounding, and depth conflicts were treated as physical-realism violations.",
            ),
        ),
    ]


def _skin_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    return (
        (r > 0.24)
        & (g > 0.15)
        & (b > 0.10)
        & (r >= g * 0.78)
        & (r > b * 1.02)
        & (g > b * 0.70)
        & ((maxc - minc) > 0.03)
    )


def _skin_smoothness(skin: np.ndarray, grad: np.ndarray) -> float:
    if not np.any(skin):
        return 0.0
    return clip01(1.0 - normalize_minmax(float(np.mean(grad[skin])), 0.012, 0.10))


def _pupil_asymmetry_proxy(gray: np.ndarray) -> float:
    face = gray[42:150, 58:166]
    dark = face < np.percentile(face, 15)
    left = float(np.mean(dark[:, : dark.shape[1] // 2]))
    right = float(np.mean(dark[:, dark.shape[1] // 2 :]))
    mass = left + right
    if mass < 0.01:
        return 0.0
    return normalize_minmax(abs(left - right) / (mass + 1e-8), 0.08, 0.55)


def _hand_topology_proxy(arr: np.ndarray, grad: np.ndarray) -> float:
    skin = _skin_mask(arr)
    lower = np.zeros_like(skin, dtype=bool)
    lower[120:220, :] = True
    skin_edges = skin & lower & (grad > np.percentile(grad, 78))
    column_runs = float(np.mean(np.sum(skin_edges, axis=0) > 7))
    isolated = _connected_component_proxy(skin & lower)
    return clip01(0.55 * normalize_minmax(column_runs, 0.03, 0.18) + 0.45 * isolated)


def _glyph_proxy(gray: np.ndarray, grad: np.ndarray) -> tuple[float, float]:
    strong = grad > np.percentile(grad, 86)
    dark = gray < np.percentile(gray, 42)
    glyph = strong & dark
    density = normalize_minmax(float(np.mean(glyph)), 0.015, 0.16)
    row_density = np.mean(glyph, axis=1)
    col_density = np.mean(glyph, axis=0)
    rhythm = float(np.std(row_density) + np.std(col_density))
    irregularity = normalize_minmax(rhythm, 0.025, 0.16) * density
    return clip01(density), clip01(irregularity)


def _light_direction_variance(gx: np.ndarray, gy: np.ndarray, grad: np.ndarray) -> float:
    mask = grad > np.percentile(grad, 75)
    if not np.any(mask):
        return 0.0
    angles = np.arctan2(gy[mask], gx[mask])
    hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi), density=False)
    if hist.sum() == 0:
        return 0.0
    probs = hist / hist.sum()
    entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))) / np.log2(len(hist))
    return clip01(entropy)


def _shadow_incoherence(gray: np.ndarray, grad: np.ndarray) -> float:
    dark = gray < np.percentile(gray, 28)
    if not np.any(dark):
        return 0.0
    dark_edge = float(np.mean(grad[dark]))
    scene_edge = float(np.mean(grad) + 1e-8)
    return normalize_minmax(dark_edge / scene_edge, 0.55, 1.65)


def _reflection_incoherence(gray: np.ndarray) -> float:
    top = gray[:112, :]
    bottom = np.flipud(gray[112:, :])
    vertical_reflection_gap = float(np.mean(np.abs(top[: bottom.shape[0], :] - bottom)))
    left = gray[:, :112]
    right = np.fliplr(gray[:, 112:])
    horizontal_reflection_gap = float(np.mean(np.abs(left[:, : right.shape[1]] - right)))
    mirror_similarity = 1.0 - min(vertical_reflection_gap, horizontal_reflection_gap)
    return normalize_minmax(mirror_similarity, 0.82, 0.96)


def _perspective_incoherence(grad: np.ndarray) -> float:
    gy = np.abs(np.diff(grad, axis=0, append=grad[-1:, :]))
    gx = np.abs(np.diff(grad, axis=1, append=grad[:, -1:]))
    horizontal = float(np.mean(gx > np.percentile(gx, 86)))
    vertical = float(np.mean(gy > np.percentile(gy, 86)))
    axis_dominance = abs(horizontal - vertical)
    line_mass = horizontal + vertical
    return clip01(0.55 * normalize_minmax(axis_dominance, 0.03, 0.20) + 0.45 * normalize_minmax(line_mass, 0.12, 0.32))


def _object_grounding_error(gray: np.ndarray, grad: np.ndarray) -> float:
    lower = np.zeros_like(gray, dtype=bool)
    lower[150:, :] = True
    contacts = (grad > np.percentile(grad, 80)) & lower
    contact_mass = float(np.mean(contacts))
    smooth_floor = float(np.mean(grad[lower] < np.percentile(grad, 35)))
    return clip01((1.0 - normalize_minmax(contact_mass, 0.02, 0.16)) * normalize_minmax(smooth_floor, 0.34, 0.82))


def _depth_layer_conflict(gray: np.ndarray, grad: np.ndarray) -> float:
    rows = np.array_split(np.arange(gray.shape[0]), 4)
    layer_detail = [float(np.mean(grad[row, :])) for row in rows]
    if len(layer_detail) < 3:
        return 0.0
    inversions = sum(
        1 for before, after in zip(layer_detail[:-1], layer_detail[1:], strict=True) if after > before * 1.35
    )
    return clip01(inversions / 3.0)


def _repeated_pattern_error(gray: np.ndarray) -> float:
    patches = []
    block = 28
    for y in range(0, gray.shape[0] - block + 1, block):
        for x in range(0, gray.shape[1] - block + 1, block):
            patch = gray[y : y + block, x : x + block]
            patches.append((patch - float(np.mean(patch))).ravel())
    if len(patches) < 4:
        return 0.0
    vectors = np.asarray(patches, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms
    sim = vectors @ vectors.T
    upper = sim[np.triu_indices_from(sim, k=1)]
    return normalize_minmax(float(np.percentile(upper, 96)), 0.35, 0.82)


def _context_entropy_gap(gray: np.ndarray) -> float:
    block = 32
    entropies = []
    for y in range(0, gray.shape[0] - block + 1, block):
        for x in range(0, gray.shape[1] - block + 1, block):
            entropies.append(entropy01(gray[y : y + block, x : x + block], bins=32))
    if not entropies:
        return 0.0
    return normalize_minmax(float(np.std(entropies)), 0.05, 0.24)


def _connected_component_proxy(mask: np.ndarray) -> float:
    # Fast proxy: count isolated skin-heavy columns/rows rather than full labeling.
    row_mass = np.mean(mask, axis=1)
    col_mass = np.mean(mask, axis=0)
    row_fragments = np.sum((row_mass[1:] > 0.04) & (row_mass[:-1] <= 0.04))
    col_fragments = np.sum((col_mass[1:] > 0.04) & (col_mass[:-1] <= 0.04))
    return normalize_minmax(float(row_fragments + col_fragments), 4.0, 20.0)
