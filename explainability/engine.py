from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from utils.image_io import image_to_array, load_image
from utils.math_utils import clip01
from utils.types import PredictionResult


class ExplainabilityEngine:
    """Generates compact forensic evidence reports and visual maps."""

    def explain(self, image_input: Any, result: PredictionResult) -> dict[str, Any]:
        image = load_image(image_input)
        maps = self.heatmaps(image)
        top_features = sorted(
            (
                {"name": name, "value": float(value)}
                for name, value in result.features.items()
                if isinstance(value, (int, float))
            ),
            key=lambda item: item["value"],
            reverse=True,
        )[:12]
        uncertainty = result.details.get("uncertainty", {})
        evidence_fusion = result.details.get("evidence_fusion", {})
        return {
            "image_hash": result.image_hash,
            "decision": result.label,
            "confidence": round(result.confidence, 2),
            "fake_probability": round(result.fake_probability, 6),
            "top_evidence": [
                {"name": item["name"], "value": round(item["value"], 6)}
                for item in top_features
            ],
            "confidence_decomposition": {
                "fusion": evidence_fusion,
                "uncertainty": uncertainty,
                "review": result.details.get("review_decision", {}),
            },
            "semantic_explanations": _semantic_explanations(result.details.get("stream_details", {})),
            "forensic_explanations": _forensic_explanations(result.details.get("stream_details", {})),
            "maps": maps,
        }

    def heatmaps(self, image_input: Any) -> dict[str, str]:
        image = load_image(image_input)
        arr = image_to_array(image, size=(256, 256), grayscale=False)
        gray = np.mean(arr, axis=2)
        grad = _normalize(_gradient(gray))
        residual = _normalize(np.abs(gray - _box_blur(gray, 5)))
        frequency = _frequency_map(gray)
        semantic = _semantic_attention_map(arr, gray, grad)
        confidence = _normalize(0.42 * residual + 0.32 * frequency + 0.26 * semantic)
        return {
            "noise_inconsistency": _png_data_url(residual, palette="red"),
            "frequency_anomaly": _png_data_url(frequency, palette="blue"),
            "semantic_inconsistency": _png_data_url(semantic, palette="amber"),
            "forensic_saliency": _png_data_url(confidence, palette="magma"),
        }


def _semantic_explanations(stream_details: dict[str, Any]) -> list[dict[str, str]]:
    explanations = []
    for name, details in stream_details.items():
        if name.startswith("semantic_agent_") and details.get("reasoning"):
            explanations.append(
                {
                    "agent": details.get("agent", name.replace("semantic_agent_", "")),
                    "reasoning": details["reasoning"],
                }
            )
    return explanations[:10]


def _forensic_explanations(stream_details: dict[str, Any]) -> list[dict[str, str]]:
    explanations = []
    interesting_prefixes = (
        "forensic_diffusion",
        "forensic_provenance",
        "forensic_prnu",
        "forensic_advanced_frequency",
        "forensic_noise_advanced",
    )
    for name, details in stream_details.items():
        if name.startswith(interesting_prefixes):
            explanations.append(
                {
                    "probe": name,
                    "summary": details.get("interpretation")
                    or details.get("generalization_policy")
                    or details.get("frequency_anomaly_map")
                    or "Forensic probe contributed normalized evidence.",
                }
            )
    return explanations[:12]


def _gradient(gray: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    return gx + gy


def _box_blur(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(kernel_size):
        for x in range(kernel_size):
            out += padded[y : y + arr.shape[0], x : x + arr.shape[1]]
    return out / float(kernel_size * kernel_size)


def _frequency_map(gray: np.ndarray) -> np.ndarray:
    centered = gray - float(np.mean(gray))
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))
    low_removed = magnitude - _box_blur(magnitude, 9)
    spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(low_removed))).real
    return _normalize(spatial)


def _semantic_attention_map(arr: np.ndarray, gray: np.ndarray, grad: np.ndarray) -> np.ndarray:
    chroma = np.max(arr, axis=2) - np.min(arr, axis=2)
    center_prior = np.zeros_like(gray)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    center_prior = 1.0 - distance / max(float(distance.max()), 1e-8)
    text_like = ((gray < np.percentile(gray, 35)) & (grad > np.percentile(grad, 82))).astype(np.float32)
    return _normalize(0.38 * grad + 0.24 * chroma + 0.22 * center_prior + 0.16 * text_like)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if high <= low:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _png_data_url(values: np.ndarray, palette: str) -> str:
    rgb = _colorize(values, palette)
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    handle = BytesIO()
    image.save(handle, format="PNG")
    encoded = base64.b64encode(handle.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _colorize(values: np.ndarray, palette: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if palette == "red":
        return np.stack([values, values * 0.22, values * 0.08], axis=2)
    if palette == "blue":
        return np.stack([values * 0.08, values * 0.36, values], axis=2)
    if palette == "amber":
        return np.stack([values, values * 0.62, values * 0.10], axis=2)
    # Magma-like compact palette.
    return np.stack(
        [
            clip01_array(values * 1.28),
            clip01_array(np.sqrt(values) * 0.58),
            clip01_array((1.0 - values) * 0.24 + values * 0.10),
        ],
        axis=2,
    )


def clip01_array(values: np.ndarray) -> np.ndarray:
    return np.vectorize(clip01)(values)
