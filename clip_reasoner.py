from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, entropy01, normalize_minmax
from utils.types import StreamScore


REAL_PROMPTS = [
    "a natural photograph captured by a real camera",
    "an authentic unedited photo of a real scene",
    "a realistic camera image with natural imperfections",
]

FAKE_PROMPTS = [
    "an AI-generated image with synthetic artifacts",
    "a computer-generated fake image",
    "a diffusion model image with inconsistent details",
]


@dataclass
class _ClipBackend:
    model: Any
    processor: Any
    torch: Any


class CLIPSemanticAnalyzer:
    """Prompt-based semantic consistency analyzer.

    If Transformers/PyTorch/CLIP are available and `use_clip=True`, this class
    performs zero-shot prompt comparison. Otherwise it uses a deterministic
    semantic anomaly proxy so the semantic stream remains active.
    """

    anomaly_name = "semantic_anomaly"
    consistency_name = "semantic_consistency"

    def __init__(
        self,
        use_clip: bool = False,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        self.use_clip = use_clip
        self.model_name = model_name
        self.backend: _ClipBackend | None = None
        if use_clip:
            self.backend = self._try_load_clip(model_name)

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        if self.backend is not None:
            return self._analyze_with_clip(image)
        return self._analyze_with_fallback(image)

    @staticmethod
    def _try_load_clip(model_name: str) -> _ClipBackend | None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model.eval()
            return _ClipBackend(model=model, processor=processor, torch=torch)
        except Exception:
            return None

    def _analyze_with_clip(self, image: Image.Image) -> dict[str, StreamScore]:
        assert self.backend is not None
        prompts = REAL_PROMPTS + FAKE_PROMPTS
        inputs = self.backend.processor(
            text=prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        with self.backend.torch.no_grad():
            outputs = self.backend.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().numpy()

        real_prob = float(np.sum(probs[: len(REAL_PROMPTS)]))
        fake_prob = float(np.sum(probs[len(REAL_PROMPTS) :]))
        anomaly = clip01(fake_prob / (real_prob + fake_prob + 1e-8))
        consistency = clip01(1.0 - anomaly)
        details = {
            "backend": "clip",
            "model_name": self.model_name,
            "real_prompts": REAL_PROMPTS,
            "fake_prompts": FAKE_PROMPTS,
            "real_prompt_mass": real_prob,
            "fake_prompt_mass": fake_prob,
        }
        return {
            self.anomaly_name: StreamScore(self.anomaly_name, anomaly, details),
            self.consistency_name: StreamScore(
                self.consistency_name, consistency, details
            ),
        }

    def _analyze_with_fallback(self, image: Image.Image) -> dict[str, StreamScore]:
        original_size = image.size
        arr = image_to_array(image, size=(224, 224), grayscale=False)
        gray = np.mean(arr, axis=2)
        gradients = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        texture = normalize_minmax(float(gradients), 0.015, 0.11)
        entropy = entropy01(gray, bins=64)
        chroma = np.max(arr, axis=2) - np.min(arr, axis=2)
        saturation = float(np.mean(chroma))
        channel_balance = float(np.std(np.mean(arr, axis=(0, 1))))
        mirror_similarity = 1.0 - float(np.mean(np.abs(arr - np.flip(arr, axis=1))))
        portrait_synthetic = self._portrait_synthetic_proxy(arr, gray)
        stylized_composite = self._stylized_composite_proxy(arr)
        low_resolution_composite = self._low_resolution_composite_proxy(
            arr,
            gray,
            original_size,
        )

        over_smooth = clip01(1.0 - texture)
        low_entropy = clip01(1.0 - entropy)
        over_saturated = normalize_minmax(saturation, 0.32, 0.68)
        unusual_balance = normalize_minmax(channel_balance, 0.02, 0.18)
        excessive_symmetry = normalize_minmax(mirror_similarity, 0.78, 0.96)

        base_anomaly = clip01(
            0.28 * over_smooth
            + 0.22 * low_entropy
            + 0.20 * over_saturated
            + 0.15 * unusual_balance
            + 0.15 * excessive_symmetry
        )
        portrait_strength = normalize_minmax(portrait_synthetic["score"], 0.58, 0.90)
        portrait_anomaly = clip01(0.55 + 0.35 * portrait_strength)
        composite_strength = normalize_minmax(stylized_composite["score"], 0.62, 0.92)
        composite_anomaly = clip01(0.58 + 0.34 * composite_strength)
        low_resolution_strength = normalize_minmax(
            low_resolution_composite["score"],
            0.56,
            0.78,
        )
        low_resolution_anomaly = clip01(0.58 + 0.32 * low_resolution_strength)
        anomaly = max(
            base_anomaly,
            portrait_anomaly if portrait_strength > 0 else 0.0,
            composite_anomaly if composite_strength > 0 else 0.0,
            low_resolution_anomaly if low_resolution_strength > 0 else 0.0,
        )
        consistency = clip01(1.0 - anomaly)
        details = {
            "backend": "fallback_semantic_proxy",
            "prompt_intent": {
                "real": REAL_PROMPTS,
                "fake": FAKE_PROMPTS,
            },
            "texture": texture,
            "entropy": entropy,
            "saturation": saturation,
            "channel_balance": channel_balance,
            "mirror_similarity": mirror_similarity,
            "base_anomaly": base_anomaly,
            "portrait_synthetic": portrait_synthetic,
            "stylized_composite": stylized_composite,
            "low_resolution_composite": low_resolution_composite,
        }
        return {
            self.anomaly_name: StreamScore(self.anomaly_name, anomaly, details),
            self.consistency_name: StreamScore(
                self.consistency_name, consistency, details
            ),
        }

    @staticmethod
    def _portrait_synthetic_proxy(arr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        maxc = np.max(arr, axis=2)
        minc = np.min(arr, axis=2)
        skin_mask = (
            (r > 0.30)
            & (g > 0.20)
            & (b > 0.13)
            & ((maxc - minc) > 0.05)
            & (r > g * 0.92)
            & (r > b * 1.05)
            & (g > b * 0.85)
        )

        center_mask = np.zeros_like(skin_mask, dtype=bool)
        center_mask[40:184, 50:174] = True
        total_skin = float(np.mean(skin_mask))
        central_skin = float(np.mean(skin_mask & center_mask)) / max(
            float(np.mean(center_mask)),
            1e-8,
        )

        central_crop = arr[30:194, 40:184]
        central_symmetry = 1.0 - float(
            np.mean(np.abs(central_crop - np.flip(central_crop, axis=1)))
        )

        grad_y = np.abs(np.diff(gray, axis=0))
        grad_x = np.abs(np.diff(gray, axis=1))
        grad = np.zeros_like(gray)
        grad[:-1, :] += grad_y
        grad[:, :-1] += grad_x
        skin_gradient = float(np.mean(grad[skin_mask])) if np.any(skin_mask) else 1.0
        skin_smoothness = clip01(1.0 - normalize_minmax(skin_gradient, 0.015, 0.11))

        background = ~skin_mask
        background_entropy = entropy01(gray[background], bins=64) if np.any(background) else 0.0
        low_background_entropy = clip01(1.0 - background_entropy)

        score = clip01(
            0.25 * normalize_minmax(central_skin, 0.18, 0.48)
            + 0.20 * normalize_minmax(total_skin, 0.10, 0.35)
            + 0.25 * normalize_minmax(central_symmetry, 0.70, 0.88)
            + 0.20 * skin_smoothness
            + 0.10 * low_background_entropy
        )
        return {
            "score": score,
            "total_skin_ratio": total_skin,
            "central_skin_ratio": central_skin,
            "central_symmetry": central_symmetry,
            "skin_gradient": skin_gradient,
            "skin_smoothness": skin_smoothness,
            "background_entropy": background_entropy,
        }

    @staticmethod
    def _stylized_composite_proxy(arr: np.ndarray) -> dict[str, float]:
        hue, saturation, value = _rgb_to_hsv(arr)
        high_saturation = float(np.mean(saturation > 0.45))
        warm_dominance = float(
            np.mean(((hue < 0.13) | (hue > 0.92)) & (saturation > 0.25))
        )
        yellow_dominance = float(
            np.mean((hue > 0.08) & (hue < 0.18) & (saturation > 0.25))
        )
        red_dominance = float(
            np.mean(((hue < 0.06) | (hue > 0.94)) & (saturation > 0.35))
        )
        glow_mask = (saturation > 0.45) & (value > 0.65) & ((hue < 0.10) | (hue > 0.92))
        column_fraction = np.mean(glow_mask, axis=0)
        max_glow_column = float(np.max(column_fraction))
        glow_stripe_fraction = float(np.mean(column_fraction > 0.20))

        score = clip01(
            0.28 * normalize_minmax(high_saturation, 0.55, 0.88)
            + 0.24 * normalize_minmax(warm_dominance, 0.55, 0.85)
            + 0.18 * normalize_minmax(yellow_dominance, 0.25, 0.55)
            + 0.12 * normalize_minmax(red_dominance, 0.08, 0.35)
            + 0.10 * normalize_minmax(max_glow_column, 0.15, 0.35)
            + 0.08 * normalize_minmax(glow_stripe_fraction, 0.06, 0.18)
        )
        return {
            "score": score,
            "high_saturation_ratio": high_saturation,
            "warm_dominance": warm_dominance,
            "yellow_dominance": yellow_dominance,
            "red_dominance": red_dominance,
            "max_glow_column": max_glow_column,
            "glow_stripe_fraction": glow_stripe_fraction,
        }

    @staticmethod
    def _low_resolution_composite_proxy(
        arr: np.ndarray,
        gray: np.ndarray,
        original_size: tuple[int, int],
    ) -> dict[str, float]:
        width, height = original_size
        min_dimension = min(width, height)

        hue, saturation, value = _rgb_to_hsv(arr)
        dark_mask = value < 0.10
        column_darkness = np.mean(dark_mask, axis=0)
        row_darkness = np.mean(dark_mask, axis=1)
        border_columns = float(np.mean(column_darkness > 0.75))
        border_rows = float(np.mean(row_darkness > 0.75))

        edge_band = np.zeros_like(dark_mask, dtype=bool)
        edge_band[:, :18] = True
        edge_band[:, -18:] = True
        edge_band[:18, :] = True
        edge_band[-18:, :] = True
        edge_darkness = float(np.mean(dark_mask & edge_band)) / max(
            float(np.mean(edge_band)),
            1e-8,
        )
        center_darkness = float(np.mean(dark_mask[40:184, 40:184]))

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        maxc = np.max(arr, axis=2)
        minc = np.min(arr, axis=2)
        skin_mask = (
            (r > 0.18)
            & (g > 0.12)
            & (b > 0.08)
            & (r > g * 0.82)
            & (r > b * 1.02)
            & (g > b * 0.72)
            & ((maxc - minc) > 0.035)
        ) | (
            (r > 0.24)
            & (g > 0.16)
            & (b > 0.11)
            & (r >= g * 0.74)
            & (r > b * 0.95)
            & (g > b * 0.75)
            & ((maxc - minc) > 0.025)
        )

        center_mask = np.zeros_like(skin_mask, dtype=bool)
        center_mask[35:189, 35:189] = True
        lower_mask = np.zeros_like(skin_mask, dtype=bool)
        lower_mask[110:224, 30:194] = True
        total_skin = float(np.mean(skin_mask))
        central_skin = float(np.mean(skin_mask & center_mask)) / max(
            float(np.mean(center_mask)),
            1e-8,
        )
        lower_skin = float(np.mean(skin_mask & lower_mask)) / max(
            float(np.mean(lower_mask)),
            1e-8,
        )

        grad_y = np.abs(np.diff(gray, axis=0))
        grad_x = np.abs(np.diff(gray, axis=1))
        grid_v = float(np.mean(grad_x[:, 7::8])) if grad_x.shape[1] > 8 else 0.0
        grid_h = float(np.mean(grad_y[7::8, :])) if grad_y.shape[0] > 8 else 0.0
        base_gradient = float((np.mean(grad_x) + np.mean(grad_y)) / 2.0 + 1e-8)
        blockiness = float(((grid_v + grid_h) / 2.0) / base_gradient)

        small_image = clip01((420.0 - float(min_dimension)) / 220.0)
        border_artifact = clip01(border_columns + border_rows + edge_darkness - center_darkness)
        skin_mass = normalize_minmax(max(total_skin, central_skin, lower_skin), 0.18, 0.46)
        compression_artifact = normalize_minmax(blockiness, 1.00, 1.35)
        saturation_level = normalize_minmax(float(np.mean(saturation)), 0.25, 0.55)

        score = clip01(
            0.32 * small_image
            + 0.24 * border_artifact
            + 0.18 * skin_mass
            + 0.16 * compression_artifact
            + 0.10 * saturation_level
        )
        return {
            "score": score,
            "min_dimension": float(min_dimension),
            "small_image": small_image,
            "border_artifact": border_artifact,
            "border_columns": border_columns,
            "border_rows": border_rows,
            "edge_darkness": edge_darkness,
            "center_darkness": center_darkness,
            "skin_mass": skin_mass,
            "blockiness": blockiness,
            "compression_artifact": compression_artifact,
            "saturation_level": saturation_level,
        }


def _rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    value = maxc
    saturation = (maxc - minc) / (maxc + 1e-8)
    hue = np.zeros_like(maxc)
    mask = maxc != minc
    rc = (maxc - r) / (maxc - minc + 1e-8)
    gc = (maxc - g) / (maxc - minc + 1e-8)
    bc = (maxc - b) / (maxc - minc + 1e-8)
    hue[(r == maxc) & mask] = (bc - gc)[(r == maxc) & mask]
    hue[(g == maxc) & mask] = 2.0 + (rc - bc)[(g == maxc) & mask]
    hue[(b == maxc) & mask] = 4.0 + (gc - rc)[(b == maxc) & mask]
    hue = (hue / 6.0) % 1.0
    return hue, saturation, value
