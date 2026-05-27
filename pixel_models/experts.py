from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from pixel_models.base import PIXEL_FEATURE_NAMES, extract_pixel_features
from utils.image_io import image_to_array
from utils.math_utils import clip01, entropy01, normalize_minmax, safe_mean
from utils.types import StreamScore


@dataclass(frozen=True)
class PixelExpertSpec:
    name: str
    family: str
    optional_backend: str
    scorer: Callable[["PixelSignalProfile"], float]


class ModernPixelExpertEnsemble:
    """Modern pixel and generator-specialist expert layer.

    The class is intentionally deployable without heavyweight model weights. In
    production, each expert name maps to a ConvNeXt/Swin/DINO/CLIP/SigLIP or
    specialist checkpoint. Locally, the same contract is fulfilled by calibrated
    forensic proxy features so the whole pipeline remains runnable and testable.
    """

    def __init__(self):
        self.specs = _build_specs()

    def analyze(self, image: Image.Image) -> dict[str, StreamScore]:
        profile = PixelSignalProfile.from_image(image)
        scores: dict[str, StreamScore] = {}
        for spec in self.specs:
            probability = clip01(spec.scorer(profile))
            scores[spec.name] = StreamScore(
                name=spec.name,
                score=probability,
                details={
                    "family": spec.family,
                    "optional_backend": spec.optional_backend,
                    "backend": "forensic_proxy",
                    "probability": round(probability, 6),
                    "uncertainty": round(_score_uncertainty(probability, profile), 6),
                    "embedding": profile.embedding(),
                    "saliency_map": profile.saliency_summary(),
                    "anomaly_regions": profile.anomaly_regions(),
                },
            )

        expert_values = [item.score for item in scores.values()]
        scores["pixel_modern_ensemble"] = StreamScore(
            "pixel_modern_ensemble",
            safe_mean(expert_values),
            {
                "members": [spec.name for spec in self.specs],
                "routing": "mixture_of_pixel_and_generator_experts",
            },
        )
        scores["pixel_expert_disagreement"] = StreamScore(
            "pixel_expert_disagreement",
            clip01(float(np.std(expert_values) * 2.8)) if expert_values else 0.0,
            {
                "members": [spec.name for spec in self.specs],
                "policy": "high disagreement reduces downstream confidence",
            },
        )
        return scores


@dataclass
class PixelSignalProfile:
    base_features: dict[str, float]
    texture_density: float
    smoothness: float
    entropy: float
    saturation: float
    channel_imbalance: float
    edge_uniformity: float
    blockiness: float
    high_frequency_energy: float
    mid_frequency_energy: float
    spectral_flatness: float
    spectral_entropy: float
    periodicity: float
    anisotropy: float
    patch_texture_cv: float
    patch_color_cv: float
    patch_anomaly: float
    camera_noise_coherence: float
    denoising_smoothness: float
    edge_oversampling: float
    upscale_signature: float
    skin_smoothness: float
    center_skin_mass: float
    text_edge_density: float
    flat_region_mass: float
    compression_grid: float
    low_noise_flatness: float
    top_patches: list[dict[str, float]]

    @classmethod
    def from_image(cls, image: Image.Image) -> "PixelSignalProfile":
        arr = image_to_array(image, size=(224, 224), grayscale=False)
        gray = np.mean(arr, axis=2)
        base = extract_pixel_features(image)
        base_features = dict(zip(PIXEL_FEATURE_NAMES, base.tolist(), strict=True))

        gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
        gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
        grad = gx + gy
        flat_mask = grad < np.percentile(grad, 35)
        edge_mask = grad > np.percentile(grad, 78)

        spectrum = np.fft.fftshift(np.fft.fft2(gray - float(np.mean(gray))))
        magnitude = np.abs(spectrum)
        log_magnitude = np.log1p(magnitude)
        h, w = log_magnitude.shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius_norm = radius / max(float(radius.max()), 1e-8)
        total = float(np.sum(log_magnitude) + 1e-8)
        high_frequency_energy = float(np.sum(log_magnitude[radius_norm > 0.56]) / total)
        mid_frequency_energy = float(
            np.sum(log_magnitude[(radius_norm > 0.22) & (radius_norm <= 0.56)]) / total
        )
        spectral_flatness = _spectral_flatness(magnitude)
        spectral_entropy = entropy01(log_magnitude / max(float(np.max(log_magnitude)), 1e-8))
        periodicity = _periodicity_score(log_magnitude, radius_norm)
        anisotropy = _anisotropy_score(log_magnitude)

        patch_texture, patch_color, patch_infos = _patch_statistics(arr, gray, grad)
        patch_texture_cv = _coeff_var(patch_texture)
        patch_color_cv = _coeff_var(patch_color)
        patch_anomaly = clip01(0.55 * patch_texture_cv + 0.45 * patch_color_cv)
        top_patches = sorted(patch_infos, key=lambda item: item["anomaly"], reverse=True)[:5]

        residual = gray - _box_blur(gray, 5)
        flat_noise = float(np.mean(np.abs(residual[flat_mask]))) if np.any(flat_mask) else 0.0
        edge_noise = float(np.mean(np.abs(residual[edge_mask]))) if np.any(edge_mask) else 0.0
        camera_noise_coherence = clip01(1.0 - normalize_minmax(_block_std_cv(residual), 0.18, 0.75))
        denoising_smoothness = clip01(1.0 - normalize_minmax(flat_noise, 0.004, 0.045))
        edge_oversampling = normalize_minmax(edge_noise / (flat_noise + 1e-8), 2.4, 7.5)
        low_noise_flatness = clip01(float(np.mean(flat_mask)) * denoising_smoothness)

        compression_grid = _grid_discontinuity(gray)
        upscale_signature = clip01(
            0.38 * edge_oversampling
            + 0.30 * denoising_smoothness
            + 0.18 * normalize_minmax(high_frequency_energy, 0.24, 0.48)
            + 0.14 * compression_grid
        )

        skin_smoothness, center_skin_mass = _skin_proxy(arr, gray, grad)
        text_edge_density = _text_edge_density(gray, grad)
        flat_region_mass = float(np.mean(flat_mask))

        return cls(
            base_features=base_features,
            texture_density=base_features["texture_density"],
            smoothness=base_features["smoothness"],
            entropy=base_features["entropy"],
            saturation=base_features["saturation"],
            channel_imbalance=base_features["channel_imbalance"],
            edge_uniformity=base_features["edge_uniformity"],
            blockiness=base_features["blockiness"],
            high_frequency_energy=high_frequency_energy,
            mid_frequency_energy=mid_frequency_energy,
            spectral_flatness=spectral_flatness,
            spectral_entropy=spectral_entropy,
            periodicity=periodicity,
            anisotropy=anisotropy,
            patch_texture_cv=patch_texture_cv,
            patch_color_cv=patch_color_cv,
            patch_anomaly=patch_anomaly,
            camera_noise_coherence=camera_noise_coherence,
            denoising_smoothness=denoising_smoothness,
            edge_oversampling=edge_oversampling,
            upscale_signature=upscale_signature,
            skin_smoothness=skin_smoothness,
            center_skin_mass=center_skin_mass,
            text_edge_density=text_edge_density,
            flat_region_mass=flat_region_mass,
            compression_grid=compression_grid,
            low_noise_flatness=low_noise_flatness,
            top_patches=top_patches,
        )

    def embedding(self) -> list[float]:
        values = [
            self.texture_density,
            self.smoothness,
            self.entropy,
            self.spectral_flatness,
            self.patch_anomaly,
            self.camera_noise_coherence,
            self.denoising_smoothness,
            self.edge_oversampling,
            self.upscale_signature,
            self.text_edge_density,
        ]
        return [round(float(value), 6) for value in values]

    def saliency_summary(self) -> dict[str, float]:
        return {
            "frequency": round(clip01(0.5 * self.spectral_flatness + 0.5 * self.periodicity), 6),
            "patches": round(self.patch_anomaly, 6),
            "edges": round(self.edge_oversampling, 6),
            "smooth_regions": round(self.denoising_smoothness, 6),
        }

    def anomaly_regions(self) -> list[dict[str, float]]:
        return self.top_patches


def _build_specs() -> list[PixelExpertSpec]:
    return [
        PixelExpertSpec(
            "pixel_convnext_v2",
            "foundation_pixel",
            "ConvNeXt-V2",
            lambda p: clip01(0.34 * p.smoothness + 0.26 * p.edge_uniformity + 0.22 * p.patch_anomaly + 0.18 * p.saturation),
        ),
        PixelExpertSpec(
            "pixel_swin_v2",
            "foundation_pixel",
            "Swin Transformer V2",
            lambda p: clip01(0.42 * p.patch_anomaly + 0.24 * p.periodicity + 0.20 * p.edge_uniformity + 0.14 * p.anisotropy),
        ),
        PixelExpertSpec(
            "pixel_dinov2",
            "foundation_pixel",
            "DINOv2",
            lambda p: clip01(0.32 * p.low_noise_flatness + 0.24 * p.patch_color_cv + 0.22 * p.saturation + 0.22 * (1.0 - p.entropy)),
        ),
        PixelExpertSpec(
            "pixel_clip_vit_l14",
            "vision_language_pixel",
            "CLIP ViT-L/14",
            lambda p: clip01(0.30 * p.center_skin_mass * p.skin_smoothness + 0.26 * p.saturation + 0.24 * p.smoothness + 0.20 * p.patch_anomaly),
        ),
        PixelExpertSpec(
            "pixel_siglip",
            "vision_language_pixel",
            "SigLIP",
            lambda p: clip01(0.34 * p.patch_color_cv + 0.26 * p.channel_imbalance + 0.22 * p.saturation + 0.18 * p.low_noise_flatness),
        ),
        PixelExpertSpec(
            "pixel_frequency_vit",
            "frequency_aware_pixel",
            "Frequency-aware ViT",
            lambda p: clip01(0.36 * p.spectral_flatness + 0.25 * p.periodicity + 0.22 * normalize_minmax(p.high_frequency_energy, 0.22, 0.50) + 0.17 * p.anisotropy),
        ),
        PixelExpertSpec(
            "pixel_cross_scale_transformer",
            "cross_scale_pixel",
            "Cross-scale Transformer",
            lambda p: clip01(0.34 * p.patch_texture_cv + 0.28 * p.edge_oversampling + 0.20 * p.low_noise_flatness + 0.18 * p.patch_anomaly),
        ),
        PixelExpertSpec(
            "pixel_patch_forensic_encoder",
            "patch_forensic_pixel",
            "Patch forensic encoder",
            lambda p: clip01(0.46 * p.patch_anomaly + 0.24 * p.compression_grid + 0.18 * p.edge_oversampling + 0.12 * (1.0 - p.camera_noise_coherence)),
        ),
        PixelExpertSpec(
            "expert_gan_artifact",
            "generator_specialist",
            "GAN artifact expert",
            lambda p: clip01(0.38 * p.periodicity + 0.24 * p.spectral_flatness + 0.20 * p.edge_uniformity + 0.18 * p.saturation),
        ),
        PixelExpertSpec(
            "expert_diffusion_artifact",
            "generator_specialist",
            "Diffusion artifact expert",
            lambda p: clip01(0.34 * p.denoising_smoothness + 0.28 * p.edge_oversampling + 0.22 * p.spectral_flatness + 0.16 * p.low_noise_flatness),
        ),
        PixelExpertSpec(
            "expert_faceswap",
            "manipulation_specialist",
            "Face-swap expert",
            lambda p: clip01(0.44 * p.center_skin_mass * p.skin_smoothness + 0.24 * p.patch_anomaly + 0.18 * p.edge_oversampling + 0.14 * (1.0 - p.camera_noise_coherence)),
        ),
        PixelExpertSpec(
            "expert_inpainting",
            "manipulation_specialist",
            "Inpainting expert",
            lambda p: clip01(0.50 * p.patch_anomaly + 0.24 * p.patch_texture_cv + 0.16 * p.denoising_smoothness + 0.10 * p.patch_color_cv),
        ),
        PixelExpertSpec(
            "expert_ai_upscaling",
            "restoration_specialist",
            "AI upscaling expert",
            lambda p: p.upscale_signature,
        ),
        PixelExpertSpec(
            "expert_cgi_realism",
            "render_specialist",
            "CGI realism expert",
            lambda p: clip01(0.34 * p.edge_uniformity + 0.28 * p.low_noise_flatness + 0.22 * p.saturation + 0.16 * (1.0 - p.camera_noise_coherence)),
        ),
        PixelExpertSpec(
            "expert_social_recompression",
            "distribution_specialist",
            "Social-media recompression expert",
            lambda p: clip01(0.50 * p.compression_grid + 0.22 * p.blockiness + 0.16 * p.periodicity + 0.12 * normalize_minmax(p.high_frequency_energy, 0.18, 0.44)),
        ),
        PixelExpertSpec(
            "expert_screenshot_artifact",
            "distribution_specialist",
            "Screenshot artifact expert",
            lambda p: clip01(0.36 * p.text_edge_density + 0.28 * p.flat_region_mass + 0.22 * p.edge_oversampling + 0.14 * p.compression_grid),
        ),
    ]


def _score_uncertainty(probability: float, profile: PixelSignalProfile) -> float:
    margin = 1.0 - abs(probability - 0.5) * 2.0
    evidence = max(profile.patch_anomaly, profile.spectral_flatness, profile.edge_oversampling)
    return clip01(0.72 * margin + 0.28 * (1.0 - evidence))


def _box_blur(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for y in range(kernel_size):
        for x in range(kernel_size):
            out += padded[y : y + arr.shape[0], x : x + arr.shape[1]]
    return out / float(kernel_size * kernel_size)


def _coeff_var(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return normalize_minmax(float(np.std(values) / (np.mean(values) + 1e-8)), 0.05, 0.85)


def _block_std_cv(residual: np.ndarray, block_size: int = 28) -> float:
    values = []
    for y in range(0, residual.shape[0] - block_size + 1, block_size):
        for x in range(0, residual.shape[1] - block_size + 1, block_size):
            values.append(float(np.std(residual[y : y + block_size, x : x + block_size])))
    return float(np.std(values) / (np.mean(values) + 1e-8)) if values else 0.0


def _grid_discontinuity(gray: np.ndarray) -> float:
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    grid_v = float(np.mean(vertical[:, 7::8])) if vertical.shape[1] > 8 else 0.0
    grid_h = float(np.mean(horizontal[7::8, :])) if horizontal.shape[0] > 8 else 0.0
    base = float((np.mean(vertical) + np.mean(horizontal)) / 2.0 + 1e-6)
    return clip01(((grid_v + grid_h) / 2.0) / (base * 2.2))


def _spectral_flatness(magnitude: np.ndarray) -> float:
    values = magnitude.ravel().astype(np.float64) + 1e-8
    geo = float(np.exp(np.mean(np.log(values))))
    arith = float(np.mean(values))
    return normalize_minmax(geo / (arith + 1e-8), 0.10, 0.55)


def _periodicity_score(log_magnitude: np.ndarray, radius_norm: np.ndarray) -> float:
    outer = log_magnitude[radius_norm > 0.25]
    if outer.size < 8:
        return 0.0
    peaks = np.percentile(outer, 99.3)
    mean = float(np.mean(outer))
    std = float(np.std(outer) + 1e-8)
    return normalize_minmax((peaks - mean) / std, 2.0, 5.5)


def _anisotropy_score(log_magnitude: np.ndarray) -> float:
    h, w = log_magnitude.shape
    vertical = float(np.mean(log_magnitude[:, w // 2 - 2 : w // 2 + 3]))
    horizontal = float(np.mean(log_magnitude[h // 2 - 2 : h // 2 + 3, :]))
    diagonal = float(np.mean(np.diag(log_magnitude)) + np.mean(np.diag(np.fliplr(log_magnitude)))) / 2.0
    base = float(np.mean(log_magnitude) + 1e-8)
    return normalize_minmax(max(abs(vertical - horizontal), abs(vertical - diagonal)) / base, 0.03, 0.24)


def _patch_statistics(
    arr: np.ndarray,
    gray: np.ndarray,
    grad: np.ndarray,
    block_size: int = 28,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    texture_values = []
    color_values = []
    infos = []
    for y in range(0, gray.shape[0] - block_size + 1, block_size):
        for x in range(0, gray.shape[1] - block_size + 1, block_size):
            patch_grad = grad[y : y + block_size, x : x + block_size]
            patch_arr = arr[y : y + block_size, x : x + block_size]
            texture = float(np.mean(patch_grad))
            color = float(np.std(np.mean(patch_arr, axis=(0, 1))))
            texture_values.append(texture)
            color_values.append(color)
            infos.append(
                {
                    "x": round(x / gray.shape[1], 6),
                    "y": round(y / gray.shape[0], 6),
                    "w": round(block_size / gray.shape[1], 6),
                    "h": round(block_size / gray.shape[0], 6),
                    "anomaly": 0.0,
                }
            )
    textures = np.asarray(texture_values, dtype=np.float64)
    colors = np.asarray(color_values, dtype=np.float64)
    if textures.size:
        t_z = np.abs((textures - np.mean(textures)) / (np.std(textures) + 1e-8))
        c_z = np.abs((colors - np.mean(colors)) / (np.std(colors) + 1e-8))
        for index, info in enumerate(infos):
            info["anomaly"] = round(normalize_minmax(float(0.65 * t_z[index] + 0.35 * c_z[index]), 0.5, 3.5), 6)
    return textures, colors, infos


def _skin_proxy(arr: np.ndarray, gray: np.ndarray, grad: np.ndarray) -> tuple[float, float]:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    skin = (
        (r > 0.24)
        & (g > 0.15)
        & (b > 0.10)
        & (r >= g * 0.78)
        & (r > b * 1.02)
        & (g > b * 0.70)
        & ((maxc - minc) > 0.03)
    )
    center = np.zeros_like(skin, dtype=bool)
    center[36:188, 36:188] = True
    center_skin_mass = float(np.mean(skin & center)) / max(float(np.mean(center)), 1e-8)
    if np.any(skin):
        skin_gradient = float(np.mean(grad[skin]))
        skin_smoothness = clip01(1.0 - normalize_minmax(skin_gradient, 0.012, 0.10))
    else:
        skin_smoothness = 0.0
    return skin_smoothness, clip01(center_skin_mass)


def _text_edge_density(gray: np.ndarray, grad: np.ndarray) -> float:
    strong_edges = grad > np.percentile(grad, 88)
    dark = gray < np.percentile(gray, 38)
    bright = gray > np.percentile(gray, 62)
    contrast_edges = strong_edges & (dark | bright)
    row_runs = float(np.mean(np.sum(contrast_edges, axis=1) > gray.shape[1] * 0.08))
    col_runs = float(np.mean(np.sum(contrast_edges, axis=0) > gray.shape[0] * 0.08))
    edge_mass = float(np.mean(contrast_edges))
    return clip01(0.50 * normalize_minmax(edge_mass, 0.02, 0.18) + 0.25 * row_runs + 0.25 * col_runs)
