from __future__ import annotations

from utils.math_utils import clip01
from utils.types import FeatureVector, StreamScore


class FeatureAggregator:
    """Builds the ordered meta-classifier feature vector."""

    feature_names = [
        "pixel_efficientnet",
        "pixel_alexnet",
        "pixel_googlenet",
        "pixel_convnext_v2",
        "pixel_swin_v2",
        "pixel_dinov2",
        "pixel_clip_vit_l14",
        "pixel_siglip",
        "pixel_frequency_vit",
        "pixel_cross_scale_transformer",
        "pixel_patch_forensic_encoder",
        "expert_gan_artifact",
        "expert_diffusion_artifact",
        "expert_faceswap",
        "expert_inpainting",
        "expert_ai_upscaling",
        "expert_cgi_realism",
        "expert_social_recompression",
        "expert_screenshot_artifact",
        "pixel_modern_ensemble",
        "pixel_expert_disagreement",
        "pixel_ensemble",
        "forensic_ela",
        "forensic_fft",
        "forensic_noise",
        "forensic_compression",
        "forensic_prnu",
        "forensic_srm_residual",
        "forensic_bayar_residual",
        "forensic_dncnn_residual",
        "forensic_patch_noise_consistency",
        "forensic_camera_noise_coherence",
        "forensic_noise_advanced",
        "forensic_diffusion_denoising",
        "forensic_diffusion_latent_upsampling",
        "forensic_diffusion_timestep",
        "forensic_diffusion_microtexture",
        "forensic_diffusion_edge_oversampling",
        "forensic_diffusion_spectral_flattening",
        "forensic_diffusion_hallucination",
        "forensic_diffusion_trace",
        "forensic_radial_spectral",
        "forensic_wavelet",
        "forensic_fourier_pyramid",
        "forensic_anisotropic_spectral",
        "forensic_periodicity",
        "forensic_spectral_entropy",
        "forensic_high_frequency_attenuation",
        "forensic_band_energy",
        "forensic_advanced_frequency",
        "forensic_provenance_cfa",
        "forensic_provenance_lens",
        "forensic_provenance_rolling_shutter",
        "forensic_provenance_exif",
        "forensic_provenance_optical_aberration",
        "forensic_provenance_chromatic_aberration",
        "forensic_provenance_consistency",
        "forensic_ensemble",
        "semantic_anomaly",
        "semantic_portrait_synthetic",
        "semantic_stylized_composite",
        "semantic_low_resolution_composite",
        "semantic_nature_render",
        "semantic_architecture_render",
        "semantic_agent_anatomy",
        "semantic_agent_typography",
        "semantic_agent_lighting",
        "semantic_agent_reflection",
        "semantic_agent_perspective",
        "semantic_agent_object_relation",
        "semantic_agent_human_realism",
        "semantic_agent_contextual_logic",
        "semantic_agent_scene_geometry",
        "semantic_agent_physics",
        "semantic_multi_agent",
        "semantic_physical_realism",
        "semantic_typography",
        "forensic_spectral_intelligence",
        "forensic_camera_provenance",
        "forensic_manipulation",
        "forensic_metadata_provenance",
    ]

    def transform(
        self,
        pixel_scores: dict[str, StreamScore],
        forensic_scores: dict[str, StreamScore],
        semantic_scores: dict[str, StreamScore],
    ) -> FeatureVector:
        merged = {}
        merged.update(pixel_scores)
        merged.update(forensic_scores)
        merged.update(semantic_scores)
        values = [
            clip01(merged.get(name, StreamScore(name, 0.0)).score)
            for name in self.feature_names
        ]
        return FeatureVector(names=list(self.feature_names), values=values)

    def from_dict(self, features: dict[str, float]) -> FeatureVector:
        values = [clip01(features.get(name, 0.0)) for name in self.feature_names]
        return FeatureVector(names=list(self.feature_names), values=values)
