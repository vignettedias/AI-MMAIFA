from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetDomain:
    name: str
    label_space: tuple[str, ...]
    sources: tuple[str, ...]
    required: bool = False


@dataclass
class DatasetRegistry:
    domains: list[DatasetDomain] = field(default_factory=list)

    def add(self, domain: DatasetDomain) -> None:
        self.domains.append(domain)

    def manifest(self) -> dict[str, Any]:
        return {
            "domains": [
                {
                    "name": domain.name,
                    "label_space": list(domain.label_space),
                    "sources": list(domain.sources),
                    "required": domain.required,
                }
                for domain in self.domains
            ],
            "augmentation_policy": [
                "jpeg_recompression",
                "blur",
                "sharpening",
                "noise_injection",
                "scaling",
                "screenshotting",
                "gamma_shift",
                "color_perturbation",
                "watermarking",
                "social_media_degradation",
            ],
        }


def default_dataset_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    for name, labels, sources, required in [
        ("dslr_photography", ("REAL",), ("DSLR captures", "raw-to-jpeg exports"), True),
        ("smartphone_photos", ("REAL",), ("iOS", "Android", "social uploads"), True),
        ("screenshots_memes", ("REAL", "MANIPULATED"), ("screenshots", "memes"), False),
        ("news_surveillance", ("REAL", "MANIPULATED"), ("newswire", "surveillance"), False),
        ("medical_satellite", ("REAL", "SYNTHETIC"), ("medical", "satellite"), False),
        ("cgi_renders", ("SYNTHETIC",), ("Blender", "Unreal", "Cinema4D"), True),
        ("gan_outputs", ("SYNTHETIC",), ("StyleGAN", "BigGAN", "ProGAN"), True),
        ("diffusion_outputs", ("SYNTHETIC",), ("Stable Diffusion", "SDXL", "Midjourney", "Flux", "DALL-E", "Firefly", "Kandinsky", "Imagen"), True),
        ("face_swap_inpainting", ("MANIPULATED",), ("face swap", "inpainting", "outpainting"), True),
        ("restoration_upscaling", ("MANIPULATED", "SYNTHETIC"), ("AI upscaling", "AI restoration"), False),
    ]:
        registry.add(DatasetDomain(name, labels, sources, required))
    return registry
