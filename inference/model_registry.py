from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpertBackend:
    name: str
    task: str
    architecture: str
    artifact: str | None = None
    optional: bool = True
    notes: str = ""


@dataclass
class ModelRegistry:
    """Declarative registry for heavy forensic backends.

    Local tests use deterministic fallback streams. Production deployments can
    point these entries to ONNX, TensorRT, PyTorch, or Triton model artifacts
    without changing pipeline contracts.
    """

    root: Path = Path("models")
    backends: dict[str, ExpertBackend] = field(default_factory=dict)

    def register(self, backend: ExpertBackend) -> None:
        self.backends[backend.name] = backend

    def resolve(self, name: str) -> dict[str, Any]:
        backend = self.backends[name]
        artifact = Path(backend.artifact) if backend.artifact else self.root / f"{name}.onnx"
        return {
            "name": backend.name,
            "task": backend.task,
            "architecture": backend.architecture,
            "artifact": str(artifact),
            "available": artifact.exists(),
            "optional": backend.optional,
            "notes": backend.notes,
        }

    def manifest(self) -> dict[str, Any]:
        return {name: self.resolve(name) for name in sorted(self.backends)}


def default_registry(model_root: str | Path = "models") -> ModelRegistry:
    registry = ModelRegistry(root=Path(model_root))
    for name, task, arch in [
        ("pixel_convnext_v2", "pixel_foundation", "ConvNeXt-V2"),
        ("pixel_swin_v2", "pixel_foundation", "Swin Transformer V2"),
        ("pixel_dinov2", "self_supervised_embedding", "DINOv2"),
        ("pixel_clip_vit_l14", "vision_language", "CLIP ViT-L/14"),
        ("pixel_siglip", "vision_language", "SigLIP"),
        ("pixel_frequency_vit", "frequency_domain", "Frequency-aware ViT"),
        ("pixel_cross_scale_transformer", "cross_scale", "Cross-scale Transformer"),
        ("pixel_patch_forensic_encoder", "patch_forensics", "Patch forensic encoder"),
        ("vlm_florence2", "semantic_agent", "Florence-2"),
        ("vlm_qwen_vl", "semantic_agent", "Qwen-VL"),
        ("vlm_llava", "semantic_agent", "LLaVA"),
        ("fusion_transformer", "fusion", "Cross-modal evidence transformer"),
        ("calibrator_dirichlet", "calibration", "Dirichlet calibrator"),
    ]:
        registry.register(
            ExpertBackend(
                name=name,
                task=task,
                architecture=arch,
                notes="Optional production backend; deterministic proxy is used when absent.",
            )
        )
    return registry
