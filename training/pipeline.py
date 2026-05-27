from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import default_dataset_registry
from inference import default_registry


@dataclass(frozen=True)
class TrainingPlan:
    stages: tuple[str, ...]
    datasets: dict[str, Any]
    models: dict[str, Any]
    robustness: tuple[str, ...]
    calibration: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.stages),
            "datasets": self.datasets,
            "models": self.models,
            "robustness": list(self.robustness),
            "calibration": list(self.calibration),
        }


def build_training_plan(model_root: str = "models") -> TrainingPlan:
    return TrainingPlan(
        stages=(
            "dataset_manifest_validation",
            "pixel_foundation_finetuning",
            "specialist_expert_training",
            "forensic_probe_calibration",
            "semantic_agent_distillation",
            "cross_modal_fusion_training",
            "ood_density_indexing",
            "temperature_and_dirichlet_calibration",
            "adversarial_and_compression_robustness_evaluation",
            "onnx_tensorrt_export",
        ),
        datasets=default_dataset_registry().manifest(),
        models=default_registry(model_root).manifest(),
        robustness=(
            "PGD adversarial training",
            "randomized smoothing",
            "feature denoising",
            "frequency perturbation defense",
            "compression randomization",
            "adversarial consistency checks",
        ),
        calibration=(
            "temperature scaling",
            "Dirichlet calibration",
            "conformal prediction",
            "evidential uncertainty",
            "deep ensemble disagreement",
        ),
    )
