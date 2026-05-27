from __future__ import annotations

import json
from pathlib import Path

from inference import default_registry
from training import build_training_plan


def export_manifest(model_root: str = "models", output: str = "deployment/export_manifest.json") -> Path:
    """Write an export manifest consumed by ONNX/TensorRT/Triton jobs."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_registry": default_registry(model_root).manifest(),
        "training_plan": build_training_plan(model_root).as_dict(),
        "targets": ["onnx", "tensorrt", "triton"],
        "precision": ["fp32", "fp16", "int8_qat"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(export_manifest())
