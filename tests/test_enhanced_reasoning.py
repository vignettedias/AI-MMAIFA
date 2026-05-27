from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from evaluation import evaluate_binary_predictions
from inference import default_registry
from pipeline import AuthenticityPipeline
from training import build_training_plan
from utils.image_io import create_demo_image, image_to_png_bytes, load_image


def test_pipeline_exposes_next_generation_evidence(tmp_path):
    image_path = create_demo_image(tmp_path / "sample.png", "fake")
    pipeline = AuthenticityPipeline(model_dir=tmp_path / "models", prefer_torch=False)
    result = pipeline.predict(image_path)

    assert "pixel_convnext_v2" in result.features
    assert "expert_diffusion_artifact" in result.features
    assert "forensic_diffusion_trace" in result.features
    assert "forensic_provenance_consistency" in result.features
    assert "semantic_multi_agent" in result.features
    assert "semantic_agent_physics" in result.features
    assert "evidence_fusion" in result.details
    assert "open_set_score" in result.details["uncertainty"]
    assert result.details["uncertainty"]["conformal_decision_set"]


def test_explainability_and_ood_endpoints(tmp_path):
    app.state.pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
    )
    app.state.explainer = None
    image = load_image(create_demo_image(tmp_path / "upload.png", "fake"))
    client = TestClient(app)
    files = {"file": ("upload.png", image_to_png_bytes(image), "image/png")}

    explanation = client.post("/explain", files=files)
    assert explanation.status_code == 200
    payload = explanation.json()
    assert "top_evidence" in payload["explanation"]
    assert "forensic_saliency" in payload["explanation"]["maps"]

    heatmap = client.post(
        "/heatmap?kind=frequency_anomaly",
        files={"file": ("upload.png", image_to_png_bytes(image), "image/png")},
    )
    assert heatmap.status_code == 200
    assert heatmap.json()["maps"]["frequency_anomaly"].startswith("data:image/png;base64,")

    ood = client.post(
        "/ood",
        files={"file": ("upload.png", image_to_png_bytes(image), "image/png")},
    )
    assert ood.status_code == 200
    assert "open_set_score" in ood.json()


def test_registry_training_plan_and_metrics_are_available():
    registry = default_registry()
    plan = build_training_plan()
    metrics = evaluate_binary_predictions([0.1, 0.8, 0.6], [0, 1, 1])

    assert "pixel_convnext_v2" in registry.manifest()
    assert "cross_modal_fusion_training" in plan.stages
    assert {"auroc", "auprc", "ece", "brier"} <= set(metrics)
