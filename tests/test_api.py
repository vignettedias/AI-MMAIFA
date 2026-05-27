from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from pipeline import AuthenticityPipeline
from utils.image_io import create_demo_image, image_to_png_bytes, load_image


def test_predict_endpoint_accepts_image(tmp_path):
    app.state.pipeline = None
    image = load_image(create_demo_image(tmp_path / "upload.png", "fake"))
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("upload.png", image_to_png_bytes(image), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"REAL", "AI_GENERATED", "MANIPULATED", "HUMAN_REVIEW_REQUIRED"}
    assert 50.0 <= payload["confidence"] <= 100.0
    assert "uncertainty" in payload["details"]


def test_feedback_endpoint_teaches_exact_image_memory(tmp_path):
    app.state.pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
    )
    image = load_image(create_demo_image(tmp_path / "upload.png", "fake"))
    client = TestClient(app)
    prediction = client.post(
        "/predict",
        files={"file": ("upload.png", image_to_png_bytes(image), "image/png")},
    ).json()

    corrected_label = "REAL" if prediction["details"]["model_label"] == "FAKE" else "FAKE"
    feedback = client.post(
        "/feedback",
        headers={"X-Admin-Key": "creator-admin-key"},
        json={
            "label": corrected_label,
            "features": prediction["features"],
            "image_hash": prediction["image_hash"],
            "note": "test correction",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["status"] == "learned"

    learned_prediction = client.post(
        "/predict",
        files={"file": ("upload.png", image_to_png_bytes(image), "image/png")},
    ).json()
    expected_labels = {"AI_GENERATED", "MANIPULATED"} if corrected_label == "FAKE" else {"REAL"}
    assert learned_prediction["label"] in expected_labels
    assert learned_prediction["details"]["dynamic_memory"]["strategy"] == "exact_image_memory"


def test_feedback_endpoint_rejects_non_admin(tmp_path):
    app.state.pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
    )
    client = TestClient(app)
    response = client.post(
        "/feedback",
        json={
            "label": "FAKE",
            "features": {"pixel_ensemble": 0.2},
            "image_hash": "abc123",
        },
    )

    assert response.status_code == 403


def test_memory_stats_endpoint_is_admin_only(tmp_path):
    app.state.pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
    )
    client = TestClient(app)
    assert client.get("/memory/stats").status_code == 403

    response = client.get("/memory/stats", headers={"X-Admin-Key": "creator-admin-key"})
    assert response.status_code == 200
    assert "examples" in response.json()


def test_review_queue_endpoint_is_admin_only(tmp_path):
    app.state.pipeline = AuthenticityPipeline(
        model_dir=tmp_path / "models",
        prefer_torch=False,
    )
    app.state.review_queue = []
    client = TestClient(app)
    assert client.get("/review/queue").status_code == 403

    response = client.get("/review/queue", headers={"X-Admin-Key": "creator-admin-key"})
    assert response.status_code == 200
    assert response.json()["count"] == 0
