from __future__ import annotations

import os
import secrets
import time
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from explainability import ExplainabilityEngine
from pipeline import AuthenticityPipeline


app = FastAPI(
    title="Multi-Modal Image Authenticity Verification API",
    version="2.0.0",
    description="Evidence-centric forensic reasoning API for image authenticity, uncertainty, OOD, heatmaps, and human review.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> AuthenticityPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        model_dir = os.getenv("AUTH_MODEL_DIR", "models")
        use_clip = os.getenv("AUTH_USE_CLIP", "0").lower() in {"1", "true", "yes"}
        pipeline = AuthenticityPipeline(model_dir=model_dir, use_clip=use_clip)
        app.state.pipeline = pipeline
    return pipeline


def get_explainer() -> ExplainabilityEngine:
    explainer = getattr(app.state, "explainer", None)
    if explainer is None:
        explainer = ExplainabilityEngine()
        app.state.explainer = explainer
    return explainer


def require_admin_key(x_admin_key: str | None) -> None:
    expected = os.getenv("AUTH_ADMIN_KEY", "creator-admin-key")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=403,
            detail="Only the admin/creator can train or teach the model.",
        )


def enqueue_review(result: Any) -> None:
    review = result.details.get("review_decision", {})
    if not review.get("requires_human_review"):
        return
    queue = getattr(app.state, "review_queue", None)
    if queue is None:
        queue = []
        app.state.review_queue = queue
    existing = next((item for item in queue if item.get("image_hash") == result.image_hash), None)
    payload = {
        "image_hash": result.image_hash,
        "label": result.label,
        "model_label": result.details.get("model_label"),
        "confidence": round(result.confidence, 2),
        "fake_probability": round(result.fake_probability, 6),
        "reasons": review.get("reasons", []),
        "uncertainty": result.details.get("uncertainty", {}),
        "created_at": time.time(),
    }
    if existing:
        existing.update(payload)
    else:
        queue.append(payload)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class FeedbackRequest(BaseModel):
    label: str = Field(..., description="Correct label: REAL or FAKE/AI-GENERATED")
    features: dict[str, float]
    image_hash: str | None = None
    note: str = ""


class ReviewCorrectionRequest(BaseModel):
    image_hash: str
    label: str = Field(..., description="Correct label: REAL, FAKE, AI_GENERATED, or MANIPULATED")
    features: dict[str, float]
    note: str = ""


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    return result.to_dict(include_details=True)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    results = []
    for file in files:
        image_bytes = await _read_image_upload(file)
        result = _predict_bytes(image_bytes)
        results.append(
            {
                "filename": file.filename,
                "prediction": result.to_dict(include_details=True),
            }
        )
    return {"items": results, "count": len(results)}


@app.post("/explain")
async def explain(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    return {
        "prediction": result.to_dict(include_details=True),
        "explanation": get_explainer().explain(image_bytes, result),
    }


@app.post("/heatmap")
async def heatmap(
    file: UploadFile = File(...),
    kind: str | None = Query(default=None),
) -> dict[str, Any]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    maps = get_explainer().heatmaps(image_bytes)
    if kind:
        if kind not in maps:
            raise HTTPException(status_code=404, detail=f"Unknown heatmap kind '{kind}'.")
        maps = {kind: maps[kind]}
    return {"image_hash": result.image_hash, "maps": maps}


@app.post("/evidence")
async def evidence(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    explanation = get_explainer().explain(image_bytes, result)
    return {
        "image_hash": result.image_hash,
        "decision": result.label,
        "top_evidence": explanation["top_evidence"],
        "stream_details": result.details.get("stream_details", {}),
        "fusion": result.details.get("evidence_fusion", {}),
    }


@app.post("/uncertainty")
async def uncertainty(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    return {
        "image_hash": result.image_hash,
        "decision": result.label,
        "uncertainty": result.details.get("uncertainty", {}),
        "review_decision": result.details.get("review_decision", {}),
    }


@app.post("/ood")
async def ood(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await _read_image_upload(file)
    result = _predict_bytes(image_bytes)
    uncertainty_payload = result.details.get("uncertainty", {})
    return {
        "image_hash": result.image_hash,
        "ood_score": uncertainty_payload.get("ood_score"),
        "open_set_score": uncertainty_payload.get("open_set_score"),
        "mahalanobis_score": uncertainty_payload.get("mahalanobis_score"),
        "energy_score": uncertainty_payload.get("energy_score"),
        "nearest_neighbor_anomaly": uncertainty_payload.get("nearest_neighbor_anomaly"),
        "decision": "HUMAN_REVIEW_REQUIRED"
        if float(uncertainty_payload.get("ood_score", 0.0)) >= 0.72
        else result.label,
    }


@app.get("/review")
async def review(
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin_key(x_admin_key)
    queue = getattr(app.state, "review_queue", [])
    return {"items": queue, "count": len(queue)}


@app.post("/review")
async def review_correction(
    payload: ReviewCorrectionRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin_key(x_admin_key)
    result = get_pipeline().learn_from_feedback(
        label=payload.label,
        features=payload.features,
        image_hash=payload.image_hash,
        note=payload.note,
    )
    queue = getattr(app.state, "review_queue", [])
    app.state.review_queue = [
        item for item in queue if item.get("image_hash") != payload.image_hash
    ]
    return {"status": "review_recorded", **result}


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket) -> None:
    await websocket.accept()
    stages = [
        "upload_received",
        "pixel_experts",
        "forensic_streams",
        "semantic_agents",
        "evidence_fusion",
        "uncertainty_and_review_policy",
    ]
    for index, stage in enumerate(stages, start=1):
        await websocket.send_json(
            {"stage": stage, "progress": round(index / len(stages), 3)}
        )
    await websocket.close()


async def _read_image_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    return await file.read()


def _predict_bytes(image_bytes: bytes):
    try:
        result = get_pipeline().predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc
    enqueue_review(result)
    return result


@app.post("/feedback")
async def feedback(
    payload: FeedbackRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin_key(x_admin_key)
    try:
        result = get_pipeline().learn_from_feedback(
            label=payload.label,
            features=payload.features,
            image_hash=payload.image_hash,
            note=payload.note,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not store feedback: {exc}") from exc
    return {"status": "learned", **result}


@app.get("/memory/stats")
async def memory_stats(x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin_key(x_admin_key)
    return get_pipeline().memory_stats()


@app.get("/review/queue")
async def review_queue(x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin_key(x_admin_key)
    queue = getattr(app.state, "review_queue", [])
    return {"items": queue, "count": len(queue)}
