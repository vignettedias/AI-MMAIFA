from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import AuthenticityPipeline


app = FastAPI(
    title="Multi-Modal Image Authenticity Verification API",
    version="1.0.0",
    description="Pixel, forensic, semantic, fusion, and meta-classifier image authenticity service.",
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


def require_admin_key(x_admin_key: str | None) -> None:
    expected = os.getenv("AUTH_ADMIN_KEY", "creator-admin-key")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=403,
            detail="Only the admin/creator can train or teach the model.",
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class FeedbackRequest(BaseModel):
    label: str = Field(..., description="Correct label: REAL or FAKE/AI-GENERATED")
    features: dict[str, float]
    image_hash: str | None = None
    note: str = ""


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    image_bytes = await file.read()
    try:
        result = get_pipeline().predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc
    return result.to_dict(include_details=True)


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
