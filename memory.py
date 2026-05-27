from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.math_utils import clip01


class DynamicMemoryLearner:
    """Persistent feature memory for user-corrected predictions.

    This is an example-based learner: exact image hashes override immediately,
    while nearby feature vectors blend into the meta probability. It complements
    the static meta-classifier without pretending the system can be error-free.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        feature_names: Sequence[str] | None = None,
        filename: str = "feedback_memory.json",
        max_examples: int = 2000,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.model_dir / filename
        self.feature_names = list(feature_names or [])
        self.max_examples = max_examples
        self.examples: list[dict[str, Any]] = []
        self._load()

    def learn(
        self,
        *,
        label: str | int,
        features: dict[str, float] | Sequence[float],
        image_hash: str | None = None,
        source: str = "user_feedback",
        note: str = "",
        trust: float | None = None,
        allow_exact_override: bool | None = None,
    ) -> dict[str, Any]:
        label_value = _label_to_int(label)
        vector = self._coerce_vector(features)
        if vector is None:
            raise ValueError("Feedback requires a valid feature vector.")
        trust_value = clip01(
            1.0 if trust is None and source == "user_feedback" else 0.35 if trust is None else trust
        )
        exact_override = bool(source == "user_feedback" if allow_exact_override is None else allow_exact_override)

        if image_hash:
            for example in self.examples:
                if example.get("image_hash") == image_hash:
                    existing_trust = float(example.get("trust", 1.0))
                    if existing_trust > trust_value and source != "user_feedback":
                        return self.stats(last_action="skipped_lower_trust")
                    example.update(
                        {
                            "label": label_value,
                            "label_name": "FAKE" if label_value else "REAL",
                            "features": vector.tolist(),
                            "source": source,
                            "note": note,
                            "trust": trust_value,
                            "allow_exact_override": exact_override,
                            "updated_at": time.time(),
                        }
                    )
                    self._save()
                    return self.stats(last_action="updated")

        self.examples.append(
            {
                "id": str(uuid.uuid4()),
                "image_hash": image_hash,
                "label": label_value,
                "label_name": "FAKE" if label_value else "REAL",
                "features": vector.tolist(),
                "source": source,
                "note": note,
                "trust": trust_value,
                "allow_exact_override": exact_override,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[-self.max_examples :]
        self._save()
        return self.stats(last_action="created")

    def adjust_probability(
        self,
        *,
        base_probability: float,
        features: dict[str, float] | Sequence[float],
        image_hash: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        base_probability = clip01(base_probability)
        vector = self._coerce_vector(features)
        if vector is None or not self.examples:
            return base_probability, {
                "strategy": "no_memory",
                "base_probability": base_probability,
                "final_probability": base_probability,
                "examples": len(self.examples),
            }

        if image_hash:
            for example in reversed(self.examples):
                if example.get("image_hash") == image_hash and example.get("allow_exact_override", True):
                    memory_probability = float(example["label"])
                    return memory_probability, {
                        "strategy": "exact_image_memory",
                        "base_probability": base_probability,
                        "final_probability": memory_probability,
                        "memory_probability": memory_probability,
                        "strength": 1.0,
                        "examples": len(self.examples),
                        "matched": [
                            {
                                "id": example.get("id"),
                                "label": example.get("label_name"),
                                "source": example.get("source"),
                                "trust": round(float(example.get("trust", 1.0)), 6),
                                "distance": 0.0,
                                "similarity": 1.0,
                            }
                        ],
                    }

        neighbors = self._nearest_neighbors(vector, limit=7)
        strong_neighbors = [item for item in neighbors if item["similarity"] >= 0.58]
        if not strong_neighbors:
            return base_probability, {
                "strategy": "memory_no_close_match",
                "base_probability": base_probability,
                "final_probability": base_probability,
                "examples": len(self.examples),
                "nearest": neighbors[:3],
            }

        weights = np.asarray(
            [item["similarity"] * item["trust"] for item in strong_neighbors],
            dtype=np.float64,
        )
        labels = np.asarray([item["label"] for item in strong_neighbors], dtype=np.float64)
        memory_probability = clip01(float(np.dot(weights, labels) / max(float(np.sum(weights)), 1e-8)))
        strength = clip01(float(np.mean(weights)))
        blended = clip01((1.0 - strength) * base_probability + strength * memory_probability)
        if strength >= 0.82:
            blended = memory_probability

        return blended, {
            "strategy": "nearest_feature_memory",
            "base_probability": base_probability,
            "final_probability": blended,
            "memory_probability": memory_probability,
            "strength": strength,
            "examples": len(self.examples),
            "matched": [
                {
                    "id": item["id"],
                    "label": "FAKE" if item["label"] else "REAL",
                    "source": item["source"],
                    "trust": round(float(item["trust"]), 6),
                    "distance": round(float(item["distance"]), 6),
                    "similarity": round(float(item["similarity"]), 6),
                }
                for item in strong_neighbors[:5]
            ],
        }

    def stats(self, last_action: str | None = None) -> dict[str, Any]:
        real = sum(1 for item in self.examples if int(item.get("label", 0)) == 0)
        fake = sum(1 for item in self.examples if int(item.get("label", 0)) == 1)
        confirmed = sum(1 for item in self.examples if item.get("source") == "user_feedback")
        automatic = sum(1 for item in self.examples if item.get("source") != "user_feedback")
        payload: dict[str, Any] = {
            "examples": len(self.examples),
            "real_examples": real,
            "fake_examples": fake,
            "confirmed_examples": confirmed,
            "automatic_examples": automatic,
            "memory_path": str(self.path),
        }
        if last_action:
            payload["last_action"] = last_action
        return payload

    def _nearest_neighbors(self, vector: np.ndarray, limit: int) -> list[dict[str, Any]]:
        neighbors = []
        for example in self.examples:
            other = np.asarray(example.get("features", []), dtype=np.float64)
            if other.shape != vector.shape:
                continue
            distance = float(np.sqrt(np.mean((vector - other) ** 2)))
            similarity = float(np.exp(-((distance / 0.28) ** 2)))
            neighbors.append(
                {
                    "id": example.get("id"),
                    "label": int(example.get("label", 0)),
                    "source": example.get("source", "unknown"),
                    "trust": float(example.get("trust", 1.0)),
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return sorted(neighbors, key=lambda item: item["distance"])[:limit]

    def _coerce_vector(
        self,
        features: dict[str, float] | Sequence[float],
    ) -> np.ndarray | None:
        if isinstance(features, dict):
            names = self.feature_names or list(features)
            values = [features.get(name, 0.0) for name in names]
        else:
            values = list(features)
        if not values:
            return None
        return np.asarray([clip01(float(value)) for value in values], dtype=np.float64)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.examples = list(payload.get("examples", []))
        except Exception:
            self.examples = []

    def _save(self) -> None:
        payload = {
            "version": 1,
            "feature_names": self.feature_names,
            "examples": self.examples,
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _label_to_int(label: str | int) -> int:
    if isinstance(label, str):
        normalized = label.strip().upper()
        if normalized in {"FAKE", "AI", "AI-GENERATED", "AI_GENERATED", "GENERATED"}:
            return 1
        if normalized == "REAL":
            return 0
        raise ValueError("Label must be REAL or FAKE/AI-GENERATED.")
    return 1 if int(label) else 0
