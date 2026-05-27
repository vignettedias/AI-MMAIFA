from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from utils.math_utils import clip01


class MetaClassifier:
    """XGBoost meta-classifier with a deterministic weighted fallback."""

    def __init__(
        self,
        model_dir: str | Path = "models",
        feature_names: Sequence[str] | None = None,
        min_trusted_xgb_samples: int = 30,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(feature_names or [])
        self.config_path = self.model_dir / "meta_config.json"
        self.xgb_path = self.model_dir / "meta_xgboost.json"
        self.backend = "weighted_fallback"
        self.model = None
        self.min_trusted_xgb_samples = min_trusted_xgb_samples
        self.training_samples = 0
        self.training_class_counts = {"real": 0, "fake": 0}
        self.last_decision: dict[str, object] = {}
        self.last_profile_details: dict[str, object] = {}
        self.weights = self._default_weights(len(self.feature_names))
        self.bias = 0.0
        self._load()

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> dict[str, object]:
        matrix = np.asarray(x, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            self._save_config()
            return {"backend": self.backend, "samples": int(matrix.shape[0] if matrix.ndim else 0)}

        self.training_samples = int(matrix.shape[0])
        self.training_class_counts = {
            "real": int(np.sum(labels == 0)),
            "fake": int(np.sum(labels == 1)),
        }
        self.weights = self._default_weights(matrix.shape[1])
        try:
            import xgboost as xgb

            feature_names = self._xgb_feature_names(matrix.shape[1])
            dtrain = xgb.DMatrix(matrix, label=labels, feature_names=feature_names)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 3,
                "eta": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "seed": 42,
                "verbosity": 0,
            }
            model = xgb.train(params, dtrain, num_boost_round=80)
            model.save_model(self.xgb_path)
            self.model = model
            self.backend = "xgboost"
        except Exception:
            self._fit_weighted_fallback(matrix, labels)
            self.backend = "weighted_trained_fallback"

        self._save_config()
        return {
            "backend": self.backend,
            "samples": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "artifact": str(self.xgb_path if self.backend == "xgboost" else self.config_path),
        }

    def predict_proba(self, values: Sequence[float] | dict[str, float]) -> float:
        vector = self._coerce_vector(values)
        evidence_prob = self._weighted_evidence_probability(vector)
        if self.model is not None:
            try:
                import xgboost as xgb

                dmatrix = xgb.DMatrix(
                    vector.reshape(1, -1),
                    feature_names=self._xgb_feature_names(len(vector)),
                )
                prob = self.model.predict(dmatrix)[0]
                xgb_prob = clip01(float(prob))
                if self._trusted_xgb():
                    self.last_decision = {
                        "strategy": "xgboost",
                        "xgboost_probability": xgb_prob,
                        "weighted_evidence_probability": evidence_prob,
                        "weighted_profile": self.last_profile_details,
                        "training_samples": self.training_samples,
                        "note": "XGBoost trusted because sufficient training samples are available.",
                    }
                    return xgb_prob
                self.last_decision = {
                    "strategy": "weighted_profile_guard",
                    "xgboost_probability": xgb_prob,
                    "weighted_evidence_probability": evidence_prob,
                    "weighted_profile": self.last_profile_details,
                    "training_samples": self.training_samples,
                    "note": (
                        "XGBoost artifact is undertrained/demo-sized, so final probability "
                        "uses weighted multi-stream evidence as described in the synopsis."
                    ),
                }
                return evidence_prob
            except Exception:
                pass
        self.last_decision = {
            "strategy": self.backend,
            "weighted_evidence_probability": evidence_prob,
            "weighted_profile": self.last_profile_details,
            "training_samples": self.training_samples,
        }
        return evidence_prob

    def _coerce_vector(self, values: Sequence[float] | dict[str, float]) -> np.ndarray:
        if isinstance(values, dict):
            vector = [values.get(name, 0.0) for name in self.feature_names]
        else:
            vector = list(values)
        return np.asarray([clip01(float(value)) for value in vector], dtype=np.float32)

    def _load(self) -> None:
        configured_feature_names = list(self.feature_names)
        feature_mismatch = False
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            saved_feature_names = payload.get("feature_names", [])
            feature_mismatch = bool(
                configured_feature_names
                and saved_feature_names
                and list(saved_feature_names) != configured_feature_names
            )
            if feature_mismatch:
                self.feature_names = configured_feature_names
                self.backend = "weighted_fallback"
                self.training_samples = 0
                self.training_class_counts = {"real": 0, "fake": 0}
                self.weights = self._default_weights(len(self.feature_names))
                self.bias = 0.0
            else:
                self.feature_names = payload.get("feature_names", self.feature_names)
                self.backend = payload.get("backend", self.backend)
                self.training_samples = int(payload.get("training_samples", 0))
                self.training_class_counts = payload.get(
                    "training_class_counts",
                    self.training_class_counts,
                )
                self.weights = np.asarray(
                    payload.get("weights", self._default_weights(len(self.feature_names))),
                    dtype=np.float32,
                )
                self.bias = float(payload.get("bias", 0.0))
        if self.xgb_path.exists() and not feature_mismatch:
            try:
                import xgboost as xgb

                model = xgb.Booster()
                model.load_model(self.xgb_path)
                if self._xgb_artifact_matches_features(model):
                    self.model = model
                    self.backend = "xgboost"
                else:
                    self.model = None
                    self.backend = "weighted_fallback"
            except Exception:
                self.model = None
        if feature_mismatch:
            self._save_config()

    def _save_config(self) -> None:
        payload = {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "training_samples": self.training_samples,
            "training_class_counts": self.training_class_counts,
            "min_trusted_xgb_samples": self.min_trusted_xgb_samples,
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _fit_weighted_fallback(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        weights = self._default_weights(matrix.shape[1])
        bias = 0.0
        lr = 0.5
        for _ in range(400):
            logits = matrix @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
            error = probs - labels
            weights -= lr * (matrix.T @ error / len(matrix) + 0.02 * weights)
            bias -= lr * float(np.mean(error))
        self.weights = weights
        self.bias = bias

    def _xgb_feature_names(self, length: int) -> list[str] | None:
        if len(self.feature_names) == length and all(self.feature_names):
            return list(self.feature_names)
        return None

    def _trusted_xgb(self) -> bool:
        return (
            self.training_samples >= self.min_trusted_xgb_samples
            and self.training_class_counts.get("real", 0) > 0
            and self.training_class_counts.get("fake", 0) > 0
        )

    def _xgb_artifact_matches_features(self, model: object) -> bool:
        try:
            model_features = int(model.num_features())
        except Exception:
            model_features = len(self.feature_names)
        if self.feature_names and model_features != len(self.feature_names):
            return False

        artifact_names = getattr(model, "feature_names", None)
        if artifact_names and self.feature_names and list(artifact_names) != self.feature_names:
            return False
        return True

    def _weighted_evidence_probability(self, vector: np.ndarray) -> float:
        weights = self.weights[: len(vector)]
        if len(weights) != len(vector) or float(np.sum(weights)) <= 0:
            weights = self._default_weights(len(vector))
        base_probability = clip01(
            float(np.dot(weights, vector) / max(float(np.sum(weights)), 1e-8))
        )
        final_probability = base_probability
        adjustments: list[dict[str, float | str]] = []

        if len(vector) >= 9:
            fft_score = self._feature_value(vector, "forensic_fft", 5)
            noise_score = self._feature_value(vector, "forensic_noise", 6)
            compression_score = self._feature_value(vector, "forensic_compression", 7)
            forensic_ensemble = self._feature_value(vector, "forensic_ensemble", 8)
            semantic_score = self._feature_value(vector, "semantic_anomaly", 9)
            portrait_score = self._feature_value(vector, "semantic_portrait_synthetic", 10)
            composite_score = self._feature_value(vector, "semantic_stylized_composite", 11)
            low_resolution_score = self._feature_value(
                vector,
                "semantic_low_resolution_composite",
                12,
            )
            nature_score = self._feature_value(vector, "semantic_nature_render", 13)
            architecture_score = self._feature_value(
                vector,
                "semantic_architecture_render",
                14,
            )
            semantic_agent_score = max(
                self._feature_value(vector, "semantic_multi_agent"),
                self._feature_value(vector, "semantic_physical_realism"),
                self._feature_value(vector, "semantic_typography"),
            )
            diffusion_score = max(
                self._feature_value(vector, "expert_diffusion_artifact"),
                self._feature_value(vector, "forensic_diffusion_trace"),
                self._feature_value(vector, "forensic_diffusion_spectral_flattening"),
            )
            pixel_ensemble = self._feature_value(vector, "pixel_ensemble", 3)
            ela_score = self._feature_value(vector, "forensic_ela", 4)

            spectral_strength = min(
                clip01((fft_score - 0.68) / 0.18),
                clip01((noise_score - 0.50) / 0.20),
                clip01((forensic_ensemble - 0.42) / 0.16),
            )
            if spectral_strength >= 0.25:
                forensic_consensus_probability = clip01(0.50 + 0.25 * spectral_strength)
                if forensic_consensus_probability > final_probability:
                    final_probability = forensic_consensus_probability
                    adjustments.append(
                        {
                            "name": "forensic_frequency_noise_consensus",
                            "strength": round(spectral_strength, 6),
                            "probability": round(forensic_consensus_probability, 6),
                        }
                    )

            compression_strength = min(
                clip01((compression_score - 0.50) / 0.25),
                clip01((forensic_ensemble - 0.36) / 0.20),
            )
            if compression_strength >= 0.35:
                compression_probability = clip01(0.50 + 0.18 * compression_strength)
                if compression_probability > final_probability:
                    final_probability = compression_probability
                    adjustments.append(
                        {
                            "name": "compression_resampling_consensus",
                            "strength": round(compression_strength, 6),
                            "probability": round(compression_probability, 6),
                        }
                    )

            semantic_synthetic_score = max(
                semantic_score,
                portrait_score,
                composite_score,
                low_resolution_score,
                architecture_score,
                semantic_agent_score,
            )
            portrait_strength = min(
                clip01((semantic_synthetic_score - 0.52) / 0.28),
                clip01((0.32 - pixel_ensemble) / 0.24),
                clip01((0.10 - ela_score) / 0.10),
            )
            if portrait_strength >= 0.25:
                semantic_portrait_probability = clip01(0.72 + 0.18 * portrait_strength)
                if semantic_portrait_probability > final_probability:
                    final_probability = semantic_portrait_probability
                    adjustments.append(
                        {
                            "name": "semantic_synthetic_consensus",
                            "strength": round(portrait_strength, 6),
                            "probability": round(semantic_portrait_probability, 6),
                        }
                    )

            stylized_strength = min(
                clip01((composite_score - 0.65) / 0.25),
                clip01((fft_score - 0.62) / 0.20),
                clip01((0.35 - pixel_ensemble) / 0.25),
                clip01((0.12 - ela_score) / 0.12),
            )
            if stylized_strength >= 0.25:
                stylized_probability = clip01(0.78 + 0.16 * stylized_strength)
                if stylized_probability > final_probability:
                    final_probability = stylized_probability
                    adjustments.append(
                        {
                            "name": "stylized_composite_render_consensus",
                            "strength": round(stylized_strength, 6),
                            "probability": round(stylized_probability, 6),
                        }
                    )

            landscape_strength = min(
                clip01((nature_score - 0.56) / 0.30),
                clip01((fft_score - 0.68) / 0.18),
                clip01((0.35 - pixel_ensemble) / 0.25),
                clip01((0.12 - ela_score) / 0.12),
            )
            if landscape_strength >= 0.25:
                landscape_probability = clip01(0.78 + 0.16 * landscape_strength)
                if landscape_probability > final_probability:
                    final_probability = landscape_probability
                    adjustments.append(
                        {
                            "name": "synthetic_landscape_render_consensus",
                            "strength": round(landscape_strength, 6),
                            "probability": round(landscape_probability, 6),
                        }
                    )

            architecture_strength = min(
                clip01((architecture_score - 0.70) / 0.20),
                clip01((pixel_ensemble - 0.52) / 0.12),
                clip01((forensic_ensemble - 0.42) / 0.09),
                clip01((fft_score - 0.70) / 0.12),
            )
            if architecture_strength >= 0.22:
                architecture_probability = clip01(0.78 + 0.18 * architecture_strength)
                if architecture_probability > final_probability:
                    final_probability = architecture_probability
                    adjustments.append(
                        {
                            "name": "synthetic_architecture_render_consensus",
                            "strength": round(architecture_strength, 6),
                            "probability": round(architecture_probability, 6),
                        }
                    )

            cross_stream_strength = min(
                clip01((pixel_ensemble - 0.53) / 0.12),
                clip01((semantic_score - 0.55) / 0.22),
                clip01((forensic_ensemble - 0.43) / 0.10),
                clip01((fft_score - 0.70) / 0.15),
            )
            if cross_stream_strength >= 0.22:
                consensus_probability = clip01(0.76 + 0.16 * cross_stream_strength)
                if consensus_probability > final_probability:
                    final_probability = consensus_probability
                    adjustments.append(
                        {
                            "name": "moderate_multi_stream_consensus",
                            "strength": round(cross_stream_strength, 6),
                            "probability": round(consensus_probability, 6),
                        }
                    )

            diffusion_strength = min(
                clip01((diffusion_score - 0.55) / 0.30),
                clip01((fft_score - 0.60) / 0.24),
                clip01((forensic_ensemble - 0.38) / 0.20),
            )
            if diffusion_strength >= 0.22:
                diffusion_probability = clip01(0.70 + 0.18 * diffusion_strength)
                if diffusion_probability > final_probability:
                    final_probability = diffusion_probability
                    adjustments.append(
                        {
                            "name": "diffusion_trace_consensus",
                            "strength": round(diffusion_strength, 6),
                            "probability": round(diffusion_probability, 6),
                        }
                    )

        self.last_profile_details = {
            "base_weighted_probability": round(base_probability, 6),
            "final_weighted_probability": round(final_probability, 6),
            "adjustments": adjustments,
        }
        return clip01(final_probability)

    def _feature_value(
        self,
        vector: np.ndarray,
        name: str,
        fallback_index: int | None = None,
    ) -> float:
        if name in self.feature_names:
            index = self.feature_names.index(name)
            if index < len(vector):
                return float(vector[index])
        if fallback_index is not None and fallback_index < len(vector):
            return float(vector[fallback_index])
        return 0.0

    @staticmethod
    def _default_weights(length: int) -> np.ndarray:
        if length <= 0:
            return np.asarray([], dtype=np.float32)
        profile = np.asarray(
            [
                0.55,
                0.52,
                0.53,
                0.72,
                0.34,
                0.39,
                0.40,
                0.30,
                0.55,
                0.58,
                0.50,
                0.46,
                0.44,
                0.52,
                0.50,
            ],
            dtype=np.float32,
        )
        if length <= len(profile):
            base = profile[:length]
        else:
            tail = np.full(length - len(profile), 0.35, dtype=np.float32)
            base = np.concatenate([profile, tail])
        base = base / max(float(np.sum(base)), 1e-8)
        return base * 4.0
