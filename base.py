from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from utils.dataset import ImageSample
from utils.image_io import image_to_array, load_image
from utils.math_utils import clip01, entropy01, sigmoid
from utils.types import StreamScore


PIXEL_FEATURE_NAMES = [
    "texture_density",
    "smoothness",
    "entropy",
    "saturation",
    "channel_imbalance",
    "edge_uniformity",
    "blockiness",
]


DEFAULT_VARIANT_WEIGHTS = {
    "efficientnet": np.asarray([0.95, 0.75, -0.35, 0.45, 0.35, 0.55, 0.25], dtype=np.float64),
    "alexnet": np.asarray([0.75, 0.35, -0.15, 0.35, 0.25, 0.85, 0.60], dtype=np.float64),
    "googlenet": np.asarray([0.55, 0.85, -0.25, 0.60, 0.55, 0.35, 0.30], dtype=np.float64),
}


@dataclass
class PixelTrainingResult:
    model_name: str
    backend: str
    samples: int
    artifact: str


class PixelModel:
    """Pixel-level binary detector with Torch and lightweight fallback backends."""

    def __init__(
        self,
        model_name: str,
        model_dir: str | Path = "models",
        prefer_torch: bool = True,
    ):
        self.model_name = model_name
        self.stream_name = f"pixel_{model_name}"
        self.model_dir = Path(model_dir)
        self.prefer_torch = prefer_torch
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.heuristic_path = self.model_dir / f"pixel_{model_name}_heuristic.json"
        self.torch_path = self.model_dir / f"pixel_{model_name}.pt"
        self.weights = DEFAULT_VARIANT_WEIGHTS[model_name].copy()
        self.bias = -0.95
        self.backend = "heuristic"
        self._torch_model = None
        self._torch_transform = None
        self._torch = None
        self._load_heuristic()
        if prefer_torch:
            self._try_load_torch()

    def predict(self, image: Image.Image) -> StreamScore:
        if self._torch_model is not None:
            score = self._predict_torch(image)
            details = {"backend": "torchvision", "checkpoint": str(self.torch_path)}
        else:
            features = extract_pixel_features(image)
            score = sigmoid(float(np.dot(self.weights, features) + self.bias))
            details = {
                "backend": self.backend,
                "features": dict(zip(PIXEL_FEATURE_NAMES, features.tolist(), strict=True)),
            }
        return StreamScore(name=self.stream_name, score=clip01(score), details=details)

    def train(
        self,
        samples: list[ImageSample],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ) -> PixelTrainingResult:
        if self.prefer_torch and self._torch_available():
            try:
                return self._train_torch(samples, epochs, batch_size, learning_rate)
            except Exception:
                pass
        return self._train_heuristic(samples)

    def _load_heuristic(self) -> None:
        if not self.heuristic_path.exists():
            return
        with self.heuristic_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("feature_names") == PIXEL_FEATURE_NAMES:
            self.weights = np.asarray(payload["weights"], dtype=np.float64)
            self.bias = float(payload["bias"])
            self.backend = payload.get("backend", "heuristic_trained")

    def _save_heuristic(self) -> None:
        payload = {
            "model_name": self.model_name,
            "backend": self.backend,
            "feature_names": PIXEL_FEATURE_NAMES,
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }
        with self.heuristic_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _train_heuristic(self, samples: list[ImageSample]) -> PixelTrainingResult:
        if not samples:
            self.backend = "heuristic_default"
            self._save_heuristic()
            return PixelTrainingResult(
                self.model_name,
                self.backend,
                0,
                str(self.heuristic_path),
            )

        rows = []
        labels = []
        for sample in samples:
            image = load_image(sample.path)
            rows.append(extract_pixel_features(image))
            labels.append(sample.label)
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        weights = self.weights.copy()
        bias = self.bias
        lr = 0.35
        for _ in range(250):
            logits = x @ weights + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
            error = probs - y
            weights -= lr * (x.T @ error / len(x) + 0.01 * weights)
            bias -= lr * float(np.mean(error))
        self.weights = weights
        self.bias = bias
        self.backend = "heuristic_trained"
        self._save_heuristic()
        return PixelTrainingResult(
            self.model_name,
            self.backend,
            len(samples),
            str(self.heuristic_path),
        )

    def _try_load_torch(self) -> None:
        if not self.torch_path.exists() or not self._torch_available():
            return
        try:
            torch, _models, transforms = self._torch_modules()
            model = self._build_torch_model()
            state = torch.load(self.torch_path, map_location="cpu")
            model.load_state_dict(state["model_state"])
            model.eval()
            self._torch_model = model
            self._torch_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self._torch = torch
            self.backend = "torchvision"
        except Exception:
            self._torch_model = None

    def _predict_torch(self, image: Image.Image) -> float:
        assert self._torch_model is not None
        assert self._torch_transform is not None
        assert self._torch is not None
        tensor = self._torch_transform(image.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            logit = self._torch_model(tensor).reshape(-1)[0]
            return float(self._torch.sigmoid(logit).item())

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _torch_modules():
        import torch
        from torchvision import models, transforms

        return torch, models, transforms

    def _build_torch_model(self):
        torch, models, _transforms = self._torch_modules()
        if self.model_name == "efficientnet":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, 1)
            return model
        if self.model_name == "alexnet":
            model = models.alexnet(weights=None)
            in_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(in_features, 1)
            return model
        if self.model_name == "googlenet":
            model = models.googlenet(weights=None, aux_logits=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 1)
            return model
        raise ValueError(f"Unsupported pixel model: {self.model_name}")

    def _train_torch(
        self,
        samples: list[ImageSample],
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> PixelTrainingResult:
        if not samples:
            return self._train_heuristic(samples)
        torch, _models, transforms = self._torch_modules()
        model = self._build_torch_model()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        dataset = _TorchImageDataset(samples, transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(max(1, epochs)):
            for batch, labels in loader:
                optimizer.zero_grad()
                logits = model(batch).reshape(-1)
                loss = criterion(logits, labels.float())
                loss.backward()
                optimizer.step()
        torch.save(
            {
                "model_name": self.model_name,
                "model_state": model.state_dict(),
                "samples": len(samples),
            },
            self.torch_path,
        )
        self._try_load_torch()
        return PixelTrainingResult(
            self.model_name,
            "torchvision",
            len(samples),
            str(self.torch_path),
        )


class _TorchImageDataset:
    def __init__(self, samples: list[ImageSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.path)
        return self.transform(image), sample.label


def extract_pixel_features(image: Image.Image) -> np.ndarray:
    arr = image_to_array(image, size=(224, 224), grayscale=False)
    gray = np.mean(arr, axis=2)
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    texture_density = clip01(float((np.mean(grad_y) + np.mean(grad_x)) * 8.0))
    smoothness = clip01(1.0 - texture_density)
    entropy = entropy01(gray, bins=64)
    chroma = np.max(arr, axis=2) - np.min(arr, axis=2)
    saturation = clip01(float(np.mean(chroma) * 1.8))
    channel_imbalance = clip01(float(np.std(np.mean(arr, axis=(0, 1))) * 5.0))

    grad_mag = np.zeros_like(gray)
    grad_mag[:-1, :] += grad_y
    grad_mag[:, :-1] += grad_x
    edge_uniformity = clip01(float(1.0 - np.std(grad_mag) / (np.mean(grad_mag) + 1e-6)) * -0.25 + 0.5)

    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    grid_v = float(np.mean(vertical[:, 7::8])) if vertical.shape[1] > 8 else 0.0
    grid_h = float(np.mean(horizontal[7::8, :])) if horizontal.shape[0] > 8 else 0.0
    base = float((np.mean(vertical) + np.mean(horizontal)) / 2.0 + 1e-6)
    blockiness = clip01(((grid_v + grid_h) / 2.0) / (base * 2.0))

    return np.asarray(
        [
            texture_density,
            smoothness,
            entropy,
            saturation,
            channel_imbalance,
            edge_uniformity,
            blockiness,
        ],
        dtype=np.float64,
    )
