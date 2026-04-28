from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from forensics import ForensicAnalyzer
from fusion import FeatureAggregator
from meta import MetaClassifier
from pixel_models import PixelEnsemble
from semantic import CLIPSemanticAnalyzer
from utils.dataset import DirectoryImageDataset, ImageSample, ensure_mock_dataset
from utils.image_io import load_image


def load_or_mock_samples(
    data_dir: str | Path,
    mock_if_missing: bool = True,
    limit: int | None = None,
) -> tuple[Path, list[ImageSample]]:
    data_path = Path(data_dir)
    dataset = DirectoryImageDataset(data_path)
    if len(dataset) == 0 and mock_if_missing:
        data_path = ensure_mock_dataset(data_path)
        dataset = DirectoryImageDataset(data_path)
    return data_path, dataset.limited(limit)


def train_pixel_models(
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
    model_names: Sequence[str] | None = None,
    mock_if_missing: bool = True,
    limit: int | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    prefer_torch: bool = True,
) -> list[dict[str, object]]:
    _, samples = load_or_mock_samples(data_dir, mock_if_missing=mock_if_missing, limit=limit)
    ensemble = PixelEnsemble(model_dir=model_dir, prefer_torch=prefer_torch)
    names = list(model_names or ensemble.models.keys())
    results = []
    for name in names:
        result = ensemble.train_model(name, samples, epochs, batch_size, learning_rate)
        results.append(result.__dict__)
    return results


def train_meta_classifier(
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
    mock_if_missing: bool = True,
    limit: int | None = None,
    prefer_torch: bool = True,
    use_clip: bool = False,
) -> dict[str, object]:
    _, samples = load_or_mock_samples(data_dir, mock_if_missing=mock_if_missing, limit=limit)
    aggregator = FeatureAggregator()
    if not samples:
        x, y = generate_mock_feature_matrix(aggregator.feature_names)
    else:
        pixel = PixelEnsemble(model_dir=model_dir, prefer_torch=prefer_torch)
        forensics = ForensicAnalyzer()
        semantic = CLIPSemanticAnalyzer(use_clip=use_clip)
        x_rows = []
        y = []
        for sample in samples:
            image = load_image(sample.path)
            pixel_scores = pixel.analyze(image)
            forensic_scores = forensics.analyze(image)
            semantic_scores = semantic.analyze(image)
            features = aggregator.transform(pixel_scores, forensic_scores, semantic_scores)
            x_rows.append(features.values)
            y.append(sample.label)
        x = np.asarray(x_rows, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

    classifier = MetaClassifier(model_dir=model_dir, feature_names=aggregator.feature_names)
    return classifier.fit(x, y)


def train_all(
    data_dir: str | Path = "data",
    model_dir: str | Path = "models",
    mock_if_missing: bool = True,
    limit: int | None = None,
    prefer_torch: bool = True,
    use_clip: bool = False,
) -> dict[str, object]:
    pixel_results = train_pixel_models(
        data_dir=data_dir,
        model_dir=model_dir,
        mock_if_missing=mock_if_missing,
        limit=limit,
        prefer_torch=prefer_torch,
    )
    meta_result = train_meta_classifier(
        data_dir=data_dir,
        model_dir=model_dir,
        mock_if_missing=mock_if_missing,
        limit=limit,
        prefer_torch=prefer_torch,
        use_clip=use_clip,
    )
    return {"pixel_models": pixel_results, "meta_classifier": meta_result}


def generate_mock_feature_matrix(feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    real = rng.normal(loc=0.28, scale=0.10, size=(24, len(feature_names)))
    fake = rng.normal(loc=0.72, scale=0.12, size=(24, len(feature_names)))
    if len(feature_names) >= 9:
        real[:, 3] = rng.normal(loc=0.25, scale=0.08, size=24)
        fake[:, 3] = rng.normal(loc=0.78, scale=0.10, size=24)
        real[:, 7] = rng.normal(loc=0.32, scale=0.10, size=24)
        fake[:, 7] = rng.normal(loc=0.68, scale=0.12, size=24)
    x = np.clip(np.vstack([real, fake]), 0.0, 1.0).astype(np.float32)
    y = np.asarray([0] * len(real) + [1] * len(fake), dtype=np.int32)
    return x, y
