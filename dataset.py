from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from utils.image_io import IMAGE_EXTENSIONS, create_demo_image, load_image


REAL_LABELS = {"real", "authentic", "photo", "photograph", "human", "natural", "0"}
FAKE_LABELS = {
    "fake",
    "ai",
    "generated",
    "synthetic",
    "deepfake",
    "gan",
    "diffusion",
    "1",
}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int

    @property
    def label_name(self) -> str:
        return "FAKE" if self.label == 1 else "REAL"


class DirectoryImageDataset:
    """Recursive dataset loader for CIFAKE-style and generic image folders."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.samples = self._scan()

    def _scan(self) -> list[ImageSample]:
        if not self.root.exists():
            return []
        samples: list[ImageSample] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = infer_label_from_path(path)
            if label is not None:
                samples.append(ImageSample(path=path, label=label))
        return sorted(samples, key=lambda item: str(item.path).lower())

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self.samples)

    def load(self, sample: ImageSample) -> Image.Image:
        return load_image(sample.path)

    def labels(self) -> list[int]:
        return [sample.label for sample in self.samples]

    def limited(self, limit: int | None) -> list[ImageSample]:
        if limit is None or limit <= 0:
            return list(self.samples)
        return list(self.samples[:limit])


def infer_label_from_path(path: Path) -> int | None:
    parts = [part.lower() for part in path.parts]
    for part in reversed(parts):
        tokens = {token for token in part.replace("-", "_").split("_") if token}
        if tokens & FAKE_LABELS:
            return 1
        if tokens & REAL_LABELS:
            return 0
    return None


def ensure_mock_dataset(root: str | Path, per_class: int = 4) -> Path:
    """Create a tiny labeled dataset when real data is unavailable."""

    root = Path(root)
    real_dir = root / "mock" / "real"
    fake_dir = root / "mock" / "fake"
    for idx in range(per_class):
        create_demo_image(real_dir / f"real_{idx}.png", mode="real", size=96 + idx * 4)
        create_demo_image(fake_dir / f"fake_{idx}.png", mode="fake", size=96 + idx * 4)
    return root / "mock"
