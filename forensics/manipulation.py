from __future__ import annotations

import numpy as np
from PIL import Image

from utils.image_io import image_to_array
from utils.math_utils import clip01, normalize_minmax
from utils.types import StreamScore


class ManipulationAnalyzer:
    """Detects splice/copy/blending style inconsistencies."""

    name = "forensic_manipulation"

    def __init__(self, size: int = 256, grid: int = 16):
        self.size = size
        self.grid = grid

    def analyze(self, image: Image.Image) -> StreamScore:
        arr = image_to_array(image, size=(self.size, self.size), grayscale=False)
        gray = np.mean(arr, axis=2)
        residual = gray - _box_blur(gray)
        grad = _gradient(gray)

        block_size = self.size // self.grid
        noise_blocks = _block_means(np.abs(residual), block_size)
        edge_blocks = _block_means(grad, block_size)
        color_blocks = np.stack(
            [_block_means(arr[:, :, channel], block_size) for channel in range(3)],
            axis=-1,
        )

        noise_inconsistency = float(np.std(noise_blocks) / (np.mean(noise_blocks) + 1e-8))
        edge_noise_mismatch = float(
            np.std(noise_blocks - edge_blocks / (np.max(edge_blocks) + 1e-8))
        )
        color_discontinuity = float(np.mean(np.abs(np.diff(color_blocks, axis=0)))) + float(
            np.mean(np.abs(np.diff(color_blocks, axis=1)))
        )
        copy_move_similarity = _copy_move_proxy(gray, block_size)

        score = clip01(
            0.30 * normalize_minmax(noise_inconsistency, 0.34, 1.10)
            + 0.24 * normalize_minmax(edge_noise_mismatch, 0.05, 0.18)
            + 0.22 * normalize_minmax(color_discontinuity, 0.06, 0.20)
            + 0.24 * normalize_minmax(copy_move_similarity, 0.88, 0.985)
        )
        return StreamScore(
            self.name,
            score,
            {
                "noise_inconsistency": noise_inconsistency,
                "edge_noise_mismatch": edge_noise_mismatch,
                "color_discontinuity": color_discontinuity,
                "copy_move_similarity": copy_move_similarity,
            },
        )


def _box_blur(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _gradient(gray: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(gray, axis=0, append=gray[-1:, :])) + np.abs(
        np.diff(gray, axis=1, append=gray[:, -1:])
    )


def _block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    rows = values.shape[0] // block_size
    cols = values.shape[1] // block_size
    cropped = values[: rows * block_size, : cols * block_size]
    return cropped.reshape(rows, block_size, cols, block_size).mean(axis=(1, 3))


def _copy_move_proxy(gray: np.ndarray, block_size: int) -> float:
    blocks = []
    rows = gray.shape[0] // block_size
    cols = gray.shape[1] // block_size
    for row in range(rows):
        for col in range(cols):
            block = gray[
                row * block_size : (row + 1) * block_size,
                col * block_size : (col + 1) * block_size,
            ]
            normalized = block - float(np.mean(block))
            norm = float(np.linalg.norm(normalized))
            if norm > 1e-8:
                blocks.append((row, col, (normalized / norm).ravel()))
    best = 0.0
    for index, (row, col, block) in enumerate(blocks):
        for other_row, other_col, other in blocks[index + 1 :]:
            if abs(row - other_row) + abs(col - other_col) < 4:
                continue
            best = max(best, float(np.dot(block, other)))
    return clip01(best)
