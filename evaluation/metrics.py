from __future__ import annotations

from typing import Sequence

import numpy as np

from calibration.calibrators import expected_calibration_error
from utils.math_utils import clip01


def evaluate_binary_predictions(
    probabilities: Sequence[float],
    labels: Sequence[int],
    ood_scores: Sequence[float] | None = None,
) -> dict[str, float]:
    probs = np.asarray([clip01(value) for value in probabilities], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    if probs.size == 0:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "ood_accuracy": 0.0,
        }
    return {
        "auroc": round(_auroc(probs, y), 6),
        "auprc": round(_auprc(probs, y), 6),
        "ece": round(expected_calibration_error(probs, y), 6),
        "brier": round(float(np.mean((probs - y) ** 2)), 6),
        "ood_accuracy": round(_ood_accuracy(ood_scores, y), 6),
        "compression_robustness": 0.0,
        "adversarial_robustness": 0.0,
        "unseen_generator_performance": 0.0,
        "cross_domain_generalization": 0.0,
    }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.0
    wins = 0.0
    for pos in positives:
        wins += float(np.sum(pos > negatives))
        wins += 0.5 * float(np.sum(pos == negatives))
    return clip01(wins / float(positives.size * negatives.size))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = max(int(np.sum(sorted_labels == 1)), 1)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / positives)
    area = 0.0
    last_recall = 0.0
    for precision, recall in zip(precisions, recalls, strict=True):
        area += precision * max(0.0, recall - last_recall)
        last_recall = recall
    return clip01(area)


def _ood_accuracy(ood_scores: Sequence[float] | None, labels: np.ndarray) -> float:
    if ood_scores is None:
        return 0.0
    scores = np.asarray([clip01(value) for value in ood_scores], dtype=np.float64)
    if scores.size != labels.size or scores.size == 0:
        return 0.0
    unknown = labels < 0
    if not np.any(unknown):
        return 0.0
    return float(np.mean((scores >= 0.72) == unknown))
