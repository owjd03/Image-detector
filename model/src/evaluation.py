"""Calibration, threshold selection, and binary evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
    roc_curve,
)
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    logits, labels = logits.detach().double(), labels.detach().double()
    before = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    log_temperature = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach())
    after = float(torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels))
    if not np.isfinite(temperature) or temperature <= 0 or after > before + 1e-8:
        raise ValueError("Temperature calibration failed to improve or preserve NLL")
    return temperature, before, after


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    balanced = (tpr + 1.0 - fpr) / 2.0
    best = max(range(len(thresholds)), key=lambda index: (balanced[index], -fpr[index], -abs(thresholds[index] - 0.5)))
    threshold = float(np.clip(thresholds[best], 0.0, 1.0))
    return {"threshold": threshold, "balanced_accuracy": float(balanced[best]), "authentic_false_positive_rate": float(fpr[best])}


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels); value = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(value)


def fixed_fpr_tpr(labels: np.ndarray, probabilities: np.ndarray, target: float) -> float | None:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    eligible = np.where(fpr <= target)[0]
    return float(tpr[eligible].max()) if len(eligible) else None


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        return {"count": int(len(labels)), "class_counts": {str(int(value)): int((labels == value).sum()) for value in np.unique(labels)}, "accuracy": None, "balanced_accuracy": None, "precision": None, "recall": None, "f1": None, "roc_auc": None, "pr_auc": None, "authentic_false_positive_rate": None, "confusion_matrix": None, "brier_score": None, "ece_10_bin": None, "tpr_at_1pct_fpr": None, "tpr_at_5pct_fpr": None, "warning": "metric requires both classes"}
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "count": int(len(labels)), "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "accuracy": float(accuracy_score(labels, predictions)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)), "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)), "authentic_false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": matrix.tolist(), "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_10_bin": expected_calibration_error(labels, probabilities), "tpr_at_1pct_fpr": fixed_fpr_tpr(labels, probabilities, 0.01),
        "tpr_at_5pct_fpr": fixed_fpr_tpr(labels, probabilities, 0.05),
    }
