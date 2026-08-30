import numpy as np
import torch

from model.src.evaluation import binary_metrics, choose_threshold, fit_temperature, fixed_fpr_tpr


def test_temperature_positive_and_non_worsening() -> None:
    logits = torch.tensor([-0.2, 0.2, -0.3, 0.3])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    temperature, before, after = fit_temperature(logits, labels)
    assert temperature > 0
    assert after <= before + 1e-8


def test_threshold_and_metrics_known_fixture() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    selected = choose_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, selected["threshold"])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_fixed_fpr_known_fixture() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    assert fixed_fpr_tpr(labels, probabilities, 0.01) == 1.0
