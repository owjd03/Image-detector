import random

import numpy as np
import torch

from model.src.reproducibility import seed_everything


def _sample() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_repeats_random_streams() -> None:
    seed_everything(42)
    first = _sample()
    seed_everything(42)
    assert _sample() == first


def test_negative_seed_is_rejected() -> None:
    try:
        seed_everything(-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("A negative seed should fail")

