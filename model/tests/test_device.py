import pytest
import torch

from model.src.device import select_device


def test_cpu_can_be_selected() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, cuda, mps, cpu"):
        select_device("quantum")

