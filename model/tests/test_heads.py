import torch

from model.src.heads import LinearHead, MLPHead, better_checkpoint, consistency_loss, grouped_indices


def test_heads_forward_and_gradient() -> None:
    for model in (LinearHead(), MLPHead()):
        x = torch.randn(4, 768)
        output = model(x)
        assert output.shape == (4,)
        output.sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_grouping_keeps_clean_and_variants_together() -> None:
    sources = ["b", "a", "a", "b", "a", "b"]
    conditions = ["jpeg_q30", "blur_s1.0", "clean", "clean", "noise_s0.02", "resize_0.5"]
    groups = grouped_indices(sources, conditions)
    assert groups == [[2, 1, 4], [3, 0, 5]]


def test_consistency_loss_zero_positive_and_single_view() -> None:
    assert consistency_loss(torch.tensor([1.0, 1.0, 1.0]), [0, 0, 0], [True, False, False]).item() == 0
    assert consistency_loss(torch.tensor([1.0, 2.0, 3.0]), [0, 0, 0], [True, False, False]).item() > 0
    assert consistency_loss(torch.tensor([1.0]), [0], [True]).item() == 0


def test_checkpoint_selection_uses_loss_then_tie_break() -> None:
    assert better_checkpoint(0.4, 0.5, 0.5, 0.9)
    assert better_checkpoint(0.50005, 0.8, 0.5, 0.7)
    assert not better_checkpoint(0.50005, 0.6, 0.5, 0.7)
