import pytest
import torch

from model.src.clip_model import normalize_embeddings, validate_pinned_revision


def test_revision_must_be_full_commit_hash() -> None:
    revision = "a" * 40
    assert validate_pinned_revision(revision) == revision
    for invalid in ("main", "abc123", "g" * 40):
        with pytest.raises(ValueError, match="immutable commit hash"):
            validate_pinned_revision(invalid)


def test_embedding_normalization() -> None:
    result = normalize_embeddings(torch.tensor([[3.0, 4.0]]))
    assert result.shape == (1, 2)
    assert torch.allclose(result.norm(dim=-1), torch.ones(1))
