import pytest

import torch

from model.scripts.extract_embeddings import _parse_conditions, _stable_subsample, _write_shard


def test_training_rejects_held_out_condition() -> None:
    with pytest.raises(ValueError, match="forbidden for training"):
        _parse_conditions("clean,heldout_jpeg_q20", training=True)


def test_validation_subsample_is_stable_and_keeps_variants() -> None:
    rows = [
        {"source_id": f"source-{source}", "condition_id": condition}
        for source in range(10)
        for condition in ("clean", "jpeg_q30")
    ]
    first = _stable_subsample(rows, 3, 42)
    second = _stable_subsample(list(reversed(rows)), 3, 42)
    first_sources = {row["source_id"] for row in first}
    second_sources = {row["source_id"] for row in second}
    assert first_sources == second_sources
    assert len(first_sources) == 3
    assert len(first) == 6


def test_shard_writer_accepts_nullable_tampered_label(tmp_path) -> None:
    row = {"source_id": "tampered", "official_split": "test", "role": "exploratory_tampered", "dataset": "sid", "binary_label": None, "condition_id": "clean", "seed": 42}
    embedding = torch.zeros((1, 768), dtype=torch.float16)
    embedding[0, 0] = 1
    _write_shard(tmp_path, 0, embedding, [row])
