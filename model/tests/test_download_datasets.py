from pathlib import Path

import pyarrow.parquet as pq
import pytest

from model.scripts.download_datasets import _materialize_role, _selected_row


def _row(identifier: str, label: int) -> dict:
    return {"img_id": identifier, "image": {"bytes": b"image", "path": None}, "mask": None, "width": 1, "height": 1, "label": label}


def test_selected_row_preserves_official_split() -> None:
    result = _selected_row(_row("x", 1), "validation", "calibration")
    assert result["official_split"] == "validation"
    assert result["role"] == "calibration"


def test_materialization_is_stratified_and_resumable(tmp_path: Path) -> None:
    stream = [_row(f"real-{i}", 0) for i in range(3)] + [_row(f"fake-{i}", 1) for i in range(3)]
    counts = _materialize_role(stream, tmp_path, "train", "train", {0: 2, 1: 2}, resume=False, shard_size=2)
    assert counts == {0: 2, 1: 2}
    resumed = _materialize_role(stream, tmp_path, "train", "train", {0: 2, 1: 2}, resume=True, shard_size=2)
    assert resumed == {0: 2, 1: 2}
    assert sum(pq.read_metadata(path).num_rows for path in tmp_path.glob("*.parquet")) == 4


def test_partial_data_requires_resume(tmp_path: Path) -> None:
    _materialize_role([_row("x", 0)], tmp_path, "train", "train", {0: 1}, resume=False)
    with pytest.raises(RuntimeError, match="--resume"):
        _materialize_role([_row("x", 0)], tmp_path, "train", "train", {0: 1}, resume=False)


def test_missing_sid_column_is_rejected() -> None:
    row = _row("x", 0)
    del row["width"]
    with pytest.raises(ValueError, match="missing required columns"):
        _selected_row(row, "train", "train")
