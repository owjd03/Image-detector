from pathlib import Path
from types import SimpleNamespace

import yaml

from model.scripts.download_model import resolve_revision, update_config_revision


def test_revision_resolution_uses_hugging_face_sha(monkeypatch) -> None:
    expected = "1" * 40
    monkeypatch.setattr(
        "model.scripts.download_model.HfApi.model_info",
        lambda self, model_id, revision: SimpleNamespace(sha=expected),
    )
    assert resolve_revision("organization/model") == expected


def test_config_revision_update(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  revision: old\n", encoding="utf-8")
    update_config_revision(path, "2" * 40)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["model"]["revision"] == "2" * 40
