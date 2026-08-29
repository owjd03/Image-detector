from pathlib import Path

import pytest

from model.src.config import ConfigError, load_config


def test_default_config_loads() -> None:
    config = load_config()
    assert config.project.seed == 42
    assert config.model.model_id == "openai/clip-vit-large-patch14"
    assert config.model.embedding_dim == 768
    assert config.paths.resources_dir.name == "resources"


def test_path_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_resources = tmp_path / "resources"
    monkeypatch.setenv("ROBUST_DETECTOR_RESOURCES_DIR", str(custom_resources))
    assert load_config().paths.resources_dir == custom_resources


def test_missing_configuration_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(missing)

