"""Typed project configuration with environment-variable path overrides."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "model" / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    resources_dir: Path
    outputs_dir: Path
    hf_cache_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    embedding_dim: int


@dataclass(frozen=True)
class ImageLimitsConfig:
    max_upload_bytes: int
    max_pixels: int


@dataclass(frozen=True)
class SidDatasetConfig:
    dataset_id: str
    revision: str


@dataclass(frozen=True)
class CifakeDatasetConfig:
    dataset_id: str


@dataclass(frozen=True)
class DatasetsConfig:
    sid: SidDatasetConfig
    cifake: CifakeDatasetConfig


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    model: ModelConfig
    image_limits: ImageLimitsConfig
    datasets: DatasetsConfig


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section '{key}' is required and must be a mapping")
    return value


def _required(section: Mapping[str, Any], section_name: str, key: str, expected: type) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required configuration key '{section_name}.{key}'")
    value = section[key]
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ConfigError(
            f"Configuration key '{section_name}.{key}' must be {expected.__name__}"
        )
    if expected is str and not value.strip():
        raise ConfigError(f"Configuration key '{section_name}.{key}' cannot be empty")
    return value


def _resolve_path(value: str, environment_key: str) -> Path:
    selected = os.environ.get(environment_key, value)
    if not selected.strip():
        raise ConfigError(f"Path configured by {environment_key} cannot be empty")
    path = Path(selected).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load, validate, and resolve the central YAML configuration."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPOSITORY_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ConfigError("Configuration root must be a mapping")

    project = _section(raw, "project")
    paths = _section(raw, "paths")
    model = _section(raw, "model")
    limits = _section(raw, "image_limits")
    datasets = _section(raw, "datasets")
    sid = _section(datasets, "sid")
    cifake = _section(datasets, "cifake")

    result = AppConfig(
        project=ProjectConfig(
            name=_required(project, "project", "name", str),
            seed=_required(project, "project", "seed", int),
        ),
        paths=PathsConfig(
            resources_dir=_resolve_path(
                _required(paths, "paths", "resources_dir", str),
                "ROBUST_DETECTOR_RESOURCES_DIR",
            ),
            outputs_dir=_resolve_path(
                _required(paths, "paths", "outputs_dir", str),
                "ROBUST_DETECTOR_OUTPUTS_DIR",
            ),
            hf_cache_dir=_resolve_path(
                _required(paths, "paths", "hf_cache_dir", str),
                "ROBUST_DETECTOR_HF_CACHE_DIR",
            ),
        ),
        model=ModelConfig(
            model_id=_required(model, "model", "model_id", str),
            revision=_required(model, "model", "revision", str),
            embedding_dim=_required(model, "model", "embedding_dim", int),
        ),
        image_limits=ImageLimitsConfig(
            max_upload_bytes=_required(
                limits, "image_limits", "max_upload_bytes", int
            ),
            max_pixels=_required(limits, "image_limits", "max_pixels", int),
        ),
        datasets=DatasetsConfig(
            sid=SidDatasetConfig(
                dataset_id=_required(sid, "datasets.sid", "dataset_id", str),
                revision=_required(sid, "datasets.sid", "revision", str),
            ),
            cifake=CifakeDatasetConfig(
                dataset_id=_required(cifake, "datasets.cifake", "dataset_id", str),
            ),
        ),
    )
    if result.project.seed < 0:
        raise ConfigError("project.seed must be non-negative")
    if result.model.embedding_dim != 768:
        raise ConfigError("CLIP ViT-L/14 embedding_dim must be 768")
    if result.image_limits.max_upload_bytes <= 0 or result.image_limits.max_pixels <= 0:
        raise ConfigError("Image limits must be positive")
    return result

