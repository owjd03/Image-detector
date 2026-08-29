"""Read-only environment diagnostics; this command never downloads assets."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
import shutil
from pathlib import Path
from typing import Any

from model.src.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def collect_diagnostics(config: AppConfig) -> dict[str, Any]:
    import torch

    disk = shutil.disk_usage(config.paths.resources_dir)
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "vram_bytes": properties.total_memory,
        }

    model_cache_name = "models--" + config.model.model_id.replace("/", "--")
    dataset_root = config.paths.resources_dir / "datasets"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "transformers": _version("transformers"),
        },
        "compute": {
            "cuda_available": cuda_available,
            "mps_available": torch.backends.mps.is_available(),
            "gpu": gpu,
        },
        "disk": {
            "resources_volume_total_bytes": disk.total,
            "resources_volume_free_bytes": disk.free,
        },
        "paths": {
            "resources_dir": _display_path(config.paths.resources_dir),
            "outputs_dir": _display_path(config.paths.outputs_dir),
            "hf_cache_dir": _display_path(config.paths.hf_cache_dir),
        },
        "assets": {
            "model_cached": (config.paths.hf_cache_dir / "hub" / model_cache_name).exists()
            or (config.paths.hf_cache_dir / model_cache_name).exists(),
            "cifake_present": (dataset_root / "cifake").exists(),
            "sid_set_present": (dataset_root / "sid_set").exists(),
            "wildfake_present": (dataset_root / "wildfake_validation").exists(),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        diagnostics = collect_diagnostics(load_config(arguments.config))
    except (ConfigError, OSError) as error:
        print(f"Configuration error: {error}")
        return 2
    if arguments.as_json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
    else:
        print("Robust AI Image Detector — environment diagnostics")
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

