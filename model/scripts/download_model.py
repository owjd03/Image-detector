"""Resolve, download, and verify the pinned CLIP ViT-L/14 vision model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from huggingface_hub import HfApi
from PIL import Image
import torch
import transformers
import yaml

from model.src.clip_model import (
    PLACEHOLDER_REVISION,
    image_embeddings,
    load_frozen_clip,
    validate_pinned_revision,
)
from model.src.config import DEFAULT_CONFIG_PATH, load_config
from model.src.device import select_device
from model.src.reproducibility import seed_everything


def resolve_revision(model_id: str, requested_revision: str | None = None) -> str:
    """Resolve a branch/tag to the immutable commit currently backing it."""

    info = HfApi().model_info(model_id, revision=requested_revision or "main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no commit hash for {model_id}")
    return validate_pinned_revision(info.sha)


def update_config_revision(config_path: Path, revision: str) -> None:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["model"]["revision"] = validate_pinned_revision(revision)
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _portable_cache_path(cache_dir: Path) -> str:
    try:
        return cache_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return "<HF_CACHE>/" + cache_dir.name


def build_manifest(
    *, config: Any, revision: str, model: Any, processor: Any,
    peak_memory_bytes: int | None, images_per_second: float | None,
) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_id": config.model.model_id,
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "embedding_dimension": config.model.embedding_dim,
        "parameter_counts": {"vision_model": total_parameters},
        "processor": processor.to_dict() if hasattr(processor, "to_dict") else {},
        "cache_location": _portable_cache_path(config.paths.hf_cache_dir),
        "under_2_billion_parameters": total_parameters < 2_000_000_000,
        "cuda_benchmark": {
            "peak_allocated_memory_bytes": peak_memory_bytes,
            "images_per_second": images_per_second,
        },
    }


def smoke_inference(model: Any, processor: Any, device: torch.device, dimension: int) -> tuple[int | None, float | None]:
    seed_everything(42)
    image = Image.new("RGB", (224, 224), color=(23, 101, 177))
    inputs = processor(images=[image, image], return_tensors="pt")
    model.to(device)
    pixel_values = inputs["pixel_values"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    precision = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    started = time.perf_counter()
    with torch.inference_mode(), precision:
        first = image_embeddings(model, pixel_values)
        second = image_embeddings(model, pixel_values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if first.shape != (2, dimension):
        raise RuntimeError(f"Unexpected embedding shape {tuple(first.shape)}")
    if not torch.isfinite(first).all() or not torch.allclose(first.norm(dim=-1), torch.ones(2, device=device), atol=1e-4):
        raise RuntimeError("Embeddings must be finite and unit length")
    if not torch.allclose(first, second, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Repeated inference was not deterministic")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CLIP parameters were not fully frozen")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    throughput = 4 / elapsed if device.type == "cuda" else None
    return peak, throughput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--cache-dir")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else config.paths.hf_cache_dir
        revision = config.model.revision
        if not args.verify_only:
            revision = resolve_revision(config.model.model_id, None if revision == PLACEHOLDER_REVISION else revision)
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            validate_pinned_revision(revision)
        model, processor = load_frozen_clip(
            config.model.model_id, revision, cache_dir, offline=args.offline or args.verify_only,
            dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        device = select_device("auto")
        peak, throughput = smoke_inference(model, processor, device, config.model.embedding_dim)
        if not args.verify_only:
            update_config_revision(config_path, revision)
            manifest = build_manifest(
                config=config, revision=revision, model=model, processor=processor,
                peak_memory_bytes=peak, images_per_second=throughput,
            )
            manifest_path = config.paths.outputs_dir / "manifests" / "model_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(manifest_path)
        print(json.dumps({"model_id": config.model.model_id, "revision": revision, "device": str(device), "peak_memory_bytes": peak, "images_per_second": throughput}, indent=2))
        return 0
    except Exception as error:
        print(f"Model setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
