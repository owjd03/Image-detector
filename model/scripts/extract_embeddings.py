"""Resumable, disk-backed CLIP embedding extraction.

Each completed shard consists of a pickle-free safetensors tensor and a Parquet
sidecar.  An interrupted, partially written shard is ignored on the next run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import gc
import threading
import time
from typing import Any, Iterable

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import load_file, save_file
import torch
import yaml

from model.src.clip_model import image_embeddings, load_frozen_clip
from model.src.data import canonical_rgb
from model.src.logging_utils import configure_logging
from model.src.transforms import (
    GRADED_CONDITIONS,
    HELD_OUT_CONDITIONS,
    apply_condition,
)


LOGGER = logging.getLogger("extract_embeddings")
ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "model" / "outputs" / "manifests"
DEFAULT_OUTPUT = ROOT / "model" / "outputs" / "embeddings"
ALL_CONDITIONS = {"clean", *GRADED_CONDITIONS, *HELD_OUT_CONDITIONS}
TRANSFORM_VERSION = "stage04-v1"
SID_TABLE_PATH: Path | None = None
SID_TABLE: pa.Table | None = None
SID_TABLE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExtractionPlan:
    rows: list[dict[str, Any]]
    fingerprint: str
    scope: str


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_subsample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Choose source IDs stably, then retain every selected source's variants."""
    sources = sorted(
        {row["source_id"] for row in rows},
        key=lambda value: sha256(f"{seed}\0{value}".encode()).digest(),
    )[:count]
    selected = set(sources)
    return [row for row in rows if row["source_id"] in selected]


def _parse_conditions(value: str, *, training: bool) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    unknown = result - ALL_CONDITIONS
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    forbidden = result & HELD_OUT_CONDITIONS.keys()
    if training and forbidden:
        raise ValueError(f"Held-out conditions are forbidden for training: {sorted(forbidden)}")
    return result


def build_plan(
    split: str,
    conditions: str,
    subsample: int | None,
    dry_run: int | None,
    seed: int,
    config: dict[str, Any],
) -> ExtractionPlan:
    manifest_path = MANIFESTS / "dataset_manifest.parquet"
    training_path = MANIFESTS / "training_augmentation_plan.parquet"
    evaluation_path = MANIFESTS / "evaluation_descriptors.parquet"
    is_training = split == "train"
    descriptors_path = training_path if is_training else evaluation_path
    rows = pq.read_table(descriptors_path).to_pylist()
    role_by_split = {
        "val": {"calibration"},
        "test": {"internal_final_evaluation"},
        "tampered": {"exploratory_tampered"},
        "cifake": {"cross_dataset_eval"},
        "evaluation": {"calibration", "internal_final_evaluation", "exploratory_tampered", "cross_dataset_eval"},
    }
    if not is_training:
        permitted = role_by_split[split]
        rows = [row for row in rows if row["role"] in permitted]
    selected_conditions = _parse_conditions(conditions, training=is_training)
    if selected_conditions is not None:
        rows = [row for row in rows if row["condition_id"] in selected_conditions]
    if is_training and any(row["condition_id"] in HELD_OUT_CONDITIONS for row in rows):
        raise ValueError("Training plan contains a held-out condition")
    if subsample is not None:
        if subsample < 1:
            raise ValueError("--subsample must be positive")
        rows = _stable_subsample(rows, subsample, seed)
    manifest_rows = pq.read_table(manifest_path, columns=["source_id", "native_locator", "official_split", "role"]).to_pylist()
    sources = {row["source_id"]: row for row in manifest_rows}
    for row in rows:
        source = sources.get(row["source_id"])
        if source is None:
            raise ValueError(f"Descriptor source missing from manifest: {row['source_id']}")
        if source["official_split"] != row["official_split"] or source["role"] != row["role"]:
            raise ValueError(f"Descriptor/manifest split disagreement: {row['source_id']}")
        row["native_locator"] = source["native_locator"]
    # Consecutive reads stay inside one Parquet file. SID files contain one
    # relatively large row group, so this bounds resident image data.
    def storage_order(row: dict[str, Any]) -> tuple[str, int, int]:
        locator = row["native_locator"]
        if "#row=" in locator:
            relative, row_index = locator.rsplit("#row=", 1)
            return relative, int(row_index), int(row["view_index"])
        return locator, 0, int(row["view_index"])
    rows.sort(key=storage_order)
    if dry_run is not None:
        rows = rows[:dry_run]

    fingerprint_payload = {
        "dataset_manifest_sha256": _hash_file(manifest_path),
        "descriptor_plan_sha256": _hash_file(descriptors_path),
        "model_id": config["model"]["model_id"],
        "revision": config["model"]["revision"],
        "processor": "official AutoImageProcessor configuration at pinned revision",
        "transform_registry_version": TRANSFORM_VERSION,
        "seed": seed,
        "scope": split,
        "conditions": conditions,
        "subsample": subsample,
        "dry_run": dry_run,
    }
    fingerprint = sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    return ExtractionPlan(rows, fingerprint, split)


def _read_sid_row(path: Path, row_index: int) -> Image.Image:
    global SID_TABLE_PATH, SID_TABLE
    # Copy bytes while locked so workers cannot retain an evicted Arrow table.
    with SID_TABLE_LOCK:
        if SID_TABLE_PATH != path:
            SID_TABLE = None
            SID_TABLE_PATH = None
            gc.collect()
            pa.default_memory_pool().release_unused()
            SID_TABLE = pq.read_table(path, columns=["image"])
            SID_TABLE_PATH = path
        if SID_TABLE is None:
            raise RuntimeError("SID table failed to load")
        value = SID_TABLE.column("image")[row_index].as_py()
        image_bytes = bytes(value["bytes"])
    with Image.open(BytesIO(image_bytes)) as decoded:
        return canonical_rgb(decoded)


def _load_transformed(row: dict[str, Any]) -> Image.Image:
    locator = row["native_locator"]
    if "#row=" in locator:
        relative, index = locator.rsplit("#row=", 1)
        image = _read_sid_row(ROOT / relative, int(index))
    else:
        with Image.open(ROOT / locator) as decoded:
            image = canonical_rgb(decoded)
    return apply_condition(image, row["condition_id"], seed=int(row["seed"]))


def _completed_keys(directory: Path, fingerprint: str) -> tuple[set[tuple[Any, ...]], int]:
    metadata_path = directory / "cache.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing["fingerprint"] != fingerprint:
            raise ValueError("Existing embedding cache fingerprint does not match this request")
    keys: set[tuple[Any, ...]] = set()
    next_index = 0
    for sidecar in sorted(directory.glob("shard-*.parquet")):
        tensor_path = sidecar.with_suffix(".safetensors")
        if not tensor_path.exists():
            continue
        metadata = pq.read_table(sidecar).to_pylist()
        tensor = load_file(tensor_path)["embeddings"]
        if tensor.ndim != 2 or tensor.shape[1] != 768 or tensor.shape[0] != len(metadata):
            raise ValueError(f"Invalid completed shard: {sidecar}")
        for row in metadata:
            key = (row["source_id"], row["condition"], int(row["seed"]))
            if key in keys:
                raise ValueError(f"Duplicate cache key: {key}")
            keys.add(key)
        next_index = max(next_index, int(sidecar.stem.split("-")[1]) + 1)
    return keys, next_index


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_shard(directory: Path, shard_index: int, embeddings: torch.Tensor, rows: list[dict[str, Any]]) -> None:
    stem = f"shard-{shard_index:05d}"
    tensor_path = directory / f"{stem}.safetensors"
    sidecar_path = directory / f"{stem}.parquet"
    tensor_tmp = directory / f"{stem}.safetensors.tmp"
    sidecar_tmp = directory / f"{stem}.parquet.tmp"
    if embeddings.shape != (len(rows), 768):
        raise ValueError("Embedding and metadata counts do not align")
    norms = embeddings.float().norm(dim=1)
    if not torch.isfinite(embeddings).all() or not torch.allclose(norms, torch.ones_like(norms), atol=2e-3):
        raise ValueError("Embeddings must be finite and near unit length")
    metadata = []
    for offset, row in enumerate(rows):
        metadata.append({
            "source_id": row["source_id"], "split": row["official_split"],
            "role": row["role"], "dataset": row["dataset"],
            "label": int(row["binary_label"]) if row["binary_label"] is not None else None,
            "condition": row["condition_id"],
            "seed": int(row["seed"]), "shard": shard_index, "row_offset": offset,
        })
    save_file({"embeddings": embeddings.contiguous()}, tensor_tmp)
    pq.write_table(pa.Table.from_pylist(metadata), sidecar_tmp, compression="zstd")
    os.replace(tensor_tmp, tensor_path)
    os.replace(sidecar_tmp, sidecar_path)


def _batches(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def extract(args: argparse.Namespace) -> None:
    config = yaml.safe_load((ROOT / "model" / "configs" / "default.yaml").read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    plan = build_plan(args.split, args.conditions, args.subsample, args.dry_run, seed, config)
    namespace = f"{args.split}_dryrun_{args.dry_run}" if args.dry_run is not None else args.split
    output = Path(args.output) / namespace
    output.mkdir(parents=True, exist_ok=True)
    completed, shard_index = _completed_keys(output, plan.fingerprint)
    remaining = [row for row in plan.rows if (row["source_id"], row["condition_id"], int(row["seed"])) not in completed]
    _atomic_json(output / "cache.json", {"fingerprint": plan.fingerprint, "planned": len(plan.rows), "scope": args.split})

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.batch_size is None:
        if device == "cuda":
            # GPU vendors state capacity in decimal GB (the RTX 5080 reports
            # 16.0 GB, approximately 15.9 GiB), matching the prompt's tiers.
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = 32 if gb >= 16 else 16 if gb >= 12 else 8 if gb >= 8 else 4
        else:
            batch_size = 4
    else:
        batch_size = args.batch_size
    dtype = torch.float16 if device == "cuda" else torch.float32
    estimated_bytes = len(plan.rows) * 768 * 2
    LOGGER.info("plan=%d complete=%d remaining=%d estimated_embedding_mib=%.1f device=%s batch=%d fingerprint=%s", len(plan.rows), len(completed), len(remaining), estimated_bytes / 2**20, device, batch_size, plan.fingerprint)
    if not remaining:
        return

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, processor = load_frozen_clip(
        config["model"]["model_id"], config["model"]["revision"],
        ROOT / config["paths"]["hf_cache_dir"], offline=True, dtype=dtype,
    )
    model.to(device)
    started = time.perf_counter()
    pending_embeddings: list[torch.Tensor] = []
    pending_rows: list[dict[str, Any]] = []
    processed = 0
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for batch_rows in _batches(remaining, batch_size):
            try:
                images = list(executor.map(_load_transformed, batch_rows))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device=device, dtype=dtype)
                with torch.inference_mode():
                    values = image_embeddings(model, pixels).cpu().to(torch.float16)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                raise RuntimeError(f"CUDA OOM at batch {batch_size}; rerun with --batch-size {max(1, batch_size // 2)}. Completed shards are safe.")
            pending_embeddings.append(values)
            pending_rows.extend(batch_rows)
            processed += len(batch_rows)
            while len(pending_rows) >= args.shard_size:
                combined = torch.cat(pending_embeddings)
                _write_shard(output, shard_index, combined[:args.shard_size], pending_rows[:args.shard_size])
                pending_rows = pending_rows[args.shard_size:]
                pending_embeddings = [combined[args.shard_size:]] if len(combined) > args.shard_size else []
                shard_index += 1
            elapsed = time.perf_counter() - started
            rate = processed / elapsed
            eta = (len(remaining) - processed) / rate if rate else math.inf
            LOGGER.info("processed=%d/%d batches=%d images_per_second=%.2f elapsed_s=%.1f eta_s=%.1f", processed, len(remaining), math.ceil(processed / batch_size), rate, elapsed, eta)
        if pending_rows:
            _write_shard(output, shard_index, torch.cat(pending_embeddings), pending_rows)
    finally:
        executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    LOGGER.info("complete=%d duration_s=%.1f images_per_second=%.2f cuda_peak_gib=%.2f", len(remaining), elapsed, len(remaining) / elapsed, peak)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--split", required=True, choices=("train", "val", "test", "tampered", "cifake", "evaluation"))
    result.add_argument("--conditions", default="all")
    result.add_argument("--subsample", type=int)
    result.add_argument("--dry-run", type=int, metavar="N")
    result.add_argument("--batch-size", type=int)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--shard-size", type=int, default=4096)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--verbose", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    configure_logging(verbose=args.verbose)
    extract(args)


if __name__ == "__main__":
    main()
