"""Acquire, resume, verify, and inventory project datasets without split leakage."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

from PIL import Image as PilImage
import pyarrow as pa
import pyarrow.parquet as pq

from model.src.config import DEFAULT_CONFIG_PATH, AppConfig, load_config


SID_COLUMNS = {"img_id", "image", "mask", "width", "height", "label"}
SID_LABELS = {0: "real", 1: "full_synthetic", 2: "tampered"}
TIER_A_DEFAULTS = {
    "train_per_class": 5_000,
    "calibration_per_class": 1_000,
    "evaluation_per_class": 1_000,
    "tampered_limit": 250,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _image_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {"bytes": value.get("bytes"), "path": value.get("path")}
    raise TypeError(f"Unexpected Hugging Face image payload: {type(value).__name__}")


def _selected_row(row: dict[str, Any], official_split: str, role: str) -> dict[str, Any]:
    missing = SID_COLUMNS.difference(row)
    if missing:
        raise ValueError(f"SID row is missing required columns: {sorted(missing)}")
    label = int(row["label"])
    if label not in SID_LABELS:
        raise ValueError(f"Unexpected SID label: {label}")
    return {
        "img_id": str(row["img_id"]),
        "image": _image_payload(row["image"]),
        "mask": _image_payload(row["mask"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "label": label,
        "official_split": official_split,
        "role": role,
    }


def _write_shard(directory: Path, index: int, rows: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"part-{index:05d}.parquet"
    temporary = target.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError(f"Shard validation failed: {temporary}")
    temporary.replace(target)
    print(
        f"saved role={directory.name} shard={target.name} rows={len(rows)} "
        f"bytes={target.stat().st_size}",
        flush=True,
    )
    return target


def _existing_ids(directory: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted(directory.glob("part-*.parquet")):
        result.update(pq.read_table(path, columns=["img_id"])["img_id"].to_pylist())
    return result


def _materialize_role(
    stream: Iterable[dict[str, Any]], directory: Path, official_split: str,
    role: str, targets: dict[int, int], *, resume: bool, shard_size: int = 500,
) -> dict[int, int]:
    existing = _existing_ids(directory) if resume else set()
    if any(directory.glob("part-*.parquet")) and not resume:
        raise RuntimeError(f"Partial data exists at {directory}; rerun with --resume")
    counts = Counter()
    if existing:
        for path in directory.glob("part-*.parquet"):
            counts.update(pq.read_table(path, columns=["label"])["label"].to_pylist())
    shard_index = len(list(directory.glob("part-*.parquet")))
    pending: list[dict[str, Any]] = []
    for raw in stream:
        label = int(raw["label"])
        if label not in targets or counts[label] >= targets[label] or str(raw["img_id"]) in existing:
            continue
        pending.append(_selected_row(raw, official_split, role))
        counts[label] += 1
        if len(pending) >= shard_size:
            _write_shard(directory, shard_index, pending)
            shard_index += 1
            pending = []
        if all(counts[label] >= target for label, target in targets.items()):
            break
    if pending:
        _write_shard(directory, shard_index, pending)
    missing = {label: target - counts[label] for label, target in targets.items() if counts[label] < target}
    if missing:
        raise RuntimeError(f"SID stream ended before quotas were met for {role}: {missing}")
    return dict(counts)


def _sid_stream(config: AppConfig, split: str, seed: int, buffer_size: int):
    from datasets import Image, load_dataset

    stream = load_dataset(
        config.datasets.sid.dataset_id,
        split=split,
        revision=config.datasets.sid.revision,
        streaming=True,
        cache_dir=str(config.paths.resources_dir / "datasets" / "_hf_cache"),
    )
    stream = stream.cast_column("image", Image(decode=False))
    stream = stream.cast_column("mask", Image(decode=False))
    return stream.shuffle(seed=seed, buffer_size=buffer_size)


def acquire_sid(config: AppConfig, args: argparse.Namespace) -> dict[str, Any]:
    if _free_bytes(config.paths.resources_dir) < 30 * 1024**3:
        raise RuntimeError("Tier A requires at least 30 GiB free on the resources volume")
    root = config.paths.resources_dir / "datasets" / "sid_set" / "tier_a"
    print(
        f"starting SID_Set Tier A revision={config.datasets.sid.revision} "
        f"destination={root}",
        flush=True,
    )
    train_counts = _materialize_role(
        _sid_stream(config, "train", config.project.seed, args.shuffle_buffer), root / "train", "train", "train",
        {0: args.train_per_class, 1: args.train_per_class}, resume=args.resume,
    )
    # Both report roles remain children of the official validation split.
    validation = _sid_stream(config, "validation", config.project.seed, args.shuffle_buffer)
    calibration_counts = _materialize_role(
        validation, root / "calibration", "validation", "calibration",
        {0: args.calibration_per_class, 1: args.calibration_per_class}, resume=args.resume,
    )
    reserved = _existing_ids(root / "calibration")
    evaluation_stream = (row for row in _sid_stream(config, "validation", config.project.seed, args.shuffle_buffer) if str(row["img_id"]) not in reserved)
    evaluation_counts = _materialize_role(
        evaluation_stream, root / "internal_final_evaluation", "validation", "internal_final_evaluation",
        {0: args.evaluation_per_class, 1: args.evaluation_per_class}, resume=args.resume,
    )
    binary_reserved = reserved | _existing_ids(root / "internal_final_evaluation")
    tampered_stream = (row for row in _sid_stream(config, "validation", config.project.seed, args.shuffle_buffer) if str(row["img_id"]) not in binary_reserved)
    tampered_counts = _materialize_role(
        tampered_stream, root / "exploratory_tampered", "validation", "exploratory_tampered",
        {2: args.tampered_limit}, resume=args.resume,
    )
    id_groups = (
        _existing_ids(root / role)
        for role in ("train", "calibration", "internal_final_evaluation", "exploratory_tampered")
    )
    flat_ids = sorted(identifier for group in id_groups for identifier in group)
    return {
        "name": "SID_Set", "source_url": "https://huggingface.co/datasets/saberzl/SID_Set",
        "revision": config.datasets.sid.revision, "license": "CC BY 4.0",
        "local_location": "resources/datasets/sid_set/tier_a", "retrieval_status": "complete",
        "intended_role": "core_train_calibration_internal_evaluation_and_tampered_diagnostic",
        "official_test_available": False,
        "counts": {"train": train_counts, "calibration": calibration_counts, "internal_final_evaluation": evaluation_counts, "exploratory_tampered": tampered_counts},
        "selection_sha256": hashlib.sha256("\n".join(flat_ids).encode()).hexdigest(),
    }


def acquire_cifake(config: AppConfig) -> dict[str, Any]:
    target = config.paths.resources_dir / "datasets" / "cifake"
    local = _inspect_cifake(target)
    if local["complete"]:
        return {
            "name": "CIFAKE", "source_url": f"https://www.kaggle.com/datasets/{config.datasets.cifake.dataset_id}",
            "revision": "manually downloaded Kaggle version; archive revision unavailable",
            "license": "MIT (per dataset card)",
            "local_location": "resources/datasets/cifake", "retrieval_status": "complete_manual_download",
            "intended_role": "cross_dataset_eval_only", "counts": local["counts"],
            "sample_dimensions": local["sample_dimensions"],
        }
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError("Install the pinned Kaggle API dependency first") from error
    target.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        raise RuntimeError("Kaggle credentials are missing or invalid; configure ~/.kaggle/kaggle.json") from error
    api.dataset_download_files(config.datasets.cifake.dataset_id, path=str(target), unzip=True, quiet=False)
    local = _inspect_cifake(target)
    if not local["complete"]:
        raise RuntimeError(f"CIFAKE download completed but validation failed: {local}")
    return {
        "name": "CIFAKE", "source_url": f"https://www.kaggle.com/datasets/{config.datasets.cifake.dataset_id}",
        "revision": "Kaggle current version at retrieval", "license": "MIT (per dataset card)",
        "local_location": "resources/datasets/cifake", "retrieval_status": "complete",
        "intended_role": "cross_dataset_eval_only", "counts": local["counts"],
        "sample_dimensions": local["sample_dimensions"],
    }


def _inspect_cifake(root: Path) -> dict[str, Any]:
    expected = {("train", "REAL"): 50_000, ("train", "FAKE"): 50_000, ("test", "REAL"): 10_000, ("test", "FAKE"): 10_000}
    counts: dict[str, int] = {}
    sample_dimensions: dict[str, list[int] | None] = {}
    complete = True
    for (split, label), target in expected.items():
        directory = root / split / label
        files = sorted(path for path in directory.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if directory.is_dir() else []
        key = f"{split}/{label}"
        counts[key] = len(files)
        complete = complete and len(files) == target
        if files:
            try:
                with PilImage.open(files[0]) as image:
                    image.load()
                    sample_dimensions[key] = list(image.size)
            except OSError:
                complete = False
                sample_dimensions[key] = None
        else:
            sample_dimensions[key] = None
    return {"complete": complete, "counts": counts, "sample_dimensions": sample_dimensions}


def wildfake_record(config: AppConfig) -> dict[str, Any]:
    return {
        "name": "WildFake challenge subset", "source_url": "https://modelscope.cn/datasets/hy2628982280/WildFake/summary",
        "revision": None, "license": "not evaluated because dataset was omitted",
        "local_location": None, "retrieval_status": "omitted_by_plan",
        "intended_role": "none; external demonstration deferred",
    }


def _dataset_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def verify(config: AppConfig, selected: str) -> dict[str, Any]:
    root = config.paths.resources_dir / "datasets"
    result: dict[str, Any] = {}
    if selected in {"sid", "all"}:
        sid_root = root / "sid_set" / "tier_a"
        result["sid"] = {role: len(_existing_ids(sid_root / role)) for role in ("train", "calibration", "internal_final_evaluation", "exploratory_tampered")}
    if selected in {"cifake", "all"}:
        result["cifake"] = _inspect_cifake(root / "cifake")
    if selected in {"wildfake", "all"}:
        result["wildfake"] = {"status": "omitted_by_plan"}
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", choices=("cifake", "sid", "wildfake", "all"), default="all")
    parser.add_argument("--tier", choices=("a", "b", "c"), default="a")
    parser.add_argument("--train-per-class", type=int, default=TIER_A_DEFAULTS["train_per_class"])
    parser.add_argument("--calibration-per-class", type=int, default=TIER_A_DEFAULTS["calibration_per_class"])
    parser.add_argument("--evaluation-per-class", type=int, default=TIER_A_DEFAULTS["evaluation_per_class"])
    parser.add_argument("--tampered-limit", type=int, default=TIER_A_DEFAULTS["tampered_limit"])
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.tier != "a":
            raise ValueError("This deadline-locked implementation currently permits Tier A only")
        if min(args.train_per_class, args.calibration_per_class, args.evaluation_per_class, args.tampered_limit, args.shuffle_buffer) <= 0:
            raise ValueError("All quotas and the shuffle buffer must be positive")
        config = load_config(args.config)
        if args.verify_only:
            print(json.dumps(verify(config, args.dataset), indent=2, sort_keys=True))
            return 0
        records = []
        if args.dataset in {"sid", "all"}:
            records.append(acquire_sid(config, args))
        if args.dataset in {"cifake", "all"}:
            records.append(acquire_cifake(config))
        if args.dataset in {"wildfake", "all"}:
            records.append(wildfake_record(config))
        inventory_path = config.paths.resources_dir / "datasets" / "dataset_inventory.json"
        previous_records: list[dict[str, Any]] = []
        if inventory_path.is_file():
            previous = json.loads(inventory_path.read_text(encoding="utf-8"))
            previous_records = previous.get("datasets", [])
        replaced_names = {record["name"] for record in records}
        records = [record for record in previous_records if record.get("name") not in replaced_names] + records
        inventory = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selected_tier": "A",
            "sampling_seed": config.project.seed,
            "quotas": {key: getattr(args, key) for key in TIER_A_DEFAULTS}, "datasets": records,
        }
        datasets_root = config.paths.resources_dir / "datasets"
        for record in records:
            local_location = record.get("local_location")
            record["byte_total"] = (
                _dataset_bytes(config.paths.resources_dir.parent / local_location)
                if local_location
                else 0
            )
        _atomic_json(inventory_path, inventory)
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"Dataset setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
