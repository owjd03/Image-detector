"""Manifest hashing, leakage checks, and transformation descriptor planning."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq

from model.src.data import ImageSample, validate_sample_role
from model.src.transforms import GRADED_CONDITIONS, HELD_OUT_CONDITIONS, seed_for_source


MANIFEST_COLUMNS = ("source_id", "dataset", "native_locator", "official_split", "original_label", "binary_label", "role", "width", "height", "sha256", "phash", "source_group_id")


def canonical_sha256(sample: ImageSample) -> str:
    image = sample.image.convert("RGB")
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(width.to_bytes(8, "big"))
    digest.update(height.to_bytes(8, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def manifest_row(sample: ImageSample) -> dict[str, Any]:
    validate_sample_role(sample)
    width, height = sample.image.size
    return {
        "source_id": sample.source_id, "dataset": sample.dataset,
        "native_locator": sample.native_locator, "official_split": sample.official_split,
        "original_label": sample.original_label, "binary_label": sample.binary_label,
        "role": sample.role, "width": width, "height": height,
        "sha256": canonical_sha256(sample), "phash": str(imagehash.phash(sample.image)),
        "source_group_id": sample.source_group_id,
    }


def validate_manifest(rows: list[dict[str, Any]], *, near_threshold: int = 4) -> dict[str, Any]:
    identities: set[str] = set()
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_id"] in identities:
            raise ValueError(f"Duplicate source_id: {row['source_id']}")
        identities.add(row["source_id"])
        exact[row["sha256"]].append(row)
        if row["role"] == "train" and not (row["dataset"] == "sid" and row["official_split"] == "train"):
            raise ValueError("Manifest contains a forbidden training role")
    cross_exact = []
    for digest, group in exact.items():
        partitions = {(row["dataset"], row["official_split"], row["role"]) for row in group}
        if len(partitions) > 1:
            cross_exact.append({"sha256": digest, "source_ids": [row["source_id"] for row in group], "partitions": [list(value) for value in sorted(partitions)]})
    if cross_exact:
        raise ValueError(
            f"Exact cross-role duplicates detected: {len(cross_exact)}; "
            f"examples={cross_exact[:5]}"
        )

    # Bounded near-duplicate review: compare hashes sharing the first 16 bits.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["phash"][:4]].append(row)
    near = []
    for group in buckets.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if (left["dataset"], left["official_split"], left["role"]) == (right["dataset"], right["official_split"], right["role"]):
                    continue
                distance = imagehash.hex_to_hash(left["phash"]) - imagehash.hex_to_hash(right["phash"])
                if distance <= near_threshold:
                    near.append({"left": left["source_id"], "right": right["source_id"], "distance": int(distance), "partitions": [[left["dataset"], left["official_split"], left["role"]], [right["dataset"], right["official_split"], right["role"]]]})
                    if len(near) >= 1000:
                        return {"exact_cross_role": [], "near_cross_role": near, "near_threshold": near_threshold, "near_report_truncated": True}
    return {"exact_cross_role": [], "near_cross_role": near, "near_threshold": near_threshold, "near_report_truncated": False}


def write_parquet_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def training_descriptors(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    permitted = [row for row in rows if row["dataset"] == "sid" and row["role"] == "train" and row["binary_label"] in {0, 1}]
    condition_ids = sorted(GRADED_CONDITIONS)
    result = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in permitted:
        by_class[int(row["binary_label"])].append(row)
    for label, class_rows in sorted(by_class.items()):
        for index, row in enumerate(sorted(class_rows, key=lambda item: item["source_id"])):
            base = {"source_id": row["source_id"], "source_group_id": row["source_group_id"], "dataset": row["dataset"], "official_split": row["official_split"], "role": row["role"], "binary_label": label}
            result.append({**base, "condition_id": "clean", "seed": seed_for_source(seed, row["source_id"], 0), "view_index": 0})
            for view_index, offset in ((1, 0), (2, 7)):
                condition_id = condition_ids[(index * 2 + offset + label) % len(condition_ids)]
                if condition_id in HELD_OUT_CONDITIONS:
                    raise AssertionError("Held-out condition entered training plan")
                result.append({**base, "condition_id": condition_id, "seed": seed_for_source(seed, row["source_id"], view_index), "view_index": view_index})
    return result


def evaluation_descriptors(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = []
    sid_conditions = ["clean", *sorted(GRADED_CONDITIONS), *sorted(HELD_OUT_CONDITIONS)]
    for row in rows:
        if row["dataset"] == "sid" and row["role"] in {"calibration", "internal_final_evaluation", "exploratory_tampered"}:
            conditions = sid_conditions
        elif row["dataset"] == "cifake" and row["official_split"] == "test":
            conditions = ["clean"]
        else:
            continue
        for index, condition_id in enumerate(conditions):
            result.append({"source_id": row["source_id"], "source_group_id": row["source_group_id"], "dataset": row["dataset"], "official_split": row["official_split"], "role": row["role"], "binary_label": row["binary_label"], "condition_id": condition_id, "seed": seed_for_source(seed, row["source_id"], index), "view_index": index})
    return result
