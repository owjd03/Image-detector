"""Build standardized manifests, leakage evidence, and transform plans."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

from model.src.config import DEFAULT_CONFIG_PATH, load_config
from model.src.data import iter_cifake, iter_sid
from model.src.manifests import evaluation_descriptors, manifest_row, training_descriptors, validate_manifest, write_parquet_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--sid-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--replan-existing", action="store_true", help="Regenerate descriptor plans from the existing dataset manifest without rescanning images")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
        datasets_root = config.paths.resources_dir / "datasets"
        manifests = config.paths.outputs_dir / "manifests"
        if args.replan_existing:
            rows = pq.read_table(manifests / "dataset_manifest.parquet").to_pylist()
            training = training_descriptors(rows, config.project.seed)
            evaluation = evaluation_descriptors(rows, config.project.seed)
            write_parquet_atomic(training, manifests / "training_augmentation_plan.parquet")
            write_parquet_atomic(evaluation, manifests / "evaluation_descriptors.parquet")
            print(json.dumps({"training_descriptor_rows": len(training), "evaluation_descriptor_rows": len(evaluation)}, indent=2))
            return 0
        if args.finalize_existing:
            rows = pq.read_table(manifests / "dataset_manifest.parquet").to_pylist()
            training = pq.read_table(manifests / "training_augmentation_plan.parquet").to_pylist()
            evaluation = pq.read_table(manifests / "evaluation_descriptors.parquet").to_pylist()
            report = validate_manifest(rows, near_threshold=args.near_threshold)
            report.update({
                "manifest_rows": len(rows), "training_descriptor_rows": len(training),
                "evaluation_descriptor_rows": len(evaluation), "roles": dict(Counter(row["role"] for row in rows)),
                "datasets": dict(Counter(row["dataset"] for row in rows)),
                "cifake_scope": "official_test_split_only",
                "condition_counts": dict(Counter(row["condition_id"] for row in training)),
                "elapsed_seconds": 0.0, "finalized_from_existing_artifacts": True,
            })
            (manifests / "leakage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        rows = []
        started = time.perf_counter()
        for index, sample in enumerate(iter_sid(datasets_root / "sid_set" / "tier_a"), 1):
            rows.append(manifest_row(sample))
            if index % 500 == 0:
                print(f"manifested SID {index}", flush=True)
        if not args.sid_only:
            # CIFAKE is report-only; its official test split is the benchmark.
            # Excluding its unused train folder also prevents known train/test
            # pixel duplicates from contaminating the cross-dataset check.
            for index, sample in enumerate(iter_cifake(datasets_root / "cifake", include_train=False), 1):
                rows.append(manifest_row(sample))
                if index % 10_000 == 0:
                    print(f"manifested CIFAKE {index}", flush=True)
        report = validate_manifest(rows, near_threshold=args.near_threshold)
        write_parquet_atomic(rows, manifests / "dataset_manifest.parquet")
        training = training_descriptors(rows, config.project.seed)
        evaluation = evaluation_descriptors(rows, config.project.seed)
        write_parquet_atomic(training, manifests / "training_augmentation_plan.parquet")
        write_parquet_atomic(evaluation, manifests / "evaluation_descriptors.parquet")
        report.update({
            "manifest_rows": len(rows), "training_descriptor_rows": len(training),
            "evaluation_descriptor_rows": len(evaluation), "roles": dict(Counter(row["role"] for row in rows)),
            "datasets": dict(Counter(row["dataset"] for row in rows)),
            "cifake_scope": "official_test_split_only",
            "condition_counts": dict(Counter(row["condition_id"] for row in training)),
            "elapsed_seconds": time.perf_counter() - started,
        })
        report_path = manifests / "leakage_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"Manifest build failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
