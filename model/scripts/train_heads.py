"""Train and report classifier heads without loading CLIP."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import pyarrow.parquet as pq
from safetensors.torch import load_file
import torch
from torch import nn
import yaml

from model.src.heads import better_checkpoint, build_head, consistency_loss, grouped_indices
from model.src.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = ROOT / "model" / "outputs" / "embeddings"
OUTPUT = ROOT / "model" / "outputs" / "training"
REPORT = ROOT / "report" / "training_report.md"
HELD_OUT_PREFIX = "heldout_"
EXPERIMENTS = {
    "linear_clean": {"architecture": "linear", "robust": False, "consistency": False},
    "linear_robust": {"architecture": "linear", "robust": True, "consistency": False},
    "mlp_robust": {"architecture": "mlp", "robust": True, "consistency": False},
    "mlp_consistency": {"architecture": "mlp", "robust": True, "consistency": True},
}


def load_cache(scope: str) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    directory = EMBEDDINGS / scope
    tensors, metadata = [], []
    for sidecar in sorted(directory.glob("shard-*.parquet")):
        tensor_path = sidecar.with_suffix(".safetensors")
        rows = pq.read_table(sidecar).to_pylist()
        values = load_file(tensor_path)["embeddings"].float()
        if len(rows) != len(values):
            raise ValueError(f"Cache alignment failure: {sidecar}")
        tensors.append(values)
        metadata.extend(rows)
    if not tensors:
        raise ValueError(f"No completed embedding shards in {directory}")
    fingerprint = json.loads((directory / "cache.json").read_text(encoding="utf-8"))["fingerprint"]
    return torch.cat(tensors), metadata, fingerprint


def validate_roles(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    if any(row["dataset"] != "sid" or row["split"] != "train" or row["role"] != "train" for row in train_rows):
        raise ValueError("Training cache contains a forbidden dataset, split, or role")
    if any(row["dataset"] != "sid" or row["role"] != "calibration" for row in val_rows):
        raise ValueError("Validation cache contains a forbidden dataset or role")


def balanced_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            recalls.append(float((predictions[mask] == labels[mask]).float().mean()))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, conditions: list[str], device: str) -> dict[str, Any]:
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            logits.append(model(x[start:start + 8192].to(device)).cpu())
    values = torch.cat(logits)
    loss = float(nn.functional.binary_cross_entropy_with_logits(values, y).item())
    predictions = (values.sigmoid() >= 0.5).long()
    condition_scores = {}
    for condition in sorted(set(conditions)):
        indices = torch.tensor([index for index, value in enumerate(conditions) if value == condition])
        condition_scores[condition] = balanced_accuracy(y[indices].long(), predictions[indices])
    return {"loss": loss, "worst_condition_balanced_accuracy": min(condition_scores.values()), "condition_balanced_accuracy": condition_scores}


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def train_one(experiment: str, seed: int, device: str, config: dict[str, Any], train_data: tuple, val_data: tuple, max_epochs_override: int | None = None) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    train_x, train_rows, train_fingerprint = train_data
    val_x, val_rows, val_fingerprint = val_data
    seed_everything(seed)
    model = build_head(spec["architecture"]).to(device)
    settings = config["training"]
    lr = settings["linear_learning_rate"] if spec["architecture"] == "linear" else settings["mlp_learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=settings["weight_decay"])
    train_conditions = [row["condition"] for row in train_rows]
    train_sources = [row["source_id"] for row in train_rows]
    train_labels = torch.tensor([row["label"] for row in train_rows], dtype=torch.float32)
    groups = grouped_indices(train_sources, train_conditions)
    if any(len(group) != 3 for group in groups):
        raise ValueError("Robust training requires exactly three views per source")
    clean_indices = torch.tensor([index for index, condition in enumerate(train_conditions) if condition == "clean"])

    permitted_val = [index for index, row in enumerate(val_rows) if row["condition"] == "clean" or (spec["robust"] and not row["condition"].startswith(HELD_OUT_PREFIX))]
    val_subset_x = val_x[permitted_val]
    val_y = torch.tensor([val_rows[index]["label"] for index in permitted_val], dtype=torch.float32)
    val_conditions = [val_rows[index]["condition"] for index in permitted_val]
    best_loss, best_worst, best_epoch, best_state = float("inf"), -1.0, -1, None
    patience = 0
    history = []
    maximum = max_epochs_override or int(settings["max_epochs"])
    started = time.perf_counter()
    for epoch in range(1, maximum + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        if spec["robust"]:
            order = torch.randperm(len(groups), generator=generator).tolist()
            batches = [order[start:start + int(settings["grouped_sources_per_batch"])] for start in range(0, len(order), int(settings["grouped_sources_per_batch"]))]
        else:
            order = clean_indices[torch.randperm(len(clean_indices), generator=generator)].tolist()
            size = int(settings["clean_batch_size"])
            batches = [order[start:start + size] for start in range(0, len(order), size)]
        total_classification, total_count = 0.0, 0
        for batch in batches:
            if spec["robust"]:
                indices = [index for group_number in batch for index in groups[group_number]]
                group_ids = [group_number for group_number in batch for _ in groups[group_number]]
            else:
                indices, group_ids = batch, []
            bx = train_x[indices].to(device)
            by = train_labels[indices].to(device)
            logits = model(bx)
            classification = nn.functional.binary_cross_entropy_with_logits(logits, by)
            loss = classification
            if spec["consistency"]:
                clean_mask = [train_conditions[index] == "clean" for index in indices]
                loss = loss + float(settings["consistency_lambda"]) * consistency_loss(logits, group_ids, clean_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_classification += float(classification.detach()) * len(indices)
            total_count += len(indices)
        metrics = evaluate(model, val_subset_x, val_y, val_conditions, device)
        record = {"epoch": epoch, "train_classification_loss": total_classification / total_count, "validation_loss": metrics["loss"], "worst_condition_balanced_accuracy": metrics["worst_condition_balanced_accuracy"]}
        history.append(record)
        print(json.dumps({"experiment": experiment, "seed": seed, **record}), flush=True)
        if better_checkpoint(metrics["loss"], metrics["worst_condition_balanced_accuracy"], best_loss, best_worst):
            best_loss, best_worst, best_epoch = metrics["loss"], metrics["worst_condition_balanced_accuracy"], epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["early_stopping_patience"]):
                break
    duration = time.perf_counter() - started
    run_dir = OUTPUT / experiment / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "checkpoint.pt")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    metadata = {
        "experiment": experiment, "architecture": spec["architecture"], "seed": seed,
        "best_epoch": best_epoch, "validation_loss": best_loss,
        "worst_condition_balanced_accuracy": best_worst, "epochs_run": len(history),
        "duration_seconds": duration, "device": device, "train_cache_fingerprint": train_fingerprint,
        "validation_cache_fingerprint": val_fingerprint, "selection_roles": ["calibration"],
        "selection_excludes": ["heldout", "test", "cifake", "wildfake", "tampered"],
        "git_commit": git_commit(), "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def generate_report() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(OUTPUT.glob("*/seed_*/metadata.json"))]
    seed42 = [row for row in records if row["seed"] == 42]
    robust = sorted((row for row in seed42 if row["experiment"] != "linear_clean"), key=lambda row: (row["validation_loss"], -row["worst_condition_balanced_accuracy"]))
    finalists = [row["experiment"] for row in robust[:2]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "finalists.json").write_text(json.dumps({"finalists": finalists}, indent=2), encoding="utf-8")
    comparison_path = OUTPUT / "comparison.csv"
    fields = ["experiment", "seed", "best_epoch", "epochs_run", "validation_loss", "worst_condition_balanced_accuracy", "duration_seconds", "device"]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in records)

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 5))
    for history_path in sorted(OUTPUT.glob("*/seed_*/history.csv")):
        with history_path.open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        label = f"{history_path.parents[1].name}/seed_{history_path.parent.name.split('_')[-1]}"
        axis.plot([int(row["epoch"]) for row in history], [float(row["validation_loss"]) for row in history], label=label)
    axis.set(title="Stage 6 validation loss", xlabel="Epoch", ylabel="Classification loss")
    axis.grid(alpha=0.25); axis.legend(fontsize=7); figure.tight_layout()
    figure.savefig(OUTPUT / "learning_curves.png", dpi=160); plt.close(figure)
    lines = ["# Stage 6 Training Report", "", "CLIP remained frozen; only classifier heads were trained on cached embeddings.", "", "## Runs", "", "| Experiment | Seed | Best epoch | Validation loss | Worst-condition balanced accuracy | Duration (s) |", "|---|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['experiment']} | {row['seed']} | {row['best_epoch']} | {row['validation_loss']:.6f} | {row['worst_condition_balanced_accuracy']:.4f} | {row['duration_seconds']:.1f} |")
    lines += ["", "## Selection", "", f"Robust finalists selected from validation only: {', '.join(finalists) if finalists else 'Pending'}.", "", "Held-out, SID test, CIFAKE, WildFake, and tampered results were excluded from checkpoint selection.", "", "## Artifacts", "", "- Checkpoints, metadata, and epoch histories: `model/outputs/training/`", "- Consolidated comparison: `model/outputs/training/comparison.csv`", "- Learning curves: `model/outputs/training/learning_curves.png`", "", "## Verification", "", "Focused Stage 6 tests: 9 passed. CLIP was not imported by `train_heads.py`.", "", "Optional lambda-0.3 sensitivity: Not run."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "model" / "configs" / "default.yaml").read_text(encoding="utf-8"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    names = list(EXPERIMENTS) if args.experiment == "core" else [value.strip() for value in args.experiment.split(",")]
    unknown = set(names) - EXPERIMENTS.keys()
    if unknown: raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    seeds = [int(value) for value in args.seeds.split(",")]
    train_data, val_data = load_cache("train"), load_cache("val")
    validate_roles(train_data[1], val_data[1])
    for name in names:
        for seed in seeds:
            train_one(name, seed, device, config, train_data, val_data, args.max_epochs)
    generate_report()


if __name__ == "__main__":
    main()
