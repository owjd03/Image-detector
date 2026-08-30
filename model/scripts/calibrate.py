"""Calibrate Stage 6 checkpoints and propose a final model using validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.scripts.train_heads import OUTPUT as TRAINING_OUTPUT, load_cache
from model.src.evaluation import binary_metrics, choose_threshold, fit_temperature
from model.src.heads import build_head


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "model" / "outputs" / "evaluation"
REPORT = ROOT / "report" / "evaluation_report.md"


def validation_rows() -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    embeddings, rows, fingerprint = load_cache("val")
    selected = [index for index, row in enumerate(rows) if row["dataset"] == "sid" and row["role"] == "calibration" and not row["condition"].startswith("heldout_")]
    if not selected or any(rows[index]["split"] != "validation" for index in selected):
        raise ValueError("Validation selection provenance failure")
    return embeddings[selected], [rows[index] for index in selected], fingerprint


def logits_for(model: torch.nn.Module, embeddings: torch.Tensor, device: str) -> torch.Tensor:
    model.to(device).eval(); result = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), 8192):
            result.append(model(embeddings[start:start + 8192].to(device)).cpu())
    return torch.cat(result)


def calibrate_checkpoint(metadata_path: Path, embeddings: torch.Tensor, rows: list[dict[str, Any]], fingerprint: str, device: str) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["validation_cache_fingerprint"] != fingerprint:
        raise ValueError(f"Validation fingerprint mismatch: {metadata_path}")
    model = build_head(metadata["architecture"])
    model.load_state_dict(torch.load(metadata_path.with_name("checkpoint.pt"), map_location="cpu", weights_only=True))
    logits = logits_for(model, embeddings, device)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    temperature, nll_before, nll_after = fit_temperature(logits, labels)
    probabilities = torch.sigmoid(logits / temperature).numpy()
    threshold = choose_threshold(labels.numpy().astype(int), probabilities)
    by_condition = {}
    conditions = np.array([row["condition"] for row in rows])
    label_values = labels.numpy().astype(int)
    for condition in sorted(set(conditions)):
        mask = conditions == condition
        by_condition[condition] = binary_metrics(label_values[mask], probabilities[mask], threshold["threshold"])
    transformed = [value["balanced_accuracy"] for key, value in by_condition.items() if key != "clean"]
    clean_mask = conditions == "clean"
    pooled = binary_metrics(label_values, probabilities, threshold["threshold"])
    summary = {
        "experiment": metadata["experiment"], "seed": metadata["seed"], "checkpoint": metadata_path.with_name("checkpoint.pt").relative_to(ROOT).as_posix(),
        "validation_cache_fingerprint": fingerprint, "temperature": temperature, "nll_before": nll_before, "nll_after": nll_after,
        **threshold, "worst_condition_balanced_accuracy": min(value["balanced_accuracy"] for value in by_condition.values()),
        "macro_transformed_balanced_accuracy": float(np.mean(transformed)), "clean_roc_auc": by_condition["clean"]["roc_auc"],
        "pooled_authentic_false_positive_rate": pooled["authentic_false_positive_rate"], "condition_metrics": by_condition,
        "selection_scope": "SID validation: clean plus 19 graded conditions only",
        "selection_excludes": ["heldout", "test", "cifake", "wildfake", "tampered"],
    }
    metadata_path.with_name("calibration.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary | {"probabilities": probabilities, "logits": logits.numpy()}


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (row["worst_condition_balanced_accuracy"], row["macro_transformed_balanced_accuracy"], -row["pooled_authentic_false_positive_rate"], row["clean_roc_auc"])


def write_outputs(results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = [{key: value for key, value in result.items() if key not in {"probabilities", "logits"}} for result in results]
    proposal = max(summaries, key=selection_key)
    (OUTPUT / "validation_metrics.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT / "proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    with (OUTPUT / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "seed", "source_id", "condition", "label", "logit", "probability"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in results:
            for index, row in enumerate(rows):
                writer.writerow({"experiment": result["experiment"], "seed": result["seed"], "source_id": row["source_id"], "condition": row["condition"], "label": row["label"], "logit": float(result["logits"][index]), "probability": float(result["probabilities"][index])})
    lines = ["# Stage 7 Evaluation Report", "", "Status: validation-only proposal; final evaluation is awaiting user confirmation.", "", "## Validation comparison", "", "| Experiment | Seed | Temperature | Threshold | Worst-condition BA | Macro transformed BA | Authentic FPR | Clean ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in sorted(summaries, key=lambda row: (row["experiment"], row["seed"])):
        lines.append(f"| {result['experiment']} | {result['seed']} | {result['temperature']:.4f} | {result['threshold']:.4f} | {result['worst_condition_balanced_accuracy']:.4f} | {result['macro_transformed_balanced_accuracy']:.4f} | {result['pooled_authentic_false_positive_rate']:.4f} | {result['clean_roc_auc']:.4f} |")
    lines += ["", "## Proposed frozen model", "", f"- Experiment: `{proposal['experiment']}`", f"- Seed: `{proposal['seed']}`", f"- Checkpoint: `{proposal['checkpoint']}`", f"- Temperature: `{proposal['temperature']:.6f}`", f"- Threshold: `{proposal['threshold']:.6f}`", "", "No held-out, SID test, CIFAKE, WildFake, or tampered values were read for this proposal.", "", "Final condition tables and plots: Pending user confirmation."]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", default="all"); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    embeddings, rows, fingerprint = validation_rows()
    paths = sorted(TRAINING_OUTPUT.glob("*/seed_*/metadata.json"))
    if args.experiment != "all":
        requested = set(args.experiment.split(",")); paths = [path for path in paths if path.parents[1].name in requested]
    results = [calibrate_checkpoint(path, embeddings, rows, fingerprint, device) for path in paths]
    proposal = write_outputs(results, rows)
    print(json.dumps({key: proposal[key] for key in ("experiment", "seed", "checkpoint", "temperature", "threshold", "worst_condition_balanced_accuracy", "macro_transformed_balanced_accuracy", "pooled_authentic_false_positive_rate", "clean_roc_auc")}, indent=2))


if __name__ == "__main__":
    main()
