"""Freeze the validation proposal and run report-only final evaluation once."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from model.scripts.train_heads import load_cache
from model.src.evaluation import binary_metrics
from model.src.heads import build_head


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "model" / "outputs" / "evaluation"
FINAL = ROOT / "model" / "outputs" / "final"
REPORT = ROOT / "report" / "evaluation_report.md"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy one deployment artifact without exposing a partial destination."""

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def freeze_proposal() -> dict[str, Any]:
    proposal_path = EVALUATION / "proposal.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    source_checkpoint = ROOT / proposal["checkpoint"]
    source_metadata = source_checkpoint.with_name("metadata.json")
    if not source_checkpoint.is_file() or not source_metadata.is_file():
        raise ValueError("Selected checkpoint and adjacent metadata must both exist")
    frozen_path = FINAL / "frozen_model.json"
    FINAL.mkdir(parents=True, exist_ok=True)
    final_checkpoint = FINAL / "checkpoint.pt"
    final_metadata = FINAL / "metadata.json"
    checkpoint_hash = file_hash(source_checkpoint)
    payload = {
        **proposal,
        "checkpoint": final_checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "proposal_sha256": file_hash(proposal_path),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "no_post_evaluation_tuning": True,
    }
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        stable_keys = ("checkpoint_sha256", "temperature", "threshold", "proposal_sha256")
        if any(existing[key] != payload[key] for key in stable_keys):
            raise ValueError("A different final model is already frozen")
        payload["frozen_utc"] = existing.get("frozen_utc", payload["frozen_utc"])
    atomic_copy(source_checkpoint, final_checkpoint)
    atomic_copy(source_metadata, final_metadata)
    if file_hash(final_checkpoint) != checkpoint_hash:
        raise ValueError("Packaged checkpoint hash does not match its source")
    temporary = frozen_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, frozen_path)
    return payload


def select_scope(scope: str, rows: list[dict[str, Any]]) -> list[int]:
    if scope == "heldout":
        return [index for index, row in enumerate(rows) if row["role"] == "calibration" and row["condition"].startswith("heldout_")]
    if scope == "test":
        return [index for index, row in enumerate(rows) if row["role"] == "internal_final_evaluation"]
    if scope == "cifake":
        return [index for index, row in enumerate(rows) if row["dataset"] == "cifake" and row["role"] == "cross_dataset_eval"]
    if scope == "tampered":
        return [index for index, row in enumerate(rows) if row["role"] == "exploratory_tampered"]
    raise ValueError(f"Unknown final-evaluation scope: {scope}")


def load_model(frozen: dict[str, Any], device: str) -> torch.nn.Module:
    checkpoint = ROOT / frozen["checkpoint"]
    metadata_path = checkpoint.with_name("metadata.json")
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise ValueError("Frozen checkpoint and adjacent metadata must both exist")
    if file_hash(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = build_head(metadata["architecture"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def infer(model: torch.nn.Module, embeddings: torch.Tensor, temperature: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), 8192):
            logits.append(model(embeddings[start:start + 8192].to(device)).cpu())
    values = torch.cat(logits)
    return values.numpy(), torch.sigmoid(values / temperature).numpy()


def evaluate_scope(scope: str, model: torch.nn.Module, frozen: dict[str, Any], device: str) -> dict[str, Any]:
    cache_scope = "val" if scope == "heldout" else scope
    embeddings, rows, fingerprint = load_cache(cache_scope)
    indices = select_scope(scope, rows)
    selected_rows = [rows[index] for index in indices]
    if not indices:
        raise ValueError(f"No rows for scope {scope}")
    logits, probabilities = infer(model, embeddings[indices], frozen["temperature"], device)
    output_dir = EVALUATION / "final"; output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{scope}_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_id", "dataset", "split", "role", "condition", "label", "logit", "probability", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, row in enumerate(selected_rows):
            writer.writerow({"source_id": row["source_id"], "dataset": row["dataset"], "split": row["split"], "role": row["role"], "condition": row["condition"], "label": row["label"], "logit": float(logits[index]), "probability": float(probabilities[index]), "prediction": int(probabilities[index] >= frozen["threshold"])})
    conditions = np.array([row["condition"] for row in selected_rows])
    if scope == "tampered":
        metrics = {}
        for condition in sorted(set(conditions)):
            values = probabilities[conditions == condition]
            metrics[condition] = {"count": int(len(values)), "mean_probability": float(values.mean()), "median_probability": float(np.median(values)), "q10": float(np.quantile(values, 0.1)), "q90": float(np.quantile(values, 0.9)), "fraction_above_threshold": float((values >= frozen["threshold"]).mean())}
    else:
        labels = np.array([row["label"] for row in selected_rows], dtype=int)
        metrics = {condition: binary_metrics(labels[conditions == condition], probabilities[conditions == condition], frozen["threshold"]) for condition in sorted(set(conditions))}
    result = {"scope": scope, "cache_fingerprint": fingerprint, "row_count": len(indices), "metrics": metrics, "prediction_file": prediction_path.relative_to(ROOT).as_posix()}
    (output_dir / f"{scope}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def paired_test_summary(prediction_path: Path, threshold: float) -> dict[str, Any]:
    with prediction_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clean = {row["source_id"]: float(row["probability"]) for row in rows if row["condition"] == "clean"}
    transformed = [row for row in rows if row["condition"] != "clean"]
    if len(clean) != len({row["source_id"] for row in rows}):
        raise ValueError("Paired join requires exactly one clean row per source")
    deltas, flips = [], 0
    for row in transformed:
        base = clean[row["source_id"]]; probability = float(row["probability"]); label = int(row["label"])
        deltas.append(probability - base)
        flips += int((base >= threshold) == label and (probability >= threshold) != label)
    return {"paired_transformed_rows": len(transformed), "mean_probability_delta": float(np.mean(deltas)), "mean_absolute_probability_delta": float(np.mean(np.abs(deltas))), "clean_to_transformed_correctness_flips": flips}


def table(lines: list[str], title: str, metrics: dict[str, Any], tampered: bool = False) -> None:
    lines += ["", f"## {title}", ""]
    if tampered:
        lines += ["| Condition | N | Mean probability | Median | Above threshold |", "|---|---:|---:|---:|---:|"]
        for condition, value in metrics.items():
            lines.append(f"| {condition} | {value['count']} | {value['mean_probability']:.4f} | {value['median_probability']:.4f} | {value['fraction_above_threshold']:.4f} |")
    else:
        lines += ["| Condition | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Authentic FPR | Brier | ECE | TPR@1% FPR | TPR@5% FPR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for condition, value in metrics.items():
            fmt = lambda key: "null" if value[key] is None else f"{value[key]:.4f}"
            lines.append(f"| {condition} | {value['count']} | {fmt('accuracy')} | {fmt('balanced_accuracy')} | {fmt('precision')} | {fmt('recall')} | {fmt('f1')} | {fmt('roc_auc')} | {fmt('pr_auc')} | {fmt('authentic_false_positive_rate')} | {fmt('brier_score')} | {fmt('ece_10_bin')} | {fmt('tpr_at_1pct_fpr')} | {fmt('tpr_at_5pct_fpr')} |")


def write_report(frozen: dict[str, Any], results: dict[str, dict[str, Any]], paired: dict[str, Any]) -> None:
    calibration = json.loads((ROOT / frozen["checkpoint"]).with_name("calibration.json").read_text(encoding="utf-8"))
    test_metrics = results["test"]["metrics"]
    graded = [value["balanced_accuracy"] for key, value in test_metrics.items() if key != "clean" and not key.startswith("heldout_")]
    heldout_test = [value["balanced_accuracy"] for key, value in test_metrics.items() if key.startswith("heldout_")]
    lines = ["# Stage 7 Evaluation Report", "", "Status: final configuration frozen; no post-evaluation tuning occurred.", "", "## Frozen model", "", f"- Experiment: `{frozen['experiment']}`", f"- Seed: `{frozen['seed']}`", f"- Checkpoint: `{frozen['checkpoint']}`", f"- Checkpoint SHA-256: `{frozen['checkpoint_sha256']}`", f"- Temperature: `{frozen['temperature']:.6f}`", f"- Threshold: `{frozen['threshold']:.6f}`", "", "Selection used SID validation clean plus graded conditions only. Held-out, SID test, CIFAKE, WildFake, and tampered data were report-only.", "", "## Core trade-offs", "", "| Scope | Balanced accuracy | Notes |", "|---|---:|---|", f"| SID test clean | {test_metrics['clean']['balanced_accuracy']:.4f} | In-domain clean baseline |", f"| SID test graded worst | {min(graded):.4f} | Worst of 19 trained transformation conditions |", f"| SID test graded macro | {float(np.mean(graded)):.4f} | Mean across 19 graded conditions |", f"| SID test held-out worst | {min(heldout_test):.4f} | Never used for training or selection |", f"| SID test held-out macro | {float(np.mean(heldout_test)):.4f} | WebP, composition, and out-of-range JPEG |", f"| CIFAKE clean | {results['cifake']['metrics']['clean']['balanced_accuracy']:.4f} | Cross-dataset; 32x32 inputs upscaled |", "", "Per-condition authentic FPR and TPR at 1%/5% FPR appear in the complete tables below."]
    table(lines, "Validation graded conditions", calibration["condition_metrics"])
    table(lines, "Validation held-out conditions", results["heldout"]["metrics"])
    table(lines, "SID final evaluation", results["test"]["metrics"])
    table(lines, "CIFAKE cross-dataset evaluation", results["cifake"]["metrics"])
    table(lines, "Exploratory tampered diagnostic", results["tampered"]["metrics"], tampered=True)
    lines += ["", "## Paired robustness", "", f"- Transformed pairs: {paired['paired_transformed_rows']}", f"- Mean probability delta: {paired['mean_probability_delta']:.6f}", f"- Mean absolute probability delta: {paired['mean_absolute_probability_delta']:.6f}", f"- Clean-to-transformed correctness flips: {paired['clean_to_transformed_correctness_flips']}", "", "## Generalisation and limitations", "", "CIFAKE is a non-scoring cross-dataset check. Its source images are 32x32 and are upscaled by the CLIP processor. The large SID-to-CIFAKE drop demonstrates that strong in-domain robustness does not guarantee cross-dataset generalisation.", "", "WildFake: Not run; the user explicitly omitted this optional demonstration dataset.", "", "Tampered images are outside the trained fully-synthetic-versus-authentic binary task and are reported only as score distributions.", "", "## Plots and artifacts", "", "- Reliability plot: `model/outputs/evaluation/final/reliability.png`", "- Condition degradation plot: `model/outputs/evaluation/final/condition_balanced_accuracy.png`", "- Clean confusion matrix: `model/outputs/evaluation/final/clean_confusion_matrix.png`", "- Machine-readable predictions and metrics: `model/outputs/evaluation/final/`"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(results: dict[str, dict[str, Any]]) -> None:
    output = EVALUATION / "final"
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (output / "test_predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clean = [row for row in rows if row["condition"] == "clean"]
    probabilities = np.array([float(row["probability"]) for row in clean]); labels = np.array([int(row["label"]) for row in clean])
    edges = np.linspace(0, 1, 11); confidence, observed = [], []
    for index in range(10):
        mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1] if index == 9 else probabilities < edges[index + 1])
        if mask.any(): confidence.append(float(probabilities[mask].mean())); observed.append(float(labels[mask].mean()))
    figure, axis = plt.subplots(figsize=(5, 5)); axis.plot([0, 1], [0, 1], "--", color="gray"); axis.plot(confidence, observed, marker="o"); axis.set(xlabel="Mean predicted probability", ylabel="Observed synthetic fraction", title="SID test clean reliability"); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(output / "reliability.png", dpi=160); plt.close(figure)
    metrics = results["test"]["metrics"]; names = list(metrics); values = [metrics[name]["balanced_accuracy"] for name in names]
    figure, axis = plt.subplots(figsize=(12, 6)); axis.bar(range(len(names)), values); axis.set_xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7); axis.set_ylim(.5, 1.01); axis.set(ylabel="Balanced accuracy", title="SID test performance by condition"); axis.grid(axis="y", alpha=.25); figure.tight_layout(); figure.savefig(output / "condition_balanced_accuracy.png", dpi=160); plt.close(figure)
    matrix = np.array(metrics["clean"]["confusion_matrix"]); figure, axis = plt.subplots(figsize=(4, 4)); image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix): axis.text(column, row, str(value), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Authentic", "Synthetic"], yticklabels=["Authentic", "Synthetic"], xlabel="Predicted", ylabel="Actual", title="SID test clean confusion matrix"); figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output / "clean_confusion_matrix.png", dpi=160); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--scope", default="all", choices=("all", "heldout", "test", "cifake", "tampered")); parser.add_argument("--frozen", action="store_true"); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); args = parser.parse_args()
    if not args.frozen: raise ValueError("Final evaluation requires --frozen")
    frozen = freeze_proposal(); device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device); model = load_model(frozen, device)
    scopes = ("heldout", "test", "cifake", "tampered") if args.scope == "all" else (args.scope,)
    results = {scope: evaluate_scope(scope, model, frozen, device) for scope in scopes}
    if args.scope == "all":
        paired = paired_test_summary(ROOT / results["test"]["prediction_file"], frozen["threshold"]); (EVALUATION / "final" / "paired_robustness.json").write_text(json.dumps(paired, indent=2), encoding="utf-8"); write_report(frozen, results, paired)
    print(json.dumps({scope: result["row_count"] for scope, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
