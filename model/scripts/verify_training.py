"""Verify Stage 6 checkpoints, provenance isolation, and seed summaries."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import torch

from model.src.heads import build_head


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    training = ROOT / "model" / "outputs" / "training"
    rows = []
    errors = []
    total_bytes = 0
    expected_exclusions = {"heldout", "test", "cifake", "wildfake", "tampered"}
    for metadata_path in sorted(training.glob("*/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = metadata_path.with_name("checkpoint.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        build_head(metadata["architecture"]).load_state_dict(state, strict=True)
        total_bytes += checkpoint.stat().st_size
        forbidden = [key for key in state if "clip" in key.lower() or "vision" in key.lower()]
        if forbidden:
            errors.append(f"Backbone parameters in {checkpoint}: {forbidden}")
        if set(metadata["selection_excludes"]) != expected_exclusions:
            errors.append(f"Bad isolation metadata: {metadata_path}")
        rows.append(metadata)
        print(
            metadata["experiment"], metadata["seed"],
            "best_epoch", metadata["best_epoch"],
            "val_loss", round(metadata["validation_loss"], 6),
            "worst_BA", round(metadata["worst_condition_balanced_accuracy"], 4),
            "seconds", round(metadata["duration_seconds"], 1),
        )
    for experiment in ("mlp_robust", "linear_robust"):
        values = [row["validation_loss"] for row in rows if row["experiment"] == experiment]
        print(experiment, "seed_loss_mean", round(statistics.mean(values), 6), "stdev", round(statistics.stdev(values), 6))
    print("checkpoint_total_MiB", round(total_bytes / 2**20, 3))
    print("errors", errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
