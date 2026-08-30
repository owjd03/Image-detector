"""Regenerate Stage 7 report and plots from frozen saved outputs only."""

from __future__ import annotations

import json

from model.scripts.evaluate import EVALUATION, FINAL, paired_test_summary, write_plots, write_report


def main() -> None:
    frozen = json.loads((FINAL / "frozen_model.json").read_text(encoding="utf-8"))
    results = {scope: json.loads((EVALUATION / "final" / f"{scope}_metrics.json").read_text(encoding="utf-8")) for scope in ("heldout", "test", "cifake", "tampered")}
    paired = paired_test_summary(EVALUATION / "final" / "test_predictions.csv", frozen["threshold"])
    write_plots(results)
    write_report(frozen, results, paired)
    print("Final Stage 7 report and plots regenerated without inference")


if __name__ == "__main__":
    main()
