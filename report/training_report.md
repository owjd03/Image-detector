# Stage 6 Training Report

CLIP remained frozen; only classifier heads were trained on cached embeddings.

## Runs

| Experiment | Seed | Best epoch | Validation loss | Worst-condition balanced accuracy | Duration (s) |
|---|---:|---:|---:|---:|---:|
| linear_clean | 42 | 100 | 0.187702 | 0.9845 | 2.2 |
| linear_robust | 42 | 100 | 0.085348 | 0.9560 | 10.1 |
| linear_robust | 43 | 100 | 0.085080 | 0.9555 | 10.4 |
| linear_robust | 44 | 100 | 0.085373 | 0.9555 | 10.4 |
| mlp_consistency | 42 | 94 | 0.145016 | 0.9895 | 276.0 |
| mlp_robust | 42 | 30 | 0.024129 | 0.9875 | 4.6 |
| mlp_robust | 43 | 26 | 0.023838 | 0.9875 | 4.5 |
| mlp_robust | 44 | 27 | 0.024386 | 0.9870 | 4.3 |

## Selection

Robust finalists selected from validation only: mlp_robust, linear_robust.

Held-out, SID test, CIFAKE, WildFake, and tampered results were excluded from checkpoint selection.

## Artifacts

- Checkpoints, metadata, and epoch histories: `model/outputs/training/`
- Consolidated comparison: `model/outputs/training/comparison.csv`
- Learning curves: `model/outputs/training/learning_curves.png`

## Verification

Focused Stage 6 tests: 9 passed. CLIP was not imported by `train_heads.py`.

Optional lambda-0.3 sensitivity: Not run.
