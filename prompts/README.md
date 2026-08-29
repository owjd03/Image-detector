# Combined Execution Plan

This is the authoritative index for the TechJam robust AI-image detector. Read it with `stage_00_critical_path.md` before executing an individual stage.

## Locked decisions

- Fewer than 72 hours remain; protect judged deliverables over optional breadth.
- Hardware is an RTX 5080 machine with 32 GB system RAM. Detect VRAM at runtime with `nvidia-smi` or `python -m model.scripts.doctor --json`; do not hard-code it.
- Use the deterministic Tier A SID subset: 10,000 core training images, 2,000 calibration/validation images, 2,000 internal final-evaluation images, and up to 250 tampered diagnostic images.
- Use frozen, revision-pinned `openai/clip-vit-large-patch14` and disk-backed embedding shards.
- Must-ship artifacts are the graded directory-to-JSON CLI, README, clean-versus-transformed robustness table, and error-analysis note.
- FastAPI and Next.js are conditional: start them only after the core submission passes and at least 10 focused hours remain.
- Stage 13 is skipped under the current deadline and documented as future work.

## Execution order

1. Stage 00 records the deadline, data tier, and cut policy.
2. Stage 01 repairs and validates the scaffold.
3. Stage 02 pins, downloads, and verifies CLIP.
4. Stage 03 materializes and inventories Tier A data.
5. Stage 04 builds manifests, leakage checks, graded transforms, and three compact held-out checks.
6. Stage 05 extracts resumable, disk-backed embeddings with measured batch-size tuning.
7. Stage 06 trains four seed-42 core experiments and extra seeds only for two finalists.
8. Stage 07 calibrates and freezes the model before report-only evaluation.
9. Stage 08 produces the required error-analysis note.
10. Stage 09 delivers the graded CLI and runnable README.
11. Stage 12 completes submission documentation and audits.
12. Stages 10–11 run only if their time gate passes; otherwise the CLI and plots drive the demo video.
13. Stage 13 is not executed under the current budget.

## Gate policy

Ordinary stages report evidence and continue automatically. Stop for confirmation only before the Stage 07 model freeze and at the Stage 12 manual submission checklist. Missing credentials, inaccessible protected data, hardware failure, or external publication authority remain genuine blockers.

## Resource and cut policy

- Keep caches on disk; never assume the dataset or embedding bank fits in 32 GB RAM.
- Choose the extraction preflight batch from detected VRAM: 32 for at least 16 GB, 16 for 12–15 GB, 8 for 8–11 GB, or 4 below 8 GB. Halve after an out-of-memory error, resume from the last complete shard, and record the stable size and measured throughput.
- Cut in this order: Stage 13, expanded held-out conditions, bootstrap intervals and extra plots, lambda sensitivity, extra seeds, Stage 11, then Stage 10.
- Never cut the CLI, complete graded transform table, required error analysis, or README.

## Stage index

- `stage_00_critical_path.md`
- `stage_01_project_scaffolding.md`
- `stage_02_model_download.md`
- `stage_03_dataset_download.md`
- `stage_04_data_pipeline_and_transforms.md`
- `stage_05_embedding_extraction.md`
- `stage_06_model_training.md`
- `stage_07_calibration_and_evaluation.md`
- `stage_08_error_analysis.md`
- `stage_09_cli_inference.md`
- `stage_10_backend_api.md`
- `stage_11_frontend_demo.md`
- `stage_12_integration_documentation_submission.md`
- `stage_13_optional_stretch.md`
