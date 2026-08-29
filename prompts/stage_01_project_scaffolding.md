# Stage 01 — Project Scaffolding

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Create a reproducible repository skeleton for the ML pipeline, FastAPI service, and Next.js demo. This stage must not download any model or dataset and must not implement training.

## Prerequisites

- Run from the repository root.
- Use Python 3.12. If it is unavailable, stop and report that prerequisite rather than silently using another version.
- Preserve `prompts/` and all existing files. Never overwrite unrelated user changes.

## Required work

1. Create the top-level directories `frontend/`, `backend/`, `model/`, and `resources/`. Under `model/`, create `src/`, `scripts/`, `configs/`, `outputs/`, and `tests/`. Add Python package markers where imports require them.
2. Add a root `requirements.txt` with exact compatible versions for PyTorch/CUDA, Transformers, Pillow, torchvision, datasets, safetensors, pandas/pyarrow, scikit-learn, matplotlib/seaborn, PyYAML, imagehash, FastAPI, Uvicorn, multipart handling, and pytest. Document that the CUDA-specific PyTorch wheel may require the official PyTorch index command.
3. Add a central YAML configuration containing paths, seed `42`, model ID `openai/clip-vit-large-patch14`, a pinned model revision placeholder populated by Stage 02, embedding dimension `768`, image limits, and output locations. Paths must be relative to the repository or configurable by environment variables.
4. Add small reusable modules for configuration loading, deterministic seeding, device selection, and structured logging. Configuration validation must fail clearly for missing keys or invalid paths.
5. Add `model/scripts/doctor.py`. It must report Python, PyTorch, Transformers, CUDA availability, GPU name/VRAM, disk space, configured paths, and whether model/data assets are present without downloading them.
6. Add `.gitignore` rules for virtual environments, caches, secrets, frontend build output, raw datasets, Hugging Face caches, embedding shards, error-analysis images, and temporary files. Permit small classifier `.pt` files and report-ready `.csv`, `.json`, `.md`, `.png`, and `.svg` files under `model/outputs/` through explicit allow rules.
7. Add `resources/README.md` with source URLs, intended roles, license/attribution placeholders, credential requirements, and the prohibition on training with WildFake demonstration data.
8. Initialize a minimal Next.js TypeScript app in `frontend/` with its own lockfile, but leave the page as a placeholder. Do not add the final UI.
9. Add basic pytest configuration and smoke tests for configuration loading, seeds, imports, and `doctor.py --json`.
10. Reconcile the scaffold with the combined plan: make `resources/datasets/dataset_inventory.json` trackable while raw data remains ignored; make doctor tests valid before and after Stage 02 caches CLIP; reject required paths that are invalid, non-directories, or unwritable; describe CIFAKE as a smoke and non-scoring cross-dataset benchmark; and add the Stage 00 decision record to the root README.
11. Report detected GPU VRAM in doctor output. Never infer batch size from the GPU product name alone.

## Public interfaces established

- `python -m model.scripts.doctor [--json]`
- One typed configuration loader used by all later stages.
- Environment overrides for resource, output, and Hugging Face cache roots.

## Verification

- Create a fresh Python 3.12 virtual environment, install dependencies, and run `pytest`.
- Run the doctor command on CPU-only hardware and confirm it exits successfully while reporting CUDA as unavailable.
- Run the frontend type-check and production build.
- Run `git status --short` and confirm no cache, secret, model, dataset, or build artifact is tracked.

## Exit gate

Report the tree, commands, test results, ignore-rule checks, hardware diagnostics, and platform notes, then continue to Stage 02. Do not download CLIP or datasets during Stage 01.
