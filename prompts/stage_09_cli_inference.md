# Stage 09 — Graded CLI Inference

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Deliver the independent directory-to-JSON command required by the challenge, using exactly the same frozen inference implementation later used by the API.

## Contract

```bash
python model/scripts/predict.py --input_dir <directory> --output <predictions.json>
```

Recursively traverse `.jpg`, `.jpeg`, `.png`, and `.webp` case-insensitively in sorted relative-path order. For each readable static image, correct EXIF orientation, convert to RGB, infer a calibrated probability, and emit exactly:

```json
[{"image_path":"nested/example.jpg","pred":0.7312}]
```

`pred` must be a finite JSON number in `[0,1]`. Paths must be POSIX-style and relative to `input_dir`.

## Required work

1. Create one `InferenceEngine` in `model/src/inference.py` that loads the pinned CLIP vision backbone plus `checkpoint.pt`, `metadata.json`, and `frozen_model.json` from `model/outputs/final/` once. Verify the checkpoint SHA-256 before loading it.
2. Batch valid images and preserve deterministic output order. Support `--checkpoint`, `--config`, `--device auto|cpu|cuda`, and `--batch-size` while defaulting to the frozen Stage 07 configuration.
3. Write output atomically. For unreadable/unsupported files, continue, warn on stderr, and write `<output-stem>.errors.json` with path and sanitized reason. Never add error objects to the required predictions array.
4. Return nonzero only for fatal conditions: invalid input directory, unavailable requested device, model/config failure, no readable supported images, or output failure.
5. Avoid stack traces for expected user errors; offer `--verbose` for diagnostic traces.
6. Run the finished CLI over the supplied WildFake demonstration subset and keep the resulting `predictions.json` as a shipped artifact. It costs one command and proves the graded script runs end-to-end on the organizers' own data shape and scale rather than only on a handful of local samples. This is a demonstration run: it must not feed back into any threshold, calibration, or selection decision.
7. Write a stub `README.md` at the repository root before leaving this stage — what the project is, how to install it, and how to run `predict.py`, with a placeholder for results. Stage 12 expands it into the full deliverable. A repository with a working graded CLI and a short README is a valid submission; one with neither is not.

## Markdown report

Generate `report/cli_report.md` containing the exact tested commands, checkpoint and fingerprint, input/readable/error counts, timings, device, output-schema verification, offline status, test results, and repository-relative links to prediction/error artifacts. Do not include absolute local paths or invented results.

## Tests

- Nested folders, mixed-case extensions, EXIF rotation, grayscale/RGBA conversion, corrupt images, unsupported files, empty directories, and output overwrite behavior.
- CPU inference and mocked batched inference without network access.
- Repeated runs yield identical ordering and probabilities within tolerance.
- CLI probabilities match direct `InferenceEngine` calls.
- Output schema contains only `image_path` and `pred`.
- Model loads once, not per image.

## Exit gate

Run the CLI against a mixed sample directory from a clean process with CLIP offline. Report prediction/error JSON, timings, tests, and README status; confirm `report/cli_report.md` was generated, then continue directly to the core Stage 12 work.
