# Stage 02 — CLIP Model Download and Verification

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Download and verify a revision-pinned frozen CLIP ViT-L/14 backbone. Store it only in the configured Hugging Face cache and create reproducibility metadata without committing the weights.

## Prerequisites

- Stage 01 has passed.
- The Python 3.12 environment is active.
- Network access and sufficient cache space are available.

## Required work

1. Implement `model/scripts/download_model.py` using Hugging Face APIs, not shell-specific download commands.
2. Resolve `openai/clip-vit-large-patch14` to an immutable commit hash before downloading. Save that hash into the experiment configuration and model manifest; all future loads must use it.
3. Download the processor and the minimum model components needed for image embeddings. The text tower must never be used during inference. Do not copy the snapshot into the repository.
4. Write `model/outputs/manifests/model_manifest.json` containing model ID, revision, download timestamp, Transformers/PyTorch versions, embedding dimension, parameter counts, processor settings, cache location with user-specific prefixes removed, and the `<2B` compliance result.
5. Load the model as a vision feature extractor, set evaluation mode, set every parameter to `requires_grad=False`, and expose a shared loader in `model/src/` for later extraction and inference.
6. Add an offline verification mode that sets Hugging Face offline behavior before loading. It must fail with an actionable message if the pinned snapshot is incomplete.
7. Run a deterministic synthetic-image smoke inference. Assert output shape `[batch, 768]`, finite values, and unit-length L2-normalized embeddings after the shared normalization function.
8. On CUDA, test mixed precision and report peak allocated memory and throughput. On CPU, complete the correctness smoke test and clearly mark the GPU performance check as pending.

## Commands

- `python -m model.scripts.download_model --config model/configs/default.yaml`
- `python -m model.scripts.download_model --verify-only --offline`

Both commands must support `--cache-dir` without changing tracked configuration.

## Tests

- Mocked revision-resolution and manifest tests must run without network access.
- Verify all parameters remain frozen before and after inference.
- Verify two runs over the same image produce numerically equal embeddings within tolerance.
- Verify an invalid or unpinned revision produces a clean nonzero exit.

## Failure handling

Interrupted downloads must be resumable through the Hugging Face cache. Never delete a shared cache automatically. Never fall back to `main` if revision resolution fails.

## Exit gate

Report the immutable revision, parameter count, embedding shape, offline reload result, detected device/VRAM, peak memory, and tests, then continue to Stage 03. Do not download datasets during Stage 02.
