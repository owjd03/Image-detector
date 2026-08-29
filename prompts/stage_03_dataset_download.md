# Stage 03 — Dataset Download and Inventory

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Acquire CIFAKE, SID_Set at a deliberately chosen data tier, and the challenge-supplied WildFake demonstration subset while preserving provenance and preventing WildFake from entering any adaptive workflow.

## Prerequisites

- Stages 01–02 have passed.
- Kaggle credentials may be supplied through the standard credential mechanism; never print or commit them.
- Tier A is locked by the combined plan. Confirm at least 30 GB free for materialized images, metadata, and working headroom.

## Data tiers

The full SID_Set is approximately 140 GB. Downloading it can consume an entire hackathon window, so the tier is a deliberate, recorded choice rather than an assumption.

- **Tier A — selected.** Use Hugging Face streaming with a deterministic buffered shuffle to materialise 15,000 real and 15,000 fully synthetic training records; 3,000 per class for calibration/validation; 3,000 per class for final evaluation; and up to 1,000 tampered records for report-only diagnostics. Preserve each record's parent official split. Requires approximately 15–30 GB free.
- **Tier B — full official SID_Set.** Only when both disk (at least 220 GB free) and time clearly allow. Identical downstream code path; only the record count differs.
- **Tier C — CIFAKE fallback.** Use only if SID_Set is unreachable. CIFAKE's 32×32 images upscaled to CLIP's 224×224 input are a weak substitute; if this tier is used, say so plainly in the README rather than presenting the numbers as comparable.

Sampling must be deterministic under the global seed, stratified by split and class, and must never draw validation or test records into training. If the protected official test split is unavailable, divide only sampled official-validation records into disjoint `calibration` and `internal_final_evaluation` roles; never call the latter an official SID test.

## Required work

1. Implement `model/scripts/download_datasets.py` with `--dataset cifake|sid|wildfake|all`, `--tier a|b|c`, `--train-per-class`, `--calibration-per-class`, `--evaluation-per-class`, `--tampered-limit`, `--verify-only`, and `--resume`. Tier A defaults are 15,000, 3,000, 3,000, and 1,000 respectively:

   ```bash
   python -m model.scripts.download_datasets --dataset all --tier a
   python -m model.scripts.download_datasets --dataset sid --verify-only
   python -m model.scripts.download_datasets --dataset sid --resume
   ```

2. Download CIFAKE through the Kaggle API into `resources/datasets/cifake/`. Preserve its supplied train/test organization. Under Tiers A and B it is a smoke-test and cross-dataset generalisation set, not a training source.
3. Acquire SID_Set train, validation, and obtainable test assets through Hugging Face at the selected tier, in its cache/native Parquet representation. Do not expand or copy every record into class folders. Verify columns `img_id`, `image`, `mask`, `width`, `height`, and `label` where applicable. Record labels `0=real`, `1=full synthetic`, and `2=tampered`.
4. Follow the SID_Set repository’s official process if the protected test split requires an additional retrieval step. Never manufacture a test split if the official test data cannot be obtained; report the missing split explicitly.
5. Download or document the exact manual retrieval procedure for only the supplied WildFake demonstration subset: COCO val2017 authentic images and DALL-E Advanced synthetic images. Store it under `resources/datasets/wildfake_validation/` with a sentinel metadata file setting `role=external_demo_only`.
6. Write `resources/datasets/dataset_inventory.json` with dataset name, source URL, version/revision, license, local/native location, split and class counts, byte totals, retrieval status, and intended role. Record the selected tier, sampling seed, and all four quota values so reported results can be traced to their exact data. Do not include credentials or machine-specific absolute prefixes.
7. Sample-decode images from every available class/split, correct no files, and report unreadable samples rather than silently dropping them.
8. Update `resources/README.md` with exact commands, citations, licenses, expected sizes, and manual steps.

## Safety invariants

- WildFake must never be relabelled as train/validation/test; its only role is `external_demo_only`.
- SID tampered records are downloaded but excluded from the core binary task in later stages.
- Download scripts may resume but must not recursively delete partial or user-provided data.
- Subset sampling must preserve official split membership. A record's split is a property of the dataset, never of the sampler.

## Verification

- Run verification-only mode after download.
- Confirm inventory counts against upstream dataset cards and challenge-provided WildFake counts. Under Tier A, confirm the sampled counts are stratified as requested and that re-running with the same seed selects the same records.
- Confirm all resource files remain ignored by Git while inventory and documentation remain trackable.
- Test missing credentials, insufficient disk, partial download, unavailable test split, and corrupt sample errors.

## Exit gate

Report the Tier A inventory, counts, disk use, unavailable components, sampling limitations, and verification results, then continue to Stage 04. Stop only if credentials or accessible source data block progress.
