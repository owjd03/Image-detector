# Stage 05 — Resumable CLIP Embedding Extraction

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Extract and cache reproducible CLIP ViT-L/14 embeddings for the training bank and deterministic evaluation suite without storing transformed image copies.

## Required work

1. Implement `model/scripts/extract_embeddings.py` using the shared dataset adapters, transform registry, pinned model loader, and official CLIP processor:

   ```bash
   python -m model.scripts.extract_embeddings --split train --dry-run 64
   python -m model.scripts.extract_embeddings --split train
   python -m model.scripts.extract_embeddings --split val --conditions clean,jpeg_q30,blur_s2.0 --subsample 2000
   python -m model.scripts.extract_embeddings --split val --conditions all
   ```

2. Apply the selected image-space transform first, then the CLIP processor. Use the vision embedding projection, L2-normalize each 768-value vector, and store embeddings as float16. Convert to float32 when training heads.
3. Extract, in this priority order so that an interrupted run still leaves a trainable system:
   - Core SID training: one clean and two planned augmented variants per source. **This is the highest priority; nothing downstream works without it.**
   - SID validation on a fixed subsample across a narrow `--conditions` set, sufficient to drive Stage 06 checkpoint selection.
   - SID validation/test: clean plus every graded condition, then every held-out condition.
   - CIFAKE: clean only for the required cross-dataset generalisation check; transformed CIFAKE is optional after core artifacts exist.
   - WildFake: clean only in a physically separate `external_demo_only` namespace; transformed WildFake is optional and never adaptive.
4. Support `--conditions` (comma-separated IDs or `all`) and `--subsample N` to scope the evaluation sweep. The fixed validation subsample must be seed-derived and stable across runs so Stage 06 selection is reproducible.
5. Extract held-out conditions for validation and test sources **only**. Requesting them for a training split is a hard failure, not a warning — they exist precisely because the model must never have seen them.
6. Write fixed-size safetensors shards and Parquet sidecars mapping every row to `source_id`, split, label, condition, seed, shard, and row offset. Never use pickle for embedding caches. Stage 06's grouped-batch sampler depends on `source_id` being present and stable, so a source and all of its variants must carry the identical value.
7. Compute a cache fingerprint from dataset manifest hash, model ID/revision, processor config, transform-registry version, augmentation-plan hash, and seed. Reject mismatched caches instead of appending to them.
8. Make extraction resumable at completed-shard boundaries. Write to temporary files and atomically rename only after tensor and metadata counts validate.
9. Log images/second, batches completed, elapsed time, ETA, unreadable records, output size, device, dtype, and CUDA peak memory. Support configurable batch size, workers, split, condition, and dry-run sample count. Report the ETA before the bulk of the work starts, so an over-budget sweep can be narrowed rather than abandoned halfway.
10. Fail if WildFake metadata is requested with a training purpose or if a derivative's source split disagrees with the manifest.
11. Keep caches disk-backed. Choose the preflight batch from detected VRAM: 32 for at least 16 GB, 16 for 12–15 GB, 8 for 8–11 GB, or 4 below 8 GB. Start with four data-loader workers and prefetch factor two; increase workers to eight only if measured throughput improves without excessive RAM use. On CUDA out-of-memory, halve the batch to a minimum of one, clear only transient CUDA state, resume from the last completed shard, and record the stable settings. Do not delete valid shards.

## Verification

- Check every tensor has shape `[N, 768]`, finite values, and near-unit norm.
- Check tensor rows and sidecar rows match exactly with no duplicate composite key.
- Interrupt a test extraction and prove resume neither duplicates nor skips records.
- Re-run a sample and compare embeddings within the documented mixed-precision tolerance.
- Run a small CPU smoke extraction and a CUDA throughput benchmark.
- Confirm every variant of one source shares its `source_id`, so grouped batching in Stage 06 is possible.
- Confirm requesting a held-out condition for a training split fails.
- Confirm the fixed validation subsample selects identical records across two runs.

## Exit gate

Report cache fingerprints, counts, corrupt records, storage, detected VRAM, stable batch size, measured throughput, and duration, then continue to Stage 06.
