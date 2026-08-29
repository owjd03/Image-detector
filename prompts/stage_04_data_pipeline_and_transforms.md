# Stage 04 — Unified Data Pipeline and Robustness Transforms

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Create leakage-safe dataset manifests and exact, reusable challenge transformations. All transformations must occur before CLIP preprocessing.

## Manifest contract

Write Parquet manifests with: `source_id`, `dataset`, `native_locator`, `official_split`, `original_label`, nullable `binary_label`, `role`, `width`, `height`, `sha256`, `phash`, and `source_group_id`. Use repository-relative paths or dataset record identifiers, never user-specific absolute paths.

SID mappings are fixed: label 0 becomes binary 0, label 1 becomes binary 1, and label 2 has a null binary label and role `exploratory_tampered`. Preserve official train/validation/test assignments. CIFAKE takes role `cross_dataset_eval` — it is a smoke-test and Stage 07 generalisation benchmark, never a training source. WildFake remains `external_demo_only`. Only SID records with roles derived from its official train split may enter training.

## Required work

1. Implement adapters for CIFAKE folders, SID Parquet records, and WildFake folders behind a shared sample interface returning RGB PIL images plus manifest metadata.
2. Compute SHA-256 over canonical decoded RGB pixels plus dimensions and a perceptual hash. Detect exact cross-split duplicates as fatal. Report near duplicates across splits for review using a documented Hamming-distance threshold; never silently move official SID records.
3. Implement named deterministic conditions in `model/src/transforms.py`:
   - JPEG RGB encode/decode: quality 90, 70, 50, 30.
   - Gaussian blur: sigma 0.5, 1.0, 2.0.
   - Bicubic downscale/upscale: 0.5 and 0.25, preserving original dimensions.
   - Gaussian noise on float RGB `[0,1]`: sigma 0.02, 0.05, 0.10, clipped before conversion.
   - Brightness, contrast, and saturation independently: factors 0.8 and 1.2.
   - Centre crop: retain 80% of width and height, then bicubic-resize to original dimensions.
4. Define stable condition IDs such as `jpeg_q30`, `noise_s0.05`, and `brightness_x0.8`. Derive evaluation noise seeds from the global seed and `source_id`; do not use process-dependent Python hashes.
5. Implement a second, separate compact **held-out condition registry** that training is forbidden to use:
   - WebP re-encode at quality 50.
   - `jpeg_q50` followed by `resize_0.5`.
   - Out-of-range JPEG quality 20.

   Mark these with a distinct ID prefix (for example `heldout_webp_q50`) so no downstream stage can confuse them with the graded grid.
6. Implement a balanced training augmentation planner. For every core training source, emit one clean descriptor and two independently seeded transformed descriptors, drawn **only from the step-3 grid**. Balance assignments across transform families and severities within each class; derivatives retain the same split and source group.
7. Keep evaluation descriptors separate. For each validation/test source, enumerate clean plus every exact deterministic condition from step 3, plus every held-out condition from step 5.
8. Add `model/scripts/visualize_transforms.py` to save labelled before/after grids from user-selected licensed samples:

   ```bash
   python -m model.scripts.build_manifests --config model/configs/default.yaml
   python -m model.scripts.visualize_transforms --samples <dir> --out model/outputs/transform_grids
   ```

## Tests

- Pixel/dimension tests for every condition and parameter.
- Fixed-seed noise reproducibility and different-source seed separation.
- JPEG is a real in-memory encode/decode round trip.
- No transform mutates its input object.
- Derivative descriptors cannot change split, label, or source group.
- Manifest rejection tests for WildFake training roles and cross-split exact duplicates.
- The training augmentation plan never emits a held-out condition ID, mirroring the WildFake-role rejection test.
- Composed held-out conditions apply their steps in the documented order and are not commutative by accident.

## Exit gate

Generate manifests, a leakage report, transform counts, and sanity grids. Manually inspect and report them, then continue to Stage 05 unless leakage is unresolved.
