# Stage 8 Error Analysis

Status: required tier complete; optional saliency and manual tag vocabulary not run.

## Scope

The frozen SID final-evaluation predictions contained 46000 source-condition rows. This review selected 30 representative cases while allowing each source at most once per ranked list.

- Highest-confidence false positives: EA-001, EA-002, EA-003, EA-004, EA-005, EA-006, EA-007, EA-008, EA-009, EA-010
- Highest-confidence false negatives: EA-011, EA-012, EA-013, EA-014, EA-015, EA-016, EA-017, EA-018, EA-019, EA-020
- Largest clean-to-transformed correctness flips: EA-021, EA-022, EA-023, EA-024, EA-025, EA-026, EA-027, EA-028, EA-029, EA-030

Case metadata: `model/outputs/error_analysis/cases.csv`.

## Recurring patterns

The ranked failures concentrate in the conditions listed below. These are associations in the frozen model output, not proof that CLIP used a particular visible feature.

| Condition | Selected cases |
|---|---:|
| heldout_webp_q50 | 14 |
| noise_s0.10 | 6 |
| center_crop_0.8 | 3 |
| heldout_jpeg_q20 | 2 |
| noise_s0.05 | 2 |
| jpeg_q30 | 1 |
| heldout_jpeg_q50_resize_0.5 | 1 |
| blur_s2.0 | 1 |

## Visual review observations

The highest-confidence false positives frequently have a polished or stylized appearance: shallow-depth-of-field food/product photographs, saturated portraits, signage or text, and sparse iconic compositions. These authentic images may resemble visual conventions represented unevenly in the training data.

The false negatives include photorealistic people, sports and traffic scenes, decorative objects, and dark low-contrast imagery. Several look visually plausible at contact-sheet scale, so the errors cannot be reduced to one obvious anatomical or physical artifact.

Seven of the ten largest clean-to-transformed flips use held-out WebP compression and three use added noise. Some sources recur across an error list and the flip list, demonstrating that a modest transformation can reverse a highly confident verdict.

Visual review of the local contact sheet should be interpreted conservatively: similar composition, texture, compression, or subject matter can coexist with many unobserved embedding cues. The classifier head cannot be said to identify a specific anatomical or physical defect without a validated explanation method.

## Authentic-image false-positive harms

A false positive wrongly labels an authentic image as likely AI-generated. In moderation, journalism, education, or evidence review this can suppress legitimate work and make an accusation without provenance evidence. The detector should therefore be used as a triage signal, never as the sole basis for removal, penalties, or attribution.

## Transformation sensitivity

The selected flip list contains 10 distinct sources that were correct when clean and incorrect after a transformation. This demonstrates that aggregate robustness scores can hide individual instability. The Stage 7 paired evaluation recorded 177 such condition-level flips overall.

## Limits of pixel-only detection

A pixel classifier estimates similarity to patterns in its training distribution; it does not verify who created an image or how it was produced. Resizing, recompression, editing, screenshots, novel generators, and dataset shifts can alter those patterns. Provenance claims require trustworthy metadata, signatures, or content credentials in addition to model scores.

## Licensing and repository status

Raw SID images and the generated contact sheet remain under ignored `model/outputs/error_analysis/images/` and are not intended for Git. The tracked report references cases only by stable IDs and contains no absolute local paths.

## Optional work

- Manual artifact tags: Not run
- Saliency/attention rollout: Not run
- Public report imagery: Omitted pending explicit licensing review
