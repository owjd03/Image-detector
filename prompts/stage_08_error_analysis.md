# Stage 08 — Error Analysis and Qualitative Explainability

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Turn evaluation failures into a defensible error-analysis note while keeping licensed dataset imagery out of Git and treating visual explanations as qualitative aids.

The **Error Analysis Note is a required challenge deliverable**, so this stage always produces output. Its scope is tiered: the required tier is roughly an hour of work against artifacts Stage 07 has already written, and the optional tier is everything that makes it polished. Under the Stage 00 budget, do the required tier, ship it, and return to the optional tier only if the cut ladder has spare time.

## Required tier

1. Read the frozen Stage 07 per-image prediction CSV. No new inference and no new script is needed for this tier.
2. Rank and pull the top 10 highest-confidence false positives, the top 10 highest-confidence false negatives, and the top 10 clean-to-transformed correctness flips. Limit repeated source groups so one bad source cannot fill a list.
3. Inspect the images. Record each case in a trackable metadata CSV containing source ID, dataset, split, condition, label, clean probability, transformed probability, and error type.
4. Write the note. It must cover recurring failure patterns, **authentic-image false-positive harms** (a wrongly flagged real photo is the costly error for a platform, and the brief asks for this explicitly), transformation sensitivity, and why pixel-only detection cannot guarantee provenance.
5. Keep raw dataset imagery out of Git — write contact sheets to the ignored `model/outputs/error_analysis/images/` and reference cases by ID in tracked files.

## Optional tier

6. Implement `model/scripts/analyze_errors.py` to generate the above reproducibly, balancing transform families and severities across selected cases rather than cherry-picking visually dramatic failures.
7. Add the manual tag vocabulary: `anatomical`, `stylistic`, `functional`, `physics_violation`, `sociocultural`, `none_visible`, and `uncertain`. Allow multiple tags, requiring a free-text rationale and reviewer initials.
8. Extend the note to generator/domain shift, low-resolution shortcuts, and calibration limitations, drawing on the Stage 07 cross-dataset and held-out tables.
9. Implement ViT gradient-based patch saliency or attention rollout for a small fixed subset. Label every visualization "qualitative, not causal." Do not use Grad-CAM APIs designed only for CNN feature maps, and never let saliency block this stage.
10. Ensure report-ready images are team-created, appropriately licensed, or omitted from the public repository and demo.

## Markdown report

Write the required note to `report/error_analysis.md`. Include recurring failure patterns, authentic-image false-positive harms, transformation sensitivity, pixel-only provenance limitations, reviewed case IDs, licensing status, completed tier, and repository-relative links to metadata and plots. Do not place raw dataset images or absolute local paths in the report.

## Tests and checks

- No raw dataset image becomes tracked by Git. *(Required tier — verify before committing anything.)*
- Every discussed example has metadata and no absolute local path. *(Required tier.)*
- No training/validation examples in final test error galleries unless the report explicitly labels the split. *(Required tier.)*
- The written report distinguishes human-visible artifact categories from model-learned evidence. *(Required tier — a CLIP head does not "see" a six-fingered hand; do not claim it does.)*
- Ranking and flip classification against small known fixtures. *(Optional tier, with the script.)*

## Exit gate

Report error counts, representative metadata, licensing status, and completed tier; confirm `report/error_analysis.md` exists, then continue to Stage 09. Manual tags may be reviewed later and must not block the required note.
