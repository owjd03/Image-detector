# Stage 13 — Optional Frequency-Domain Stretch Experiment

## Current status

Skipped under the locked under-72-hour plan. Do not execute this stage. Preserve the rationale below as future-work material for the README and judge Q&A.

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and do not perform unapproved work outside this gated experiment.

## Entry gate

This experiment may be reconsidered only through a future explicit plan change after the complete submission and conditional web demo are finished.

## Objective

Test whether a lightweight frequency-domain feature branch materially improves worst-condition validation robustness without harming authentic-image false positives, latency, or core reliability.

## Why this is expected to fail, and why that is worth writing down

Some GAN and diffusion generator pipelines use repeated resampling or leave frequency-domain artifacts in fine detail. A DCT can expose such patterns, but the cue is generator-dependent and must not be described as universal.

The difficulty is that JPEG compression and downscaling work by *discarding* high-frequency content. That is their compression mechanism, not a side effect. So the fingerprint is weakest under precisely the transformations this challenge scores: `jpeg_q30`, `resize_0.25`, `blur_s2.0`. The retention rule below is likely to reject the branch, and `mlp_consistency` from Stage 06 already carries the robustness story.

Record this reasoning in the submission whether or not the experiment runs. "We considered frequency-domain features and here is why they degrade under exactly the transformations we are scored on" is a stronger answer to a judge than silence, and costs no compute.

## Required experiment

1. Derive deterministic frequency features from the same decoded RGB image before CLIP preprocessing. Use a fixed grayscale 2D DCT representation, log-magnitude scaling, removal of the DC component, and a documented low-dimensional radial-frequency summary. Do not train a second large vision backbone.
2. Normalize frequency features using statistics fitted on SID training data only. Save those statistics with the checkpoint.
3. Concatenate the frequency vector with the frozen 768-dimensional CLIP embedding and train a small MLP under the same robust augmentation bank, seeds, optimizer policy, and validation isolation as the core models.
4. Calibrate and evaluate through the existing Stage 07 interfaces. Do not edit test labels, conditions, or baseline results.
5. Compare worst-condition validation balanced accuracy, macro transformed balanced accuracy, authentic false-positive rate, parameter count, CLI/API latency, and memory against **both `mlp_robust` and `mlp_consistency`** — the latter is the incumbent robustness approach and is the bar to clear.

## Retention rule

Retain the branch in the submission only if it improves worst-condition validation balanced accuracy by at least 0.01 absolute, does not increase authentic false-positive rate by more than 0.01 absolute, and keeps warm API latency within 25% of the core model. Apply this rule before looking at test/WildFake results. Otherwise report it as an unsuccessful experiment and leave the core model selected.

## Further comparisons

Do not start partial CLIP fine-tuning or ResNet-50 unless the frequency experiment and all core regression tests are complete with more than six hours still available. If attempted, keep them comparison-only; they cannot replace the core model without the same predeclared validation rule and a complete rerun of calibration, CLI/API agreement, documentation, and audits.

## Verification and exit

- Test deterministic feature extraction, train-only normalization, checkpoint reload, and shared inference compatibility.
- Re-run all affected CLI/API tests.
- Record successful and unsuccessful results without selective omission.
- Stop with the retention-rule calculation, latency impact, regression results, and exact documentation changes.
