# Stage 07 — Calibration and Robustness Evaluation

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Calibrate model scores using validation data, freeze the final configuration, and produce clean/transformed evaluation artifacts without test-set adaptation.

## Required work

1. Implement temperature scaling over pooled clean and transformed SID validation logits. Optimize one positive scalar temperature per selected checkpoint using validation negative log-likelihood only.

   ```bash
   python -m model.scripts.calibrate --experiment all
   python -m model.scripts.evaluate --scope validation
   python -m model.scripts.evaluate --scope test --frozen
   python -m model.scripts.evaluate --scope wildfake --frozen
   ```

2. Choose the verdict threshold that maximizes balanced accuracy on the same pooled validation set. Resolve ties by the lower authentic-image false-positive rate, then the threshold closest to 0.5.
3. Save temperature and threshold beside checkpoint metadata. `pred` is always `sigmoid(logit / temperature)`; the threshold affects only the human-readable verdict.
4. Compare the four seed-42 core experiments and finalist extra-seed summaries on SID validation, then propose the final model by highest worst-condition balanced accuracy on the graded conditions, breaking ties by macro transformed balanced accuracy, authentic false-positive rate, then clean ROC-AUC. Stop for confirmation before freezing it.
5. Evaluate the frozen choices once on official SID test data, and separately on the non-scoring benchmarks: WildFake, and CIFAKE as a cross-dataset generalisation check. Never change a parameter after viewing any of these results.
6. For clean and every condition, report sample/class counts, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, authentic false-positive rate, confusion matrix, Brier score, and 10-bin expected calibration error.
7. Report the three trade-offs the challenge brief names explicitly, each as its own table:
   - **Robustness** — worst-condition and macro balanced accuracy across the graded transform grid, versus clean.
   - **Generalisation** — two separate comparisons. First, **graded versus held-out** conditions: worst-condition balanced accuracy on the transform families the model trained on, beside the WebP, composed, and out-of-range conditions it never saw. A model that holds on the first and collapses on the second memorised the augmentation bank rather than learning robustness, and the write-up must say so if that is what the numbers show. Second, **cross-dataset**: SID-trained performance evaluated on CIFAKE and WildFake, with CIFAKE's 32×32-upscaled-to-224 resolution caveat stated beside the numbers.
   - **False positives** — **TPR at 1% and at 5% authentic false-positive rate**, per condition, alongside the threshold-free metrics. A platform moderating at scale operates at a fixed false-positive budget, not at the balanced-accuracy optimum, so report the operating point a deployer would actually choose.
8. Report SID label 2 (`exploratory_tampered`) as a labelled diagnostic group: how the frozen model scores partially manipulated images, with the scope assumption stated plainly — this system scores fully synthetic versus authentic, and partial manipulation falls outside that binary. Never fold these records into the headline metrics.
9. Join transformed predictions to their clean `source_id` and report probability delta, correctness flips, and metric degradation. Stratified paired bootstrap intervals and extra plots are optional after the core tables exist.
10. Export long-form CSV/JSON, a compact Markdown table, confusion matrices, reliability plots, clean-versus-robust ablation plot, and severity-versus-degradation plots. Clearly label validation, SID test, held-out, CIFAKE, and WildFake scopes.

## Failure handling

Undefined metrics from a missing class must be explicit `null` plus a warning, never silently zero. Reject duplicate source-condition rows, uncalibrated outputs, or any WildFake reference in selection metadata.

## Isolation invariants

Selection, calibration, and thresholding read **graded-condition SID validation data only**. Held-out conditions, CIFAKE, WildFake, SID test, and tampered records are all report-only and are computed at the same freeze point, after the model is fixed. The held-out set proves nothing if it influenced the model that it is testing.

## Tests

- Known-fixture metric and threshold tests.
- Temperature must remain positive and improve or preserve validation NLL within tolerance.
- Paired joins must contain exactly one clean row per transformed source.
- A provenance test must prove no test, held-out, CIFAKE, or WildFake value is read by selection, calibration, or threshold functions.
- Fixed-FPR interpolation returns the correct TPR on a known ROC fixture, and reports `null` when the requested FPR is unreachable.

## Exit gate

Before final evaluation, present the proposed model identity, validation evidence, temperature, and threshold and stop for confirmation. After confirmation, freeze once, produce the core trade-off tables and available optional artifacts, declare that no post-evaluation tuning occurred, and continue to Stage 08.
