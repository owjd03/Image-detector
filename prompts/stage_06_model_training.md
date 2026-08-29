# Stage 06 — Classifier-Head Training

## Working protocol

Explain each new technical term in plain language with a concrete example. State assumptions, preserve unrelated user work, prefer readable documented code, show commands and results, and follow the gate policy in `prompts/README.md`.

## Objective

Train four required seed-42 frozen-embedding experiments, then run extra seeds only for the two robust finalists. Select checkpoints without consulting final-evaluation or external benchmark results.

## Fixed experiments

1. `linear_clean`: `Linear(768,1)`, clean training embeddings only.
2. `linear_robust`: the same head, clean plus two augmented variants.
3. `mlp_robust`: `768→256→64→1`, ReLU after the first two layers, dropout 0.3, clean plus augmented variants.
4. `mlp_consistency`: the same MLP and augmented bank as `mlp_robust`, plus an augmentation-consistency penalty.

Run seed 42 for all four experiments. Select the best two robust candidates using validation data, then run seeds 43 and 44 only for those finalists. Use logits, `BCEWithLogitsLoss`, AdamW, deterministic data ordering, and class-balanced sampling only when imbalance exceeds 5%.

Keep `linear_clean` because it is the minimum clean-training baseline needed to show whether robustness augmentation helped. `mlp_clean` and the lambda-0.3 sensitivity run are optional and cut before required work.

## Augmentation consistency

Plain `BCEWithLogitsLoss` asks only whether each view was classified correctly. A model can score a clean image 0.95 and its `jpeg_q30` version 0.55, be counted correct on both, and still be one step of compression away from flipping. The consistency term makes disagreement between an image's own views directly costly, so the head is pushed toward features that survive the transformation rather than features that merely outlast it.

Implement it as follows:

- Build batches with a **grouped sampler**: a source's clean and augmented variants land in the same batch, identified by the `source_id` carried in the Stage 05 sidecar.
- Add a penalty on the divergence between the logits of views sharing a `source_id` — mean squared difference against the group's clean-view logit, averaged over groups.
- Total loss is `BCEWithLogitsLoss + lambda * consistency`. Document `lambda` in YAML with default `1.0`; a `0.3` sensitivity run is optional.
- The penalty applies to training views only. It never sees validation, test, held-out, or WildFake data.
- Groups that contain only a clean view contribute zero consistency loss, not a division-by-zero.

Everything else — architecture, seeds, optimizer policy, selection rule — is identical to `mlp_robust`, so the comparison isolates the loss term. `mlp_consistency` enters Stage 07 selection on exactly the same terms as the other four; it is not privileged.

## Required work

1. Implement reusable head definitions and `model/scripts/train_heads.py`. CLIP must not be loaded during head training.

   ```bash
   python -m model.scripts.train_heads --experiment core --seeds 42
   python -m model.scripts.train_heads --experiment <finalist-a>,<finalist-b> --seeds 43,44
   ```

2. Use fixed defaults documented in YAML: effective grouped batch size 1023 (341 complete three-view groups), maximum 100 epochs, learning rate `1e-3` for linear and `3e-4` for MLP, weight decay `1e-4`, and early-stopping patience 10. Clean-only loaders may use batch size 1024.
3. For clean experiments, monitor clean SID validation loss. For robust and consistency experiments, monitor mean loss across clean and transformed SID validation groups. Use worst-condition balanced accuracy to break checkpoints tied within `1e-4` loss. Monitor the **classification component only** — the consistency penalty is a training-time regulariser, and including it in the validation criterion would let a model look good by being uniformly and confidently wrong.
4. Do not tune against official SID test or WildFake. The training loader must reject their roles.
5. Save small `.pt` state dictionaries plus adjacent JSON metadata containing architecture, seed, cache fingerprint, epoch, metrics, code commit, and dependency versions. Do not serialize model objects or CLIP weights.
6. Write epoch histories and a consolidated comparison table. Select the best seed per experiment by its specified validation rule, while preserving all seed results.

## Tests

- Forward-shape, gradient, save/reload, and deterministic-repeat tests.
- Assert training batches contain only permitted SID roles/splits and the selected augmentation regime.
- Assert no CLIP parameters or test/WildFake identifiers occur in checkpoints.
- Overfit a tiny synthetic embedding dataset as a training-loop sanity check.
- Verify early stopping and tie-breaking with deterministic fixtures.
- Assert the grouped sampler places all of a `source_id`'s training views in one batch.
- Assert the consistency term is exactly zero when all views of every group produce identical logits, and strictly positive otherwise.
- Assert `lambda = 0` reproduces `mlp_robust` within tolerance — the cheapest proof that the term is wired correctly and nothing else changed.
- Assert single-view groups contribute zero rather than raising or producing `NaN`.

## Exit gate

Report the four seed-42 summaries, finalist seed variance, selected checkpoints, learning curves, optional sensitivity status, and validation-only selection proof, then continue to Stage 07 without running final evaluation.
