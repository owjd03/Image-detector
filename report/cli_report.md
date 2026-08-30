# Stage 9 CLI Inference Report

## Outcome

The directory-to-JSON inference CLI is complete and passed an offline end-to-end smoke test on both CUDA and CPU. The CUDA CLI output also matched a direct call to the shared `InferenceEngine` exactly.

This was an execution check, not an accuracy evaluation and not an input to model selection, calibration, or threshold tuning. WildFake was intentionally omitted from this project in Stage 3, so four existing CIFAKE test images were used instead: two real and two synthetic.

## Tested commands

```powershell
.venv\Scripts\python.exe model/scripts/predict.py --input_dir resources/samples/stage09_cli --output model/outputs/cli/predictions.json --device cuda --batch-size 4

.venv\Scripts\python.exe model/scripts/predict.py --input_dir resources/samples/stage09_cli --output model/outputs/cli/predictions_repeat.json --device cuda --batch-size 4

.venv\Scripts\python.exe model/scripts/predict.py --input_dir resources/samples/stage09_cli --output model/outputs/cli/predictions_cpu.json --device cpu --batch-size 2

.venv\Scripts\python.exe -m pytest -q
```

## Frozen model

- Backbone: `openai/clip-vit-large-patch14`
- Revision: `32bd64288804d66eefd0ccbe215aa642df71cc41`
- Classifier checkpoint: `model/outputs/final/checkpoint.pt`
- Checkpoint SHA-256: `0090b39eac281125d2de72921dcc0f182b1656ed6fcca232b5385276a6301088`
- Temperature: `0.32979855563148874`
- Decision threshold: `0.6069634556770325`

## End-to-end results

| Run | Device | Inputs | Readable | Errors | Reported inference time |
|---|---:|---:|---:|---:|---:|
| Primary | CUDA | 4 | 4 | 0 | 0.4571 s |
| CUDA repeat | CUDA | 4 | 4 | 0 | 0.3887 s |
| Portability | CPU | 4 | 4 | 0 | 1.0378 s |

The reported inference time measures image batching and prediction after model initialization. The command runs in a clean Python process and loads the backbone and classifier once per process, not once per image.

## Verification

- Offline status: passed. The inference loader used `offline=True` and the pinned CLIP revision from the local Hugging Face cache.
- Recursive sorted relative paths: passed.
- Output schema: passed; every prediction object contains exactly `image_path` and `pred`.
- Probability validity: passed; every `pred` is a finite JSON number in `[0,1]`.
- Same-device repeatability: passed; the two CUDA JSON outputs were byte-identical.
- Shared-engine consistency: passed; direct `InferenceEngine` probabilities matched the primary CLI output exactly, with maximum difference `0.0`.
- CPU portability: passed. The largest CPU-versus-CUDA probability difference was `0.0036576390266418457`, reflecting float32 CPU versus float16 CUDA computation.
- Error isolation: passed in automated tests; corrupt images are omitted from the prediction array and recorded in a separate `<output-stem>.errors.json` file.
- Full tests: `43 passed in 5.65s`.
- Root README: present with setup, usage, output schema, result links, and limitations.

## Artifacts

- [Primary CUDA predictions](../model/outputs/cli/predictions.json)
- [Repeated CUDA predictions](../model/outputs/cli/predictions_repeat.json)
- [CPU predictions](../model/outputs/cli/predictions_cpu.json)
- [Frozen model configuration](../model/outputs/final/frozen_model.json)
- [Frozen classifier checkpoint](../model/outputs/final/checkpoint.pt)
- [Frozen classifier metadata](../model/outputs/final/metadata.json)
- [CLI implementation](../model/scripts/predict.py)
- [Shared inference engine](../model/src/inference.py)

No error JSON was produced by the real smoke tests because all four inputs were readable. Error-file behavior is covered by the automated CLI tests.

## Interpretation

This stage proves that the packaged command works independently and deterministically. It does not establish CIFAKE accuracy. In this four-image sample, one synthetic image received a low synthetic probability, consistent with the substantially weaker CIFAKE generalization documented in the evaluation report.

## Consolidated runtime bundle verification

The selected runtime artifacts were subsequently consolidated under
`model/outputs/final/`. The original training-run checkpoint and metadata remain
unchanged for provenance.

- Final bundle files: `checkpoint.pt`, `metadata.json`, and `frozen_model.json`.
- Final checkpoint SHA-256: `0090b39eac281125d2de72921dcc0f182b1656ed6fcca232b5385276a6301088`.
- Source and packaged checkpoint hashes: identical.
- CUDA offline smoke test: 4 readable, 0 errors, 0.4041 seconds reported inference time.
- CPU offline smoke test: 4 readable, 0 errors, 1.0412 seconds reported inference time.
- Full tests after migration: `46 passed in 5.86s`.
- [CUDA bundle predictions](../model/outputs/cli/final_bundle_cuda.json)
- [CPU bundle predictions](../model/outputs/cli/final_bundle_cpu.json)
