# Robust AI-Generated Image Detector

This project estimates whether an image is fully AI-generated. It uses a frozen
OpenAI CLIP ViT-L/14 vision backbone and a trained consistency-regularized MLP
classifier head. It is a hackathon/research prototype, not proof of image
provenance.

There are two ways to use this repository:

1. [Test the pretrained detector](#quick-start-test-the-pretrained-detector).
2. [Download the data and train from scratch](#train-and-evaluate-from-scratch).

## Prerequisites

- Git and Python 3.12.
- An NVIDIA GPU is recommended. This project was developed on an RTX 5080 with
  32 GB of system RAM and PyTorch CUDA 12.8 wheels.
- CPU inference is supported but slower. Full embedding extraction is best done
  on a CUDA GPU.
- Approximately 10–15 GB for the sampled SID dataset, plus additional free
  space for the Hugging Face cache and generated embeddings. Keeping at least
  30 GB free is recommended.
- Internet access for initial dependency, CLIP, and dataset downloads.
- A Kaggle account and API credentials to download CIFAKE.
- A Hugging Face account/token if SID_Set requires authentication for your
  account. Never commit Kaggle or Hugging Face credentials.

## Quick start: test the pretrained detector

### 1. Clone the repository

PowerShell:

```powershell
git clone https://github.com/owjd03/Image-detector.git
Set-Location Image-detector
```

Bash:

```bash
git clone https://github.com/owjd03/Image-detector.git
cd Image-detector
```

### 2. Create the Python environment

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bash:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

The CUDA command is for compatible NVIDIA systems. CPU and macOS users should
install the appropriate PyTorch 2.8.0 build for their platform, then install
`requirements.txt`.

### 3. Download and verify the pinned CLIP backbone

The following command downloads `openai/clip-vit-large-patch14` at the immutable
revision recorded in `model/configs/default.yaml`. Files are stored under
`.cache/huggingface/` and are ignored by Git.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.download_model --config model/configs/default.yaml
.\.venv\Scripts\python.exe -m model.scripts.download_model --verify-only --offline
```

Bash:

```bash
.venv/bin/python -m model.scripts.download_model --config model/configs/default.yaml
.venv/bin/python -m model.scripts.download_model --verify-only --offline
```

The first command accesses Hugging Face. The second command disables network
fallback and proves that the pinned model is complete in the local cache.

### 4. Install the trained classifier artifacts

Inference needs all three files below. The `.pt` file contains only the small
trained classifier head; CLIP remains in the Hugging Face cache.

```text
model-bundle.zip
├── checkpoint.pt
├── metadata.json
└── frozen_model.json
```

Expected repository locations after extraction:

```text
model/outputs/final/checkpoint.pt
model/outputs/final/metadata.json
model/outputs/final/frozen_model.json
```

The commands below use the latest GitHub Release asset. They will work only
after `model-bundle.zip` has been published at
`https://github.com/owjd03/Image-detector/releases`.

PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/owjd03/Image-detector/releases/latest/download/model-bundle.zip" -OutFile model-bundle.zip
Expand-Archive -LiteralPath model-bundle.zip -DestinationPath model-bundle -Force
New-Item -ItemType Directory -Force model/outputs/final | Out-Null
Copy-Item -LiteralPath model-bundle/checkpoint.pt, model-bundle/metadata.json, model-bundle/frozen_model.json -Destination model/outputs/final
```

Bash:

```bash
curl -L https://github.com/owjd03/Image-detector/releases/latest/download/model-bundle.zip -o model-bundle.zip
unzip -o model-bundle.zip -d model-bundle
mkdir -p model/outputs/final
cp model-bundle/checkpoint.pt model-bundle/metadata.json model-bundle/frozen_model.json model/outputs/final/
```

Verify the checkpoint fingerprint. The expected SHA-256 is
`0090b39eac281125d2de72921dcc0f182b1656ed6fcca232b5385276a6301088`.

PowerShell:

```powershell
(Get-FileHash model/outputs/final/checkpoint.pt -Algorithm SHA256).Hash.ToLower()
```

Bash:

```bash
sha256sum model/outputs/final/checkpoint.pt
```

Do not use the checkpoint if its fingerprint differs.

### 5. Run the tests

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Bash:

```bash
.venv/bin/python -m pytest -q
```

### 6. Test your own images

Create an `images/` directory and place static `.jpg`, `.jpeg`, `.png`, or
`.webp` files inside it. Subdirectories are supported.

PowerShell:

```powershell
New-Item -ItemType Directory -Force images | Out-Null
.\.venv\Scripts\python.exe model/scripts/predict.py --input_dir images --output predictions.json --device auto --batch-size 32
Get-Content predictions.json
```

Bash:

```bash
mkdir -p images
.venv/bin/python model/scripts/predict.py --input_dir images --output predictions.json --device auto --batch-size 32
cat predictions.json
```

The output contains exactly one object per readable image:

```json
[{"image_path":"nested/example.jpg","pred":0.7312}]
```

`pred` is the calibrated probability assigned to the synthetic class. The
frozen verdict threshold is `0.6069634556770325`, but the CLI deliberately
returns probabilities rather than replacing them with labels. Unreadable files
are skipped and recorded separately in `predictions.errors.json`.

## Train and evaluate from scratch

The commands in this section reproduce the completed training pipeline. Run
them from the repository root after creating the environment above.

### Dataset design

- SID Tier A supplies the actual training data: 5,000 real and 5,000 fully
  synthetic training sources.
- Each training source produces one clean embedding and two transformed
  embeddings, giving 30,000 training embeddings in total.
- SID calibration/validation uses 1,000 real and 1,000 synthetic sources.
- SID internal final evaluation uses another 1,000 real and 1,000 synthetic
  sources. SID_Set currently exposes no public official test split, so this is
  an internal evaluation drawn without overlap from its validation split.
- Up to 250 partially tampered SID records are diagnostic only and are excluded
  from the binary training task.
- CIFAKE is used only to measure cross-dataset generalization. It is never used
  for training, checkpoint selection, calibration, or threshold selection.
- WildFake is intentionally omitted because the exact challenge-supplied subset
  was unavailable.

### 1. Download and verify CLIP

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.download_model --config model/configs/default.yaml
.\.venv\Scripts\python.exe -m model.scripts.download_model --verify-only --offline
```

Bash:

```bash
.venv/bin/python -m model.scripts.download_model --config model/configs/default.yaml
.venv/bin/python -m model.scripts.download_model --verify-only --offline
```

### 2. Configure dataset credentials

For CIFAKE, download `kaggle.json` from the Kaggle account settings page. Store
it outside this repository at `%USERPROFILE%\.kaggle\kaggle.json` on Windows or
`~/.kaggle/kaggle.json` on Linux/macOS. On Linux/macOS, restrict it with:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

If Hugging Face requests authentication for SID_Set, log in without placing the
token in the repository:

```text
huggingface-cli login
```

### 3. Download and verify Tier A datasets

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset sid --tier a --train-per-class 5000 --calibration-per-class 1000 --evaluation-per-class 1000 --tampered-limit 250
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset sid --verify-only
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset cifake --tier a
```

Bash:

```bash
.venv/bin/python -m model.scripts.download_datasets --dataset sid --tier a --train-per-class 5000 --calibration-per-class 1000 --evaluation-per-class 1000 --tampered-limit 250
.venv/bin/python -m model.scripts.download_datasets --dataset sid --verify-only
.venv/bin/python -m model.scripts.download_datasets --dataset cifake --tier a
```

If an SID download is interrupted, repeat it with `--resume`; the script does
not delete valid completed files. Dataset files are placed under
`resources/datasets/` and remain ignored by Git. Provenance and counts are
recorded in `resources/datasets/dataset_inventory.json`.

### 4. Build manifests and transform plans

A manifest is a compact index that records each source image's location, label,
split, hash, and intended role. It lets the pipeline detect cross-split leakage
without copying every image into another directory.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.build_manifests --config model/configs/default.yaml
```

Bash:

```bash
.venv/bin/python -m model.scripts.build_manifests --config model/configs/default.yaml
```

Review `model/outputs/manifests/leakage_report.json` before continuing. Exact
cross-split duplicates are a hard failure; near-duplicate findings require
review rather than automatic deletion.

### 5. Extract CLIP embeddings

An embedding is a 768-number representation of an image produced by frozen
CLIP. These commands process SID Parquet files in bounded chunks so the complete
image columns are not cached in the 32 GB system RAM.

First run a small preflight:

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split train --dry-run 64 --device auto
```

Bash:

```bash
.venv/bin/python -m model.scripts.extract_embeddings --split train --dry-run 64 --device auto
```

Then extract every required cache sequentially:

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split train --device cuda --batch-size 32 --workers 4
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split val --conditions all --device cuda --batch-size 32 --workers 4
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split test --conditions all --device cuda --batch-size 32 --workers 4
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split tampered --conditions all --device cuda --batch-size 32 --workers 4
.\.venv\Scripts\python.exe -m model.scripts.extract_embeddings --split cifake --conditions clean --device cuda --batch-size 32 --workers 4
```

Bash:

```bash
.venv/bin/python -m model.scripts.extract_embeddings --split train --device cuda --batch-size 32 --workers 4
.venv/bin/python -m model.scripts.extract_embeddings --split val --conditions all --device cuda --batch-size 32 --workers 4
.venv/bin/python -m model.scripts.extract_embeddings --split test --conditions all --device cuda --batch-size 32 --workers 4
.venv/bin/python -m model.scripts.extract_embeddings --split tampered --conditions all --device cuda --batch-size 32 --workers 4
.venv/bin/python -m model.scripts.extract_embeddings --split cifake --conditions clean --device cuda --batch-size 32 --workers 4
```

Batch size 32 was stable on the RTX 5080. If CUDA runs out of memory, halve
`--batch-size` and rerun; extraction resumes from completed shards. Use
`--device cpu` only when CUDA is unavailable.

Expected cache counts are 30,000 train, 46,000 validation, 46,000 internal
evaluation, 5,750 tampered-condition, and 20,000 CIFAKE embeddings.

### 6. Train the classifier heads

CLIP is not loaded during this stage. Only small classifiers are trained over
the cached embeddings.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.train_heads --experiment core --seeds 42 --device cuda
.\.venv\Scripts\python.exe -m model.scripts.train_heads --experiment mlp_robust,linear_robust --seeds 43,44 --device cuda
```

Bash:

```bash
.venv/bin/python -m model.scripts.train_heads --experiment core --seeds 42 --device cuda
.venv/bin/python -m model.scripts.train_heads --experiment mlp_robust,linear_robust --seeds 43,44 --device cuda
```

The second command reproduces the two extra-seed finalists selected during this
project's run. Selection uses SID validation only. Training histories,
checkpoints, metadata, comparison tables, and plots are written under
`model/outputs/training/` and summarized in `report/training_report.md`.

### 7. Calibrate, review, freeze, and evaluate

Calibration converts raw classifier scores into probabilities and chooses a
decision threshold using SID validation data only.

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.calibrate --experiment all --device cuda
```

Bash:

```bash
.venv/bin/python -m model.scripts.calibrate --experiment all --device cuda
```

Before final evaluation, review the proposed experiment, seed, temperature, and
threshold in `model/outputs/evaluation/proposal.json`. Do not choose a model
using test, held-out, tampered, or CIFAKE results.

Once the validation-only proposal is accepted, freeze it and run all final
evaluation scopes:

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.evaluate --scope all --frozen --device cuda
```

Bash:

```bash
.venv/bin/python -m model.scripts.evaluate --scope all --frozen --device cuda
```

The evaluation command copies the selected `checkpoint.pt` and `metadata.json`
into `model/outputs/final/`, verifies the checkpoint SHA-256, and writes the
matching `frozen_model.json` there. The original training-run artifacts remain
available for provenance. Machine-readable results are written under
`model/outputs/evaluation/`. Do not tune any parameter after viewing
final-evaluation or CIFAKE results.

### 8. Run final verification

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe model/scripts/predict.py --input_dir images --output predictions.json --device auto
```

Bash:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python model/scripts/predict.py --input_dir images --output predictions.json --device auto
```

## Results and reports

- [Training report](report/training_report.md)
- [Calibration and evaluation report](report/evaluation_report.md)
- [Error-analysis report](report/error_analysis.md)
- [CLI verification report](report/cli_report.md)
- [Dataset documentation](resources/README.md)

The selected frozen model is `mlp_consistency`, seed 42. On the internal SID
evaluation it achieved 0.9925 clean balanced accuracy and 0.9855 worst-condition
balanced accuracy. CIFAKE balanced accuracy was only 0.5706, demonstrating that
strong in-dataset results do not guarantee broad real-world generalization.

## Limitations and responsible use

- The detector distinguishes fully synthetic from authentic images; it is not
  trained to localize edits or reliably classify partially manipulated images.
- It generalizes poorly from SID to CIFAKE and may also fail on generators,
  cameras, editing pipelines, and compression patterns absent from training.
- CIFAKE contains 32×32 images that must be upscaled for CLIP, so its results are
  not directly comparable to SID.
- A high or low probability is not proof of provenance. Do not use this model as
  the sole basis for moderation, accusations, disciplinary action, or other
  high-impact decisions.
