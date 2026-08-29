# Data resources

Raw datasets and demo samples belong under `resources/` and are ignored by Git.
Stage 03 will provide reproducible download and verification commands.

| Dataset | Source | Intended role | License / attribution |
|---|---|---|---|
| CIFAKE | https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images | Pipeline smoke test only | Verify and cite the dataset card and Bird & Lotfi publication before use |
| SID_Set | https://huggingface.co/datasets/saberzl/SID_Set | Core train, validation, and test data; labels 0 and 1 only | Dataset card reports CC BY 4.0; preserve upstream attributions |
| WildFake subset | https://modelscope.cn/datasets/hy2628982280/WildFake/summary | External demonstration benchmark only | Verify upstream and constituent-image terms before use |

## Credentials

CIFAKE requires Kaggle API credentials. Supply credentials through Kaggle's
standard user configuration or environment mechanism; never place them in this
repository. Hugging Face or ModelScope tokens, if required, must likewise stay
outside tracked files.

## Non-negotiable isolation rule

The challenge-provided WildFake subset must never be used for training,
checkpoint selection, hyperparameter tuning, confidence calibration, or verdict
threshold selection. Later pipeline stages must encode and test this restriction,
not rely on memory or convention.

## Stage 03 commands

Tier A is pinned to SID_Set revision
`dc03ead57929879319ce30a82bfcfb8d317b10bd` and seed 42. It selects 5,000
records per binary class for training, 1,000 per class for calibration, 1,000
per class for internal final evaluation, and up to 250 tampered diagnostics.
The latter two binary roles remain children of the upstream validation split;
SID_Set currently exposes no public official test split.

```powershell
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset sid --tier a
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset sid --verify-only
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset sid --tier a --resume
```

CIFAKE is downloaded only through Kaggle's official API. Configure
`%USERPROFILE%\.kaggle\kaggle.json` using Kaggle's account settings, then run:

```powershell
.\.venv\Scripts\python.exe -m model.scripts.download_datasets --dataset cifake --tier a
```

WildFake is omitted from the current plan because the exact organizer subset is
not available. It is not downloaded and does not influence training, selection,
calibration, evaluation, or thresholds. A future explicit plan change may add
the exact challenge subset only as `external_demo_only`.

Expected Tier A working space is 10–15 GB based on the measured preflight. SID_Set is CC BY 4.0 according to
its dataset card. CIFAKE's card states MIT and requires citation of CIFAR-10 and
Bird & Lotfi (2024). WildFake is omitted, so its component-specific licensing
was not evaluated for this run.

