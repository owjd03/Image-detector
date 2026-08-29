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

No dataset has been downloaded during Stage 01.

