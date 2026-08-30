# Stage 7 Evaluation Report

Status: final configuration frozen; no post-evaluation tuning occurred.

## Frozen model

- Experiment: `mlp_consistency`
- Seed: `42`
- Checkpoint: `model/outputs/final/checkpoint.pt`
- Checkpoint SHA-256: `0090b39eac281125d2de72921dcc0f182b1656ed6fcca232b5385276a6301088`
- Temperature: `0.329799`
- Threshold: `0.606963`

Selection used SID validation clean plus graded conditions only. Held-out, SID test, CIFAKE, WildFake, and tampered data were report-only.

## Core trade-offs

| Scope | Balanced accuracy | Notes |
|---|---:|---|
| SID test clean | 0.9925 | In-domain clean baseline |
| SID test graded worst | 0.9855 | Worst of 19 trained transformation conditions |
| SID test graded macro | 0.9907 | Mean across 19 graded conditions |
| SID test held-out worst | 0.9790 | Never used for training or selection |
| SID test held-out macro | 0.9863 | WebP, composition, and out-of-range JPEG |
| CIFAKE clean | 0.5706 | Cross-dataset; 32x32 inputs upscaled |

Per-condition authentic FPR and TPR at 1%/5% FPR appear in the complete tables below.

## Validation graded conditions

| Condition | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Authentic FPR | Brier | ECE | TPR@1% FPR | TPR@5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur_s0.5 | 2000 | 0.9955 | 0.9955 | 0.9970 | 0.9940 | 0.9955 | 0.9998 | 0.9998 | 0.0030 | 0.0043 | 0.0047 | 0.9950 | 0.9980 |
| blur_s1.0 | 2000 | 0.9955 | 0.9955 | 0.9990 | 0.9920 | 0.9955 | 0.9998 | 0.9998 | 0.0010 | 0.0039 | 0.0034 | 0.9950 | 0.9980 |
| blur_s2.0 | 2000 | 0.9910 | 0.9910 | 0.9970 | 0.9850 | 0.9909 | 0.9996 | 0.9996 | 0.0030 | 0.0061 | 0.0058 | 0.9950 | 0.9990 |
| brightness_x0.8 | 2000 | 0.9960 | 0.9960 | 0.9970 | 0.9950 | 0.9960 | 0.9998 | 0.9998 | 0.0030 | 0.0037 | 0.0031 | 0.9960 | 0.9980 |
| brightness_x1.2 | 2000 | 0.9935 | 0.9935 | 0.9970 | 0.9900 | 0.9935 | 0.9998 | 0.9998 | 0.0030 | 0.0054 | 0.0034 | 0.9960 | 0.9990 |
| center_crop_0.8 | 2000 | 0.9900 | 0.9900 | 0.9990 | 0.9810 | 0.9899 | 0.9981 | 0.9961 | 0.0010 | 0.0072 | 0.0081 | 0.9950 | 0.9970 |
| clean | 2000 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9998 | 0.9998 | 0.0050 | 0.0048 | 0.0039 | 0.9950 | 0.9980 |
| contrast_x0.8 | 2000 | 0.9955 | 0.9955 | 0.9960 | 0.9950 | 0.9955 | 0.9998 | 0.9998 | 0.0040 | 0.0041 | 0.0033 | 0.9960 | 0.9980 |
| contrast_x1.2 | 2000 | 0.9915 | 0.9915 | 0.9940 | 0.9890 | 0.9915 | 0.9997 | 0.9998 | 0.0060 | 0.0067 | 0.0054 | 0.9960 | 0.9990 |
| jpeg_q30 | 2000 | 0.9925 | 0.9925 | 0.9950 | 0.9900 | 0.9925 | 0.9997 | 0.9997 | 0.0050 | 0.0061 | 0.0068 | 0.9950 | 0.9980 |
| jpeg_q50 | 2000 | 0.9950 | 0.9950 | 0.9980 | 0.9920 | 0.9950 | 0.9998 | 0.9998 | 0.0020 | 0.0045 | 0.0038 | 0.9960 | 0.9990 |
| jpeg_q70 | 2000 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9995 | 0.9996 | 0.0050 | 0.0042 | 0.0024 | 0.9960 | 0.9980 |
| jpeg_q90 | 2000 | 0.9945 | 0.9945 | 0.9950 | 0.9940 | 0.9945 | 0.9996 | 0.9997 | 0.0050 | 0.0054 | 0.0030 | 0.9940 | 0.9980 |
| noise_s0.02 | 2000 | 0.9915 | 0.9915 | 0.9970 | 0.9860 | 0.9915 | 0.9994 | 0.9995 | 0.0030 | 0.0068 | 0.0051 | 0.9940 | 0.9980 |
| noise_s0.05 | 2000 | 0.9915 | 0.9915 | 0.9950 | 0.9880 | 0.9915 | 0.9995 | 0.9996 | 0.0050 | 0.0072 | 0.0040 | 0.9920 | 0.9970 |
| noise_s0.10 | 2000 | 0.9900 | 0.9900 | 0.9970 | 0.9830 | 0.9899 | 0.9995 | 0.9995 | 0.0030 | 0.0089 | 0.0076 | 0.9920 | 0.9970 |
| resize_0.25 | 2000 | 0.9940 | 0.9940 | 0.9930 | 0.9950 | 0.9940 | 0.9997 | 0.9998 | 0.0070 | 0.0049 | 0.0038 | 0.9960 | 0.9980 |
| resize_0.5 | 2000 | 0.9955 | 0.9955 | 0.9960 | 0.9950 | 0.9955 | 0.9997 | 0.9998 | 0.0040 | 0.0047 | 0.0047 | 0.9950 | 0.9980 |
| saturation_x0.8 | 2000 | 0.9955 | 0.9955 | 0.9960 | 0.9950 | 0.9955 | 0.9997 | 0.9997 | 0.0040 | 0.0047 | 0.0044 | 0.9950 | 0.9980 |
| saturation_x1.2 | 2000 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9997 | 0.9997 | 0.0050 | 0.0048 | 0.0034 | 0.9950 | 0.9980 |

## Validation held-out conditions

| Condition | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Authentic FPR | Brier | ECE | TPR@1% FPR | TPR@5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout_jpeg_q20 | 2000 | 0.9910 | 0.9910 | 0.9970 | 0.9850 | 0.9909 | 0.9996 | 0.9996 | 0.0030 | 0.0076 | 0.0073 | 0.9910 | 0.9980 |
| heldout_jpeg_q50_resize_0.5 | 2000 | 0.9945 | 0.9945 | 0.9980 | 0.9910 | 0.9945 | 0.9997 | 0.9997 | 0.0020 | 0.0045 | 0.0036 | 0.9950 | 0.9990 |
| heldout_webp_q50 | 2000 | 0.9780 | 0.9780 | 0.9668 | 0.9900 | 0.9783 | 0.9984 | 0.9985 | 0.0340 | 0.0195 | 0.0166 | 0.9650 | 0.9950 |

## SID final evaluation

| Condition | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Authentic FPR | Brier | ECE | TPR@1% FPR | TPR@5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur_s0.5 | 2000 | 0.9920 | 0.9920 | 0.9940 | 0.9900 | 0.9920 | 0.9998 | 0.9998 | 0.0060 | 0.0064 | 0.0037 | 0.9940 | 1.0000 |
| blur_s1.0 | 2000 | 0.9915 | 0.9915 | 0.9940 | 0.9890 | 0.9915 | 0.9998 | 0.9998 | 0.0060 | 0.0064 | 0.0042 | 0.9940 | 1.0000 |
| blur_s2.0 | 2000 | 0.9900 | 0.9900 | 0.9949 | 0.9850 | 0.9899 | 0.9997 | 0.9998 | 0.0050 | 0.0077 | 0.0058 | 0.9950 | 1.0000 |
| brightness_x0.8 | 2000 | 0.9945 | 0.9945 | 0.9950 | 0.9940 | 0.9945 | 0.9998 | 0.9998 | 0.0050 | 0.0051 | 0.0033 | 0.9950 | 0.9990 |
| brightness_x1.2 | 2000 | 0.9925 | 0.9925 | 0.9950 | 0.9900 | 0.9925 | 0.9998 | 0.9998 | 0.0050 | 0.0063 | 0.0042 | 0.9960 | 1.0000 |
| center_crop_0.8 | 2000 | 0.9885 | 0.9885 | 0.9980 | 0.9790 | 0.9884 | 0.9996 | 0.9997 | 0.0020 | 0.0093 | 0.0096 | 0.9970 | 0.9980 |
| clean | 2000 | 0.9925 | 0.9925 | 0.9950 | 0.9900 | 0.9925 | 0.9998 | 0.9998 | 0.0050 | 0.0063 | 0.0031 | 0.9930 | 1.0000 |
| contrast_x0.8 | 2000 | 0.9930 | 0.9930 | 0.9950 | 0.9910 | 0.9930 | 0.9998 | 0.9998 | 0.0050 | 0.0057 | 0.0034 | 0.9950 | 1.0000 |
| contrast_x1.2 | 2000 | 0.9900 | 0.9900 | 0.9940 | 0.9860 | 0.9900 | 0.9997 | 0.9997 | 0.0060 | 0.0074 | 0.0042 | 0.9940 | 1.0000 |
| heldout_jpeg_q20 | 2000 | 0.9890 | 0.9890 | 0.9939 | 0.9840 | 0.9889 | 0.9992 | 0.9992 | 0.0060 | 0.0087 | 0.0061 | 0.9940 | 1.0000 |
| heldout_jpeg_q50_resize_0.5 | 2000 | 0.9910 | 0.9910 | 0.9940 | 0.9880 | 0.9910 | 0.9998 | 0.9998 | 0.0060 | 0.0072 | 0.0047 | 0.9960 | 1.0000 |
| heldout_webp_q50 | 2000 | 0.9790 | 0.9790 | 0.9650 | 0.9940 | 0.9793 | 0.9984 | 0.9982 | 0.0360 | 0.0178 | 0.0164 | 0.9790 | 0.9960 |
| jpeg_q30 | 2000 | 0.9895 | 0.9895 | 0.9939 | 0.9850 | 0.9895 | 0.9993 | 0.9991 | 0.0060 | 0.0074 | 0.0051 | 0.9960 | 1.0000 |
| jpeg_q50 | 2000 | 0.9915 | 0.9915 | 0.9940 | 0.9890 | 0.9915 | 0.9998 | 0.9998 | 0.0060 | 0.0069 | 0.0050 | 0.9960 | 1.0000 |
| jpeg_q70 | 2000 | 0.9920 | 0.9920 | 0.9940 | 0.9900 | 0.9920 | 0.9998 | 0.9998 | 0.0060 | 0.0061 | 0.0035 | 0.9970 | 1.0000 |
| jpeg_q90 | 2000 | 0.9910 | 0.9910 | 0.9930 | 0.9890 | 0.9910 | 0.9997 | 0.9997 | 0.0070 | 0.0065 | 0.0031 | 0.9950 | 0.9990 |
| noise_s0.02 | 2000 | 0.9865 | 0.9865 | 0.9959 | 0.9770 | 0.9864 | 0.9996 | 0.9997 | 0.0040 | 0.0097 | 0.0081 | 0.9930 | 0.9990 |
| noise_s0.05 | 2000 | 0.9855 | 0.9855 | 0.9909 | 0.9800 | 0.9854 | 0.9994 | 0.9994 | 0.0090 | 0.0122 | 0.0092 | 0.9810 | 0.9990 |
| noise_s0.10 | 2000 | 0.9860 | 0.9860 | 0.9939 | 0.9780 | 0.9859 | 0.9995 | 0.9995 | 0.0060 | 0.0103 | 0.0075 | 0.9910 | 0.9990 |
| resize_0.25 | 2000 | 0.9935 | 0.9935 | 0.9950 | 0.9920 | 0.9935 | 0.9997 | 0.9998 | 0.0050 | 0.0062 | 0.0034 | 0.9940 | 1.0000 |
| resize_0.5 | 2000 | 0.9925 | 0.9925 | 0.9950 | 0.9900 | 0.9925 | 0.9998 | 0.9998 | 0.0050 | 0.0063 | 0.0035 | 0.9940 | 1.0000 |
| saturation_x0.8 | 2000 | 0.9910 | 0.9910 | 0.9930 | 0.9890 | 0.9910 | 0.9998 | 0.9998 | 0.0070 | 0.0070 | 0.0046 | 0.9960 | 1.0000 |
| saturation_x1.2 | 2000 | 0.9920 | 0.9920 | 0.9940 | 0.9900 | 0.9920 | 0.9998 | 0.9998 | 0.0060 | 0.0067 | 0.0041 | 0.9940 | 1.0000 |

## CIFAKE cross-dataset evaluation

| Condition | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Authentic FPR | Brier | ECE | TPR@1% FPR | TPR@5% FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 20000 | 0.5706 | 0.5706 | 0.8472 | 0.1724 | 0.2865 | 0.7524 | 0.7444 | 0.0311 | 0.3949 | 0.3963 | 0.0768 | 0.2422 |

## Exploratory tampered diagnostic

| Condition | N | Mean probability | Median | Above threshold |
|---|---:|---:|---:|---:|
| blur_s0.5 | 250 | 0.0706 | 0.0029 | 0.0640 |
| blur_s1.0 | 250 | 0.0610 | 0.0029 | 0.0520 |
| blur_s2.0 | 250 | 0.0406 | 0.0028 | 0.0400 |
| brightness_x0.8 | 250 | 0.0688 | 0.0029 | 0.0640 |
| brightness_x1.2 | 250 | 0.0638 | 0.0029 | 0.0480 |
| center_crop_0.8 | 250 | 0.0368 | 0.0029 | 0.0280 |
| clean | 250 | 0.0737 | 0.0029 | 0.0680 |
| contrast_x0.8 | 250 | 0.0691 | 0.0029 | 0.0600 |
| contrast_x1.2 | 250 | 0.0708 | 0.0029 | 0.0640 |
| heldout_jpeg_q20 | 250 | 0.0861 | 0.0029 | 0.0760 |
| heldout_jpeg_q50_resize_0.5 | 250 | 0.0517 | 0.0029 | 0.0400 |
| heldout_webp_q50 | 250 | 0.0896 | 0.0029 | 0.0720 |
| jpeg_q30 | 250 | 0.0864 | 0.0029 | 0.0760 |
| jpeg_q50 | 250 | 0.0516 | 0.0029 | 0.0400 |
| jpeg_q70 | 250 | 0.0664 | 0.0030 | 0.0560 |
| jpeg_q90 | 250 | 0.0721 | 0.0029 | 0.0640 |
| noise_s0.02 | 250 | 0.0580 | 0.0028 | 0.0520 |
| noise_s0.05 | 250 | 0.0763 | 0.0029 | 0.0680 |
| noise_s0.10 | 250 | 0.0809 | 0.0029 | 0.0680 |
| resize_0.25 | 250 | 0.0604 | 0.0029 | 0.0520 |
| resize_0.5 | 250 | 0.0699 | 0.0029 | 0.0640 |
| saturation_x0.8 | 250 | 0.0806 | 0.0030 | 0.0720 |
| saturation_x1.2 | 250 | 0.0730 | 0.0029 | 0.0720 |

## Paired robustness

- Transformed pairs: 44000
- Mean probability delta: -0.000874
- Mean absolute probability delta: 0.005554
- Clean-to-transformed correctness flips: 177

## Generalisation and limitations

CIFAKE is a non-scoring cross-dataset check. Its source images are 32x32 and are upscaled by the CLIP processor. The large SID-to-CIFAKE drop demonstrates that strong in-domain robustness does not guarantee cross-dataset generalisation.

WildFake: Not run; the user explicitly omitted this optional demonstration dataset.

Tampered images are outside the trained fully-synthetic-versus-authentic binary task and are reported only as score distributions.

## Plots and artifacts

- Reliability plot: `model/outputs/evaluation/final/reliability.png`
- Condition degradation plot: `model/outputs/evaluation/final/condition_balanced_accuracy.png`
- Clean confusion matrix: `model/outputs/evaluation/final/clean_confusion_matrix.png`
- Machine-readable predictions and metrics: `model/outputs/evaluation/final/`
