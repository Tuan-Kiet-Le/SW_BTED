# Qwen3 version 10 audit report — 2026-08-24

## Execution

- Kernel: `lehoangtuankiet/sw-bted-qwen3-baseline`, version 10
- Model: `Qwen/Qwen3-Embedding-4B`
- Tokenizer: `Qwen2Tokenizer`
- Device/dtype: CUDA / `torch.bfloat16`
- Configured maximum: 2048 tokens
- Dataset: canonical 138-pair real-only benchmark; 178 unique documents

## Tokenizer audit

| Quantity | Value |
|---|---:|
| Documents over 2048 tokens | 18/178 (10.11%) |
| Median | 637 tokens |
| P95 | 2504.2 tokens |
| Maximum | 4156 tokens |

## Metric check

The rerun reproduces the existing Qwen3 result exactly at reported precision:

- Mean-fold F1: `0.9867 ± 0.0267`
- Precision: `0.9750`
- Recall: `1.0000`
- ROC-AUC: `1.0000`
- Fold thresholds: `0.66, 0.66, 0.66, 0.63, 0.66`

## Interpretation

Qwen3 is a modern baseline, but this result is specifically a 2048-token truncation protocol. It must not be described as a full-context or truncation-free comparison. The structural conclusion is unchanged: Qwen3 ties the natural-benchmark aggregate result while SW-BTED contributes explicit structural attribution and controlled structural diagnostics.

## Artifacts

- `kaggle/qwen3_results_v10/qwen3_results/qwen3_input_length_audit.json`
- `kaggle/qwen3_results_v10/qwen3_results/qwen3_metrics.json`
- `kaggle/qwen3_results_v10/qwen3_results/qwen3_pair_scores.csv`
