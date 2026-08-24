# Qwen3-Embedding-4B Kaggle Execution Report

**Execution date:** 2026-08-24 (version 10; tokenizer-audit rerun)
**Kaggle kernel:** `lehoangtuankiet/sw-bted-qwen3-baseline`  
**Input dataset:** private `lehoangtuankiet/sw-bted-138`  
**Protocol:** canonical 138-pair real-only evaluation

## Result

Qwen3-Embedding-4B completed successfully on Kaggle GPU inference.

| Metric | Qwen3-Embedding-4B |
|---|---:|
| Pairs | 138 |
| Positive | 38 |
| Negative | 100 |
| 5-fold F1 | **0.9867 ± 0.0267** |
| Precision | 0.9750 |
| Recall | 1.0000 |
| ROC-AUC | 1.0000 |
| Fold thresholds | 0.66, 0.66, 0.66, 0.63, 0.66 |

This exactly matches the manuscript's current Full-Document SBERT result (`0.9867 ± 0.0267`) at the reported precision/recall level. Qwen3 does not improve the headline F1 on this small benchmark, but it is a stronger modern baseline and confirms that the result is not dependent on MiniLM alone.

## Score distributions

| Pair type | n | Mean cosine | Min | Max |
|---|---:|---:|---:|---:|
| Type_A | 38 | 0.8836 | 0.8208 | 0.9562 |
| Type_B | 50 | 0.5030 | 0.3795 | 0.6538 |
| Type_C | 50 | 0.4848 | 0.2819 | 0.6146 |

## Kaggle execution issues resolved

1. The first kernel runs failed because the private dataset was mounted under `/kaggle/input/datasets/lehoangtuankiet/sw-bted-138`, not the simple slug path. The runner was changed to discover `pairs.csv` recursively.
2. Kaggle initially assigned Tesla P100, while the installed PyTorch build did not support its compute capability. The kernel was rerun with `NvidiaTeslaT4`.
3. Qwen4B 4-bit loading required `bitsandbytes>=0.46.1`; the runner now installs it when absent.
4. The successful run used Qwen3-Embedding-4B with maximum length 2048 and batch size 1. The result metadata reports CUDA execution and the model dtype.

5. The version 10 input-length audit found 18/178 documents (10.11%) over the configured 2048-token cutoff; median length was 637, P95 was 2504.2, and maximum was 4156 tokens. The Qwen3 comparison must therefore be described as a 2048-token truncation protocol.

## Artifacts

- `kaggle/qwen3_results/qwen3_results/qwen3_metrics.json`
- `kaggle/qwen3_results/qwen3_results/qwen3_pair_scores.csv`
- `kaggle/qwen3_results/qwen3_results/qwen3_document_embeddings.npz`
- `kaggle/qwen3_results/qwen3_results/provenance_manifest.json`
- `kaggle/qwen3_results_v10/qwen3_results/qwen3_input_length_audit.json`
- `kaggle/run_qwen3_embedding_baseline.py`

## Manuscript implication

Replace the current modern-baseline TODO with a reported Qwen3-Embedding-4B row. The correct claim is parity, not superiority: Qwen3 and MiniLM both obtain F1 `0.9867 ± 0.0267` under the same 138-pair protocol. The method's value therefore remains structural sensitivity, edit-trace interpretability and the controlled structural-perturbation result, rather than beating the strongest flat embedding numerically.
