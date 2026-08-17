# Final canonical results — 138 pairs

Date: 2026-08-14

Protocol: canonical 138-pair dataset, five-fold StratifiedKFold (`shuffle=True`, seed 42), threshold grid step `0.005`, train-fold-only threshold selection. Bootstrap uses 2,000 resamples with seed `20260814`. Prediction vectors were verified by `(doc_a, doc_b, label, type)`.

| Method | Mean fold F1 ± std | Pooled F1 | Precision | Recall | Bootstrap 95% CI for F1 |
|---|---:|---:|---:|---:|---|
| SW-BTED | 0.9498 ± 0.0253 | 0.9500 | 0.9048 | 1.0000 | [0.8923, 0.9895] |
| SBERT MiniLM | 0.9867 ± 0.0267 | 0.9870 | 0.9744 | 1.0000 | [0.9552, 1.0000] |
| BGE-small | 0.9882 ± 0.0235 | 0.9870 | 0.9744 | 1.0000 | [0.9538, 1.0000] |
| MPNet | 0.9882 ± 0.0235 | 0.9870 | 0.9744 | 1.0000 | [0.9538, 1.0000] |
| Qwen3-Embedding-4B | 0.9867* | 0.9870 | 0.9744 | 1.0000 | [0.9538, 1.0000] |
| TF-IDF | 0.9867 ± 0.0267 | 0.9870 | 0.9744 | 1.0000 | [0.9538, 1.0000] |
| Standard TED | 0.4364 ± 0.0162 | 0.4368 | 0.2794 | 1.0000 | [0.3394, 0.5269] |
| pq-Gram | 0.9479 ± 0.0478 | 0.9474 | 0.9474 | 0.9474 | [0.8889, 0.9885] |
| Section Cosine | 0.6837 ± 0.0894 | 0.6667 | 0.5000 | 1.0000 | [0.5591, 0.7581] |
| Genuine Flat Domain SBERT | 0.4314 ± 0.0160 | 0.4318 | 0.2754 | 1.0000 | [0.3373, 0.5241] |

`*` Qwen3 uses the downloaded Kaggle fold predictions; its mean-fold metrics were not included in the downloaded pair audit CSV, so only pooled OOF and bootstrap values are reported here.

## Holm-corrected McNemar tests versus SW-BTED

| Baseline | SW-only correct | Baseline-only correct | Raw p | Holm-adjusted p | Significant |
|---|---:|---:|---:|---:|---|
| SBERT MiniLM | 1 | 4 | 0.375 | 1.000 | No |
| BGE-small | 1 | 4 | 0.375 | 1.000 | No |
| MPNet | 1 | 4 | 0.375 | 1.000 | No |
| Qwen3 | 1 | 4 | 0.375 | 1.000 | No |
| TF-IDF | 1 | 4 | 0.375 | 1.000 | No |
| Standard TED | 96 | 2 | 3.06e-26 | 2.45e-25 | Yes |
| pq-Gram | 4 | 4 | 1.000 | 1.000 | No |
| Section Cosine | 37 | 3 | 1.95e-08 | 1.36e-07 | Yes |
| Genuine Flat Domain SBERT | 96 | 0 | 2.52e-29 | 2.27e-28 | Yes |

## Interpretation

The clean primary evidence supports structural superiority over Standard TED, Section Cosine, and Genuine Flat Domain SBERT, while showing statistical parity with the strong full-document embedding baselines, Qwen3, TF-IDF, and pq-Gram. The paper should not claim overall accuracy superiority.

## Artifacts

- `reports/audit/final_canonical_results_138.json`
- `reports/audit/final_canonical_predictions_138.csv`
- `experiments/final_canonical_results_138.py`
