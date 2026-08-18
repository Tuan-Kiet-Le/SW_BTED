# Clean baseline reconciliation — canonical 138 pairs

Protocol: same 138-pair input, five-fold StratifiedKFold (seed 42), 0.005 threshold grid, train-fold-only threshold selection.

| Method | Mean F1 | Std | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SW_BTED | 0.9498 | 0.0253 | 0.9056 | 1.0000 | 38 | 4 | 96 | 0 |
| SBERT_MiniLM | 0.9867 | 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 |
| BGE_Small_v1.5 | 0.9882 | 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 |
| MPNet_Base_v2 | 0.9882 | 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 |
| TF-IDF | 0.9867 | 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 |
| Standard TED | 0.4364 | 0.0162 | 0.2792 | 1.0000 | 38 | 98 | 2 | 0 |
| pq-Gram | 0.9479 | 0.0478 | 0.9528 | 0.9464 | 36 | 2 | 98 | 2 |
| Section Cosine | 0.6837 | 0.0894 | 0.5263 | 1.0000 | 38 | 38 | 62 | 0 |
| Genuine Flat Domain SBERT | 0.4314 | 0.0160 | 0.2751 | 1.0000 | 38 | 100 | 0 | 0 |

## McNemar exact tests vs SW-BTED

| Baseline | SW-only correct | Baseline-only correct | Discordant | Exact p-value |
|---|---:|---:|---:|---:|
| SBERT_MiniLM | 1 | 4 | 5 | 0.375 |
| BGE_Small_v1.5 | 1 | 4 | 5 | 0.375 |
| MPNet_Base_v2 | 1 | 4 | 5 | 0.375 |
| TF-IDF | 1 | 4 | 5 | 0.375 |
| Standard TED | 96 | 2 | 98 | 3.06204e-26 |
| pq-Gram | 4 | 4 | 8 | 1 |
| Section Cosine | 37 | 3 | 40 | 1.9465e-08 |
| Genuine Flat Domain SBERT | 96 | 0 | 96 | 2.52435e-29 |

The old manually anchored raw-vector artifact was not used.