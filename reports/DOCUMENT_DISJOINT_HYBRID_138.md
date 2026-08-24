# Document-disjoint Hybrid audit — canonical 138 pairs

The audit uses the canonical pair order, full-document MiniLM cosine scores, alpha = 0.6, and structural beta = (0.0, 0.9, 0.8). Connected components keep documents out of both train and test groups. Thresholds are selected on training groups only using a 0.005 grid.

| Mean F1 | SD | Pooled F1 | MCC | Precision | Recall | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9200 | 0.1600 | 0.9048 | 0.8718 | 0.8261 | 1.0000 | 38 | 8 | 92 | 0 |

This is a robustness audit. It does not establish that alpha = 0.6 was selected without access to the benchmark labels; the historical provenance audit describes alpha as a fixed configuration.