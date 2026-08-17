# Clean Beta Ablation — Frozen 0.005 Protocol

This is the protocol-aligned replacement for the earlier 0.01-grid ablation
summary. It uses 138 real-only pairs, five-fold stratification, seed 42, and
training-fold-only threshold selection on a 0.005 grid.

| Schedule | T2 | T3 | T4 | F1 mean | F1 std | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Documented | 0.0 | 0.9 | 0.8 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Current | 0.0 | 0.9 | 1.0 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Uniform | 0.0 | 0.5 | 0.5 | 0.4339 | 0.0135 | 0.2771 | 1.0000 |
| Content-heavy | 0.0 | 1.0 | 1.0 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Schema-heavy | 0.0 | 0.0 | 0.0 | 0.4339 | 0.0135 | 0.2771 | 1.0000 |

The result continues to support the importance of the T3 content term. T4
variation has no effect on the classification result in these tested settings.
The output is machine-readable in `reports/ablation_138_clean_005.json`.
