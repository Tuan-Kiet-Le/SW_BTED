# Clean Ablation Report — 138 Real-Only Pairs

## Protocol

- 138 pairs: 38 Type_A positives, 100 Type_B/Type_C negatives.
- Current four-layer source.
- Structural-only mode: alpha = 1.0.
- Five-fold StratifiedKFold, shuffle=True, random_state=42.
- Threshold selected on each training fold using a 0.01 grid.

## Results

| Schedule | T2 | T3 | T4 | F1 mean | F1 std | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Documented | 0.0 | 0.9 | 0.8 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Current | 0.0 | 0.9 | 1.0 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Uniform | 0.0 | 0.5 | 0.5 | 0.4339 | 0.0135 | 0.2771 | 1.0000 |
| Content-heavy | 0.0 | 1.0 | 1.0 | 0.9498 | 0.0253 | 0.9056 | 1.0000 |
| Schema-heavy | 0.0 | 0.0 | 0.0 | 0.4339 | 0.0135 | 0.2771 | 1.0000 |

## Interpretation

The result strongly supports the importance of the T3 content term (`beta_T3=0.9`) on this benchmark. T4 beta has no effect on the reported classification result in the tested range, consistent with the earlier beta provenance audit. The result does not yet justify alpha selection because this run is structural-only; alpha sensitivity must be evaluated with document embeddings enabled in a separate hybrid ablation.

## Artifact

Raw machine-readable output: `reports/ablation_138_clean.json`  
Runner: `experiments/archive/ablation_138_clean.py`
