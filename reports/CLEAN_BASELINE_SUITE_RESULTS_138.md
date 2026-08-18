# Clean baseline suite results — canonical 138 pairs

Date: 2026-08-14

The suite was rerun with the project `.venv`, using sequential Standard TED (to avoid Windows worker failures). Corpus-based TF-IDF and Section Cosine were fitted only on documents participating in the 138 pairs; the 42 excluded regen documents were not included in their IDF corpus.

## Results

| Method | Mean F1 ± std | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.9867 ± 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 |
| Standard TED | 0.4364 ± 0.0162 | 0.2792 | 1.0000 | 38 | 98 | 2 | 0 |
| pq-Gram | 0.9479 ± 0.0478 | 0.9528 | 0.9464 | 36 | 2 | 98 | 2 |
| Section Cosine | 0.6837 ± 0.0894 | 0.5263 | 1.0000 | 38 | 38 | 62 | 0 |
| Genuine Flat Domain SBERT | 0.4314 ± 0.0160 | 0.2751 | 1.0000 | 38 | 100 | 0 | 0 |

Protocol: 138 pairs, five-fold StratifiedKFold, shuffle=True, seed 42, threshold grid step 0.005, threshold selected on the training fold only.

## Comparison with manuscript

- Standard TED, pq-Gram and Genuine Flat Domain SBERT reproduce the manuscript values at reported precision.
- TF-IDF does **not** reproduce the manuscript's `0.4364`; clean implementation gives `0.9867`.
- Section Cosine does **not** reproduce the manuscript's `0.4081`; clean implementation gives `0.6837`.

This is not sufficient evidence to silently replace the manuscript values. It demonstrates that TF-IDF and Section Cosine still have unresolved source/protocol drift. The exact implementation, input scope, and score construction that produced `0.4364` and `0.4081` must be recovered or the manuscript must disclose and adopt the clean definitions.

## Files

- Harness: `experiments/clean_baseline_suite_138.py`
- Pair scores: `reports/audit/clean_baseline_suite_pair_scores_138.csv`
- Metrics/predictions: `reports/audit/clean_baseline_suite_138.json`
- Combined comparison and McNemar tests: `reports/CLEAN_BASELINE_RECONCILIATION_138.md`
