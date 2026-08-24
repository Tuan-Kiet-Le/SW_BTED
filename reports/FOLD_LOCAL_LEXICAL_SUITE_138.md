# Fold-local lexical baseline suite — canonical 138 pairs

Date: 2026-08-24

## Protocol

- Dataset: canonical 138-pair `pairs.csv`.
- Cross-validation: 5-fold `StratifiedKFold`, seed 42.
- TF-IDF and Section Cosine vectorizers: fitted independently within each outer fold using only documents in that fold's training portion.
- Threshold selection: training-fold scores only, grid `0.00, 0.005, ..., 1.00`.
- Test-fold documents are transformed with the vectorizer fitted on the corresponding training documents.
- Pair order: canonical `pairs.csv` order.

## Results

| Method | Mean-fold F1 ± SD | Pooled F1 | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.9867 ± 0.0267 | 0.9870 | 0.9744 | 1.0000 | 38 | 1 | 99 | 0 |
| Section Cosine | 0.6837 ± 0.0894 | 0.6667 | 0.5000 | 1.0000 | 38 | 38 | 62 | 0 |

The fold-local predictions are identical to the previous clean-suite predictions for both methods (0 differing predictions out of 138 for each baseline). Therefore the reported Table 1 metrics and paired McNemar outcomes remain unchanged, while the lexical baseline provenance is now explicitly leak-free with respect to fold-local TF-IDF fitting.

Machine-readable output: `reports/audit/fold_local_lexical_suite_138.json` and `reports/audit/fold_local_lexical_pair_scores_138.csv`.
