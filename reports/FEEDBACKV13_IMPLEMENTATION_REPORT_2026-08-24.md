# Feedback v13 implementation report

Date: 2026-08-24

## Fold-local lexical rerun

The canonical 138-pair TF-IDF and Section Cosine baselines were rerun with:

- 5-fold `StratifiedKFold`, seed 42;
- vectorizers fitted separately within each outer fold on training-fold documents only;
- threshold selection on training-fold scores using the `0.005` grid;
- test-fold transformation using the corresponding training-fitted vectorizers.

Results:

| Method | Mean-fold F1 ± SD | Pooled F1 | Confusion matrix (TP, FP, TN, FN) |
|---|---:|---:|---|
| TF-IDF | 0.9867 ± 0.0267 | 0.9870 | (38, 1, 99, 0) |
| Section Cosine | 0.6837 ± 0.0894 | 0.6667 | (38, 38, 62, 0) |

Both fold-local prediction vectors are identical to the previous clean-suite prediction vectors: 0 differences out of 138 pairs for each method. Therefore Table 1 metrics and paired statistical conclusions remain unchanged. The new fold-local artifacts are now the canonical lexical provenance.

## Manuscript changes

- Removed the incorrect claim that SW-BTED has lower raw F1 than pq-Gram.
- Documented fold-local TF-IDF/Section Cosine fitting in Sections 4.3 and 5.1.
- Replaced `held-out training data` with `training portion of that fold`.
- Added the explicit `17/20 (85%)` trace between Section 5.3 and Table 3.
- Softened the negative taxonomy-search claim to describe the development process rather than make an absolute literature claim.
- Updated the scientific manifest to point to the fold-local lexical artifacts.

## Status

The TF-IDF/Section Cosine protocol issue is resolved without changing any reported metric. No further experiment is required. Remaining work is final release packaging: venue formatting, anonymization if needed, final commit/tag, repository-pointer update, and manifest/hash verification.
