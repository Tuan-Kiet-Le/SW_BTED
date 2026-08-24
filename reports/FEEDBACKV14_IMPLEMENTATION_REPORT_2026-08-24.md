# Feedback v14 implementation report

Date: 2026-08-24

## Completed change

Added the fold-local lexical-baseline protocol directly to `reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md`:

- TF-IDF and Section Cosine vectorizers are fitted independently within each cross-validation fold.
- Only training-fold documents are used for fitting.
- Test-fold documents are transformed with the corresponding training-fitted vectorizers.
- Thresholds use training-fold scores and the `0.005` grid.
- The machine-readable artifact is `reports/audit/fold_local_lexical_suite_138.json`.

## Verification

- Fold-local TF-IDF result: `0.9867 ± 0.0267`, pooled `0.9870`.
- Fold-local Section Cosine result: `0.6837 ± 0.0894`, pooled `0.6667`.
- Both prediction vectors remain identical to the previous clean-suite vectors (`0/138` differences per method).
- `git diff --check` completed successfully; only normal line-ending warnings were emitted by Git.

## Release status

Scientific content and lexical-baseline provenance are frozen. Remaining work is final release packaging: commit/tag creation, repository-pointer update from `a291bcc`, final hash/path verification, and double-blind anonymization if required by the venue.
