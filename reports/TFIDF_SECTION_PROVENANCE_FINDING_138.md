# TF-IDF and Section Cosine provenance finding

Date: 2026-08-14

## Source identified

The closest source producing the historical `0.4364` and `0.4081` values is:

`D:/FPT/Semester_8/RAG_Research/experiments/main_evaluation.py`

It defines:

- `get_fold_tfidf_similarity_for_set`
- `get_fold_section_cosine_similarity_for_set`

Both fit vectorizers on an inner training-document set and evaluate validation/test pairs. Thresholds are selected on the inner validation split over a 0.01 grid.

The historical audit output that records the resulting values is:

`D:/FPT/Semester_8/RAG_Research/results/audit/significance_report_v2.md`

## Scope mismatch

The historical `main_evaluation.py` loads:

```python
pairs = pd.read_csv("data/dataset/pairs.csv")
```

and initially operates on the branch's 180-pair file. It filters pairs by available tree keys, but it does not use the canonical explicit 138-pair filter based on `plag_regen_sections.json` in the shown loading path.

The clean suite uses the canonical 138-pair file and the explicit 138-pair document universe. Therefore the old values cannot yet be claimed as results under the current primary protocol.

## Reproduction attempts

| Protocol | TF-IDF F1 | Section Cosine F1 |
|---|---:|---:|
| Clean global corpus, canonical 138 pairs | 0.9867 ± 0.0267 | 0.6837 ± 0.0894 |
| Canonical 138, inner-train-only corpus, inner validation threshold | 0.9279 ± 0.0697 | 0.6180 ± 0.0565 |
| Historical audit value | 0.4364 ± 0.0162 | 0.4081 ± 0.0548 |

The inner-train-only reproduction is closer in protocol design but still does not reproduce the historical values. This confirms that at least one additional difference remains: branch dataset/document universe, tree/full-text version, section mapping, or an older baseline implementation.

## Current decision

The historical values `0.4364` and `0.4081` are provenance-traceable to an old evaluation family, but they are not yet reproducible under the canonical 138-pair input. They must not be silently presented as clean 138-pair results.

The clean current results should also not be inserted into the manuscript without deciding which baseline definition the paper intends to claim. The correct next step is an explicit scope-matched rerun of the historical `main_evaluation.py` on the 180-pair branch, followed by name-matched extraction of its real-only subset. That will determine whether the historical rows are genuinely from the 138 real-only pairs or are a 180-pair/branch artifact.

## Files generated in this investigation

- `experiments/clean_baseline_suite_138.py`
- `experiments/archive/reproduce_leak_free_b1_b5_138.py`
- `reports/audit/clean_baseline_suite_138.json`
- `reports/audit/clean_baseline_suite_pair_scores_138.csv`
- `reports/audit/leak_free_b1_b5_138.json`
- `reports/CLEAN_BASELINE_RECONCILIATION_138.md`
