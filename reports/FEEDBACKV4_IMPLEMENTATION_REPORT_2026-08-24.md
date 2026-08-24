# Feedback v4 implementation report — 2026-08-24

## Completed

1. Audited Hybrid provenance. The historical configuration is alpha `0.6`, full-document MiniLM cosine, structural beta `(0.0, 0.9, 0.8)`, and train-fold threshold selection on a `0.005` grid. Alpha is fixed in the historical evaluator, so the paper now avoids presenting it as a universally optimized value.
2. Ran a document-disjoint Hybrid audit on the canonical 138 pairs. Result: mean F1 `0.9200 ± 0.1600`, pooled F1 `0.9048`, MCC `0.8718`, confusion matrix `TP=38, FP=8, TN=92, FN=0`.
3. Audited the observable perturbation cutoff. The new sensitivity report shows structural rejection of `7/20` to `19/20` across cutoffs `0.40–0.70`; MiniLM rejects `0/20` throughout that interval. The manuscript now labels `0.45` illustrative and reports the sensitivity range.
4. Separated the two perturbation benchmarks in Table 3: schema reassignment versus observable perturbation.
5. Narrowed unsupported claims about uniform TED, pq-Gram/tree-kernel performance, and embedding positional insensitivity.
6. Marked GitBugs as exploratory transfer because split/tuning provenance is incomplete.
7. Removed internal citation-alignment notes from the manuscript and removed the orphan citation marker.
8. Added discussion explaining that near-identical natural-benchmark scores indicate that the structural contribution is attribution and robustness, not universal accuracy superiority.

## New audit artifacts

- `experiments/document_disjoint_hybrid_138.py`
- `reports/audit/document_disjoint_hybrid_138.json`
- `reports/DOCUMENT_DISJOINT_HYBRID_138.md`
- `experiments/audit_observable_threshold_sensitivity.py`
- `reports/audit/observable_threshold_sensitivity_20.json`
- `reports/OBSERVABLE_PERTURBATION_THRESHOLD_SENSITIVITY_20.md`
- `reports/HYBRID_PROVENANCE_AUDIT_138.md`

## Still open before submission

- Recover and document the exact GitBugs split manifest, train/test separation, and adaptation-selection procedure, or remove the cross-genre result from the main evidence.
- Add a tokenizer/input-length audit for Qwen3 only if it can be directly reproduced from the Kaggle artifact.
- Consider nested document-disjoint selection for alpha before making a strong Hybrid claim.
