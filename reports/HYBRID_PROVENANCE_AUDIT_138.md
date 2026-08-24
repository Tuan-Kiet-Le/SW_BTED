# Hybrid provenance audit — canonical 138 pairs

## Findings

- The historical Hybrid evaluator uses `alpha = 0.6`, the formula `0.6 * structural_similarity + 0.4 * full_document_SBERT_cosine`, and the structural beta schedule `(T2=0.0, T3=0.9, T4=0.8)`.
- Its threshold protocol is 5-fold stratified cross-validation with thresholds selected on the training fold only using a `0.005` grid.
- The historical script fixes alpha; it does not independently select alpha inside that evaluation. A separate configuration-search harness exists and must not be mixed with the fixed historical result.
- Pair identity was checked by `(doc_a, doc_b, label, type)` against the canonical `pairs.csv`; the SHA-256 is `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`.

## Document-disjoint audit

The new audit uses connected-component GroupKFold so a document cannot occur in both train and test groups. It reports:

| Mean F1 | SD | Pooled F1 | MCC | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9200 | 0.1600 | 0.9048 | 0.8718 | 38 | 8 | 92 | 0 |

The result is lower than the historical pair-level Hybrid result (`1.0000 ± 0.0000`) and has substantial fold variance. Therefore the manuscript should retain Hybrid as a fixed operating-point audit, not as evidence of universally perfect generalization.

## Files

- Script: `experiments/document_disjoint_hybrid_138.py`
- Machine-readable output: `reports/audit/document_disjoint_hybrid_138.json`
- Summary: `reports/DOCUMENT_DISJOINT_HYBRID_138.md`

## Remaining limitation

This audit does not prove that alpha was selected without access to benchmark labels. A future submission-grade experiment should pre-register alpha on a development set or use nested document-disjoint validation.
