# Reconciliation of Claude's Qwen3/MiniLM Feedback

Date: 2026-08-14

## Conclusion

Claude's warning identifies a real risk in principle, but the specific diagnosis is not correct for the files currently used by this project.

1. The Qwen3 CSV is aligned with the canonical 138-pair dataset by content and SHA-256 hash.
2. In that canonical dataset, zero-based index 84 is `SU26SE087`–`SP26SE001` (Type_B). This is also the pair recorded at index 84 in the existing historical raw audit artifact.
3. `SP26SE082`–`SP26SE082_plag` is a different pair from the 180-pair historical dataset. It is excluded from the real-only 138-pair evaluation because `SP26SE082_plag` is a generated/regen document.
4. The MiniLM discrepancy is explained by the threshold grid, not by four-decimal rounding: the historical protocol uses a 0.005 grid, whereas the independent audit used a 0.01 grid.

## Pair identity and ordering

The following files were compared using `(doc_a, doc_b, label, type)` rather than numeric index:

| Artifact | Rows | SHA-256 / finding |
|---|---:|---|
| `repro_candidate_138/data/dataset/pairs.csv` | 138 | `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56` |
| Kaggle `qwen3_pair_scores.csv` provenance | 138 | same hash: `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56` |
| `RAG_Research/Data/dataset/pairs.csv` | 180 | different file; contains the 42 generated-section audit branch |

The Qwen3 provenance manifest explicitly records the canonical 138-pair hash. The Qwen3 CSV has the same row ordering as the canonical file after accounting for CSV encoding/BOM; it is not a newly sorted dataset.

The relevant rows are:

| Dataset/context | Position | Pair | Type | Label |
|---|---:|---|---|---:|
| canonical 138 / Qwen3 | 84 (zero-based) | `SU26SE087` – `SP26SE001` | Type_B | 0 |
| historical 180-pair file | 26 (zero-based) | `SP26SE082` – `SP26SE082_plag` | Type_A | 1 |
| historical 180-pair file | 126 (zero-based) | `SU26SE087` – `SP26SE001` | Type_B | 0 |

Therefore the phrase “pair index 84 = `SP26SE082_plag`” is not supported by the current dataset files. Also, `SP26SE082_plag` does not occur in the Qwen3 138-pair CSV, so there is no Qwen3 score for that pair in `qwen3_pair_scores.csv` to retrieve by name.

The existing `reports/audit/raw_prediction_vectors_138.json` itself records index 84 as `SU26SE087`–`SP26SE001`; its `sbert_sim=0.6555` is attached to that pair. However, clean regeneration shows SBERT cosine `0.5151844621` for the same canonical row and reveals multiple mismatches between the old vector artifact and regenerated vectors. The old artifact is therefore not suitable as primary numerical provenance; see `CLEAN_RAW_VECTOR_REGENERATION_REPORT.md`.

## MiniLM protocol reconciliation

Using the independently generated MiniLM scores in `reports/qwen3_pair_prediction_audit.csv`, with the same 5-fold split (`StratifiedKFold`, shuffle=True, random_state=42), the following was tested:

| Score representation | Threshold grid | Mean F1 | Precision | Recall | Confusion matrix (TP, FP, TN, FN) |
|---|---|---:|---:|---:|---|
| unrounded | 0.01 | 1.0000 | 1.0000 | 1.0000 | (38, 0, 100, 0) |
| rounded to 4 decimals | 0.01 | 1.0000 | 1.0000 | 1.0000 | (38, 0, 100, 0) |
| unrounded | 0.005 | 0.9867 | 0.9750 | 1.0000 | (38, 1, 99, 0) |
| rounded to 4 decimals | 0.005 | 0.9867 | 0.9750 | 1.0000 | (38, 1, 99, 0) |

The 0.005-grid false positive is:

```text
pair: SP26SE068 – SU26SE063
type: Type_C
label: 0
MiniLM cosine: 0.5978018045
fold-specific threshold: 0.595
```

With a 0.01 grid, that fold selects `0.60`, so the pair is correctly rejected. Rounding the score to four decimals does not change this result (`0.5978` remains above `0.595` and below `0.60`). Thus the earlier explanation that rounding caused the historical `0.9867` result is incomplete/incorrect; the decisive difference is the threshold grid resolution.

## Additional note about `SP26SE082_plag`

The pair `SP26SE082`–`SP26SE082_plag` belongs to the 180-pair branch and is not part of the Qwen3 138-pair benchmark. A direct full-document MiniLM calculation on the historical document files gives cosine approximately `0.9004443`, but this number must not be inserted into the 138-pair Qwen3 comparison because it is outside that evaluation scope.

## Recommended manuscript/action changes

1. Do not compare baseline values by numeric index across the 180-pair and 138-pair branches. Use `(doc_a, doc_b)` keys and explicitly record the dataset hash.
2. State the threshold grid resolution in the evaluation protocol. For historical comparability, reproduce the 0.005 grid.
3. Correct the audit report's claim that four-decimal rounding explains MiniLM `F1=0.9867`; the reproducible explanation is the 0.005 versus 0.01 threshold grid.
4. Keep `SP26SE082`–`SP26SE082_plag` labeled as a 180-pair/generated-section audit case, not as a 138-pair real-only error.
5. Treat the existing manually assembled raw prediction vector as secondary audit evidence until it is regenerated directly from the canonical prediction pipeline.
