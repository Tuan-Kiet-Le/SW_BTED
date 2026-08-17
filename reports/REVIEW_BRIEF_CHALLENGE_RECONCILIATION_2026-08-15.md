# Review Brief Challenge Reconciliation

Date: 2026-08-15

## Direct conclusions

| Concern | Verification | Conclusion/action |
|---|---|---|
| TF-IDF `0.987` vs. `0.4364` | `reports/CLEAN_BASELINE_SUITE_138.md` and `clean_baseline_suite_138.json` | Canonical clean suite reports TF-IDF `0.9867`, TP=38, FP=1, TN=99, FN=0. The `0.4364` row in this suite belongs to Standard TED, TP=38, FP=98, TN=2, FN=0. Do not relabel it as TF-IDF. |
| Section Cosine missing from brief | Direct table check | Brief was incomplete. It is now added as F1 `0.6837 ± 0.0894`; it is already present in the manuscript Table 1. |
| Case A membership | SHA-256 and name-match check | Canonical `pairs.csv` hash is `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`; row 0 is `SU26SE102,SU26SE102_plag,1,Type_A`. Case A is in the canonical 138-pair set. The `_plag` suffix alone is not an exclusion rule. |
| Qwen3 threshold grid | Fresh 5-fold rerun from raw Qwen3 scores | Re-running train-fold threshold selection with step `0.005` gives mean fold F1 `0.9867 ± 0.0267`, pooled F1 `0.9870`, TP=38, FP=1, TN=99, FN=0. Fold thresholds are `0.655, 0.655, 0.655, 0.630, 0.655`. The earlier 0.01-grid audit produces the same predictions, but the 0.005 rerun is the comparable protocol. |
| Runtime mean < median | Pair-level timing distribution | This is possible here: most calls are below the median, while only two calls are ≥30 ms; the maximum is 56.4 ms. It is not evidence of a unit error. Report both statistics and avoid claiming a particular distribution shape. |
| pq-Gram `0.947` vs `0.9479` | Rounding check | `0.9479` is the stored mean-fold value and rounds to `0.948` at three decimals. The brief's `0.947` was too coarse and should be changed to `0.9479` or `0.948`. |

## Exact code/protocol difference

The historical `0.4364` is reproducible. The old propagation script
`RAG_Research/scratch/propagate_task11_task3.py` calls
`get_cosine_tfidf_similarity(trees_nodes, real_pairs)` **without** the
`full_texts` argument. The legacy helper therefore falls back to
`tree_to_full_text()`, which concatenates tree labels/leaves. The clean script
`experiments/clean_baseline_suite_138.py` calls the same helper with
`corpus_full`, so it vectorizes the full-document fields from `full_texts.json`.

The per-pair reconstruction is in
`reports/audit/legacy_vs_clean_lexical_scores_138.csv`. Its aggregate result is:

| Variant | Positive mean | Negative mean | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy tree-label TF-IDF | 0.0880 | 0.2271 | 0.4364 | 38 | 98 | 2 | 0 |
| Clean full-document TF-IDF | 0.6118 | 0.1020 | 0.9867 | 38 | 1 | 99 | 0 |

This is a real input-scope change, not a cache or copy-paste error. It explains
the mechanism: the legacy tree-label representation has higher lexical
similarity for negative same-domain pairs, while the full-document
representation reverses that ordering on this dataset.

Section Cosine has the same kind of scope/implementation split. The legacy
tree-label reconstruction gives F1 `0.4081`; the clean full-text implementation
gives F1 `0.6837`. The variants differ in section schema, weights, empty
section handling, and whether `full_texts.json` is supplied. They are now
explicitly separated rather than treated as one baseline.

Therefore the current manuscript's claim is: under the clean full-document
baseline protocol, SW-BTED significantly beats Standard TED, Section Cosine,
and Genuine Flat Domain SBERT; it is at parity with clean TF-IDF and embedding
baselines. The historical tree-label results should be presented, if retained,
as a separately labeled legacy/input-scope ablation—not silently merged into
the clean baseline table.

## Resulting status

- The reviewer brief's missing Section Cosine row is fixed.
- Qwen3 is now confirmed under the same 0.005 threshold grid.
- Case A is confirmed as a canonical 138-pair example.
- The only remaining presentation correction is to show pq-Gram as `0.9479`
  (or `0.948`), not `0.947`.
