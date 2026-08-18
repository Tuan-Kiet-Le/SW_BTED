# SW-BTED Source Provenance — Timestamp-First Triage

Date: 2026-08-14  
Mode: read-only investigation; no experiment was executed and no source file was modified.

## 1. Scope and target

The target is the four-layer manuscript:

`draft/SW_BTED_FULL_DRAFT_CITED_V2.md`

The primary manuscript scope is reported as 138 real-only pairs. The main structural-only result to trace is:

```text
SW-BTED structural-only F1 = 0.9498 ± 0.0253
```

The 180-pair branch containing 42 GPT-generated sections is treated as secondary audit material.

## 2. Anchor timestamps

| Artifact | Last modified | Interpretation |
|---|---:|---|
| Manuscript `draft/SW_BTED_FULL_DRAFT_CITED_V2.md` | 2026-08-14 03:56:54 | Current manuscript; edited after the experiment artifacts |
| `RAG_Research/results/evaluation_metrics.csv` | 2026-07-20 11:58:37 | Combined evaluation output; SW-BTED F1 = 0.9697 |
| `RAG_Research/results/results_leak_free.csv` | 2026-07-20 11:58:37 | Leak-free evaluation output |
| `RAG_Research/results/mcnemar_results.csv` | 2026-07-20 11:58:37 | Main paired significance output |
| `RAG_Research/results/audit/real_vs_augmented_breakdown.csv` | 2026-07-20 13:25:32 | 138 real-only / 42 augmented-only breakdown |
| `RAG_Research/results/audit/significance_report_v2.md` | 2026-07-22 07:51:24 | Later audit/reconciliation document |

## 3. Strongest timestamp candidate chain

The closest chronological chain found is:

| Stage | File | Last modified | SHA-256 prefix |
|---|---|---:|---|
| Raw/processed project records | `RAG_Research/data/processed/topics_sp26.json` | 2026-07-20 09:58:56 | `2ABDB5AA3FA63E82` |
| Plagiarism sections | `RAG_Research/data/processed/plag_sections.json` | 2026-07-20 10:08:51 | `7A6D270006D3F5BA` |
| Evaluation harness | `RAG_Research/experiments/main_evaluation.py` | 2026-07-20 10:59:50 | `98E298099B86EE31` |
| TF-IDF tree input | `RAG_Research/data/dataset/trees_section_tfidf.json` | 2026-07-20 10:59:09 | `9C5274EF012E0035` |
| Final section trees | `RAG_Research/data/dataset/trees_section.json` | 2026-07-20 11:56:21 | `35F0797183F9A389` |
| Evaluation metrics | `RAG_Research/results/evaluation_metrics.csv` | 2026-07-20 11:58:37 | `8E6A4FA78DCD8B4C` |
| Pair-level predictions | `RAG_Research/results/pair_similarities.csv` | 2026-07-20 11:58:37 | `661B4E16AF4F4956` |
| Leak-free summary | `RAG_Research/results/results_leak_free.csv` | 2026-07-20 11:58:37 | `DC013D313CF977B6` |
| Real/augmented split | `RAG_Research/results/audit/real_vs_augmented_breakdown.csv` | 2026-07-20 13:25:32 | `6CE69364CFBCE3A7` |
| Cost engine currently present | `RAG_Research/src/05_sw_bted.py` | 2026-07-20 13:25:59 | `ECFFA3F0ECD3794E` |
| Baselines currently present | `RAG_Research/src/baselines.py` | 2026-07-21 15:21:03 | `024194658E820D5D` |

This chain is the best first candidate because the timestamp order is consistent with data preparation, evaluation, result generation, and later audit extraction.

## 4. Evidence that the candidate contains the manuscript's main SW-BTED result

`RAG_Research/results/audit/real_vs_augmented_breakdown.csv` contains this row:

```text
Real-only,SW-BTED,0.9498,0.0253,0.9056,1.0,1.0,0.9818,0.9424
```

This matches the manuscript's structural-only SW-BTED result exactly at the displayed precision.

The same file reports 138 real-only pairs and 42 augmented-only pairs, consistent with the manuscript's distinction between the real-only primary slice and the GPT-augmented audit branch.

## 5. Important mismatch discovered

The timestamp candidate is not yet proven to be the complete source for manuscript Table 1.

The manuscript reports the following real-only baseline values:

| Method | Manuscript F1 |
|---|---:|
| SW-BTED structural-only | 0.9498 |
| Genuine flat domain SBERT | 0.4314 |
| Full-document SBERT | 0.9867 |
| pq-Gram | 0.9479 |
| Section Cosine | 0.4081 |

The timestamp-near `real_vs_augmented_breakdown.csv` reports:

| Method | Candidate real-only F1 |
|---|---:|
| SW-BTED | 0.9498 |
| B1 Cosine TF-IDF | 0.4364 |
| B2 Cosine SBERT | 0.4131 |
| B4 pq-Gram | 0.9579 |
| B5 Section Cosine | 0.4314 |

Therefore:

- SW-BTED matches.
- Several baseline values do not match the manuscript table.
- The baseline input scope or implementation likely changed between result generation, audit slicing, manuscript drafting, or later baseline correction.
- The source set must not yet be copied or declared canonical.

## 6. Historical runs considered

The experiment history contains multiple four-layer-era runs, including:

```text
experiments/history/20260603_150600_v3_leak_free
experiments/history/20260603_165808_v4_leak_free_clean_keywords
experiments/history/20260603_224536_v5_clean_all_reproducible
experiments/history/20260606_101359_v6_stratified_cv_180pairs
experiments/history/20260606_101534_v6_stratified_cv_180pairs_final
experiments/history/20260606_101740_v6_stratified_cv_180pairs_final_benchmark
experiments/history/20260606_143235_v7_final_mandatory_tech_replacement
```

The v3–v5 runs are older 138-ish-pair-era candidates, but their archived outputs do not immediately identify the manuscript's exact `0.9498` table. The v6/v7 runs are explicitly 180-pair branches and therefore cannot be accepted as the manuscript's primary run without separating the 138-pair slice.

## 7. Current provenance status

Status: **PARTIALLY IDENTIFIED — NOT YET REPRODUCTION-READY**

Confidence by artifact:

| Artifact | Status |
|---|---|
| 138 real-only SW-BTED F1 = 0.9498 | Strong timestamp/result match |
| 180-pair combined evaluation | Identified at `results/evaluation_metrics.csv` |
| Exact manuscript baseline table | Not yet uniquely identified |
| Exact code/config pair for every manuscript number | Not yet proven |
| Canonical source manifest | Not yet safe to create |

## 8. Recommended next read-only checks

Before running anything:

1. Inspect the exact baseline functions and input selection in the version of `baselines.py` used when the manuscript table was produced.
2. Compare the manuscript's baseline labels with columns and method names in `pair_similarities.csv`.
3. Compare hashes and contents of the current `baselines.py` against the workspace copy and any archived/source snapshots.
4. Trace which script generated `results/audit/real_vs_augmented_breakdown.csv` and whether it recomputed baselines or only sliced an existing prediction file.
5. Identify the source of the manuscript's full-document SBERT F1 = 0.9867.
6. Only after these checks, run a smoke test and then the recovered 138-pair evaluation.

## 9. Conclusion

Timestamp triage successfully narrowed the search to the July 20, 2026 evaluation chain. It strongly links that chain to the manuscript's SW-BTED structural-only F1 = 0.9498, but it also exposed baseline inconsistencies. The next action should be a read-only code/output reconciliation, not a source copy or full experiment run.
