# Structural-Perturbation Benchmark Reproduction

Date: 2026-08-14  
Scope: controlled 20-pair D2↔D3 section-reordering benchmark

## Result

The benchmark was rerun successfully using the four-layer workspace data and
the pinned local `all-MiniLM-L6-v2` snapshot. The new score table is byte-for-
byte equivalent in score columns to the historical
`reports/audit/structural_perturbation_results.csv`; the historical file was
not overwritten.

| Method | Threshold | TP | FP | TN | FN | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full-document SBERT | 0.45 | 0 | 20 | 0 | 0 | 0.0000 |
| SW-BTED structural-only | 0.45 | 0 | 0 | 20 | 0 | 1.0000 |
| SW-BTED hybrid | 0.45 | 0 | 20 | 0 | 0 | 0.0000 |

The structural-only versus full-document SBERT comparison has `n10 = 20`,
`n01 = 0`, with exact McNemar p = `1.9073486 × 10^-6`.

Score ranges were:

- Structural similarity: mean `0.3064`, range `0.2696–0.3701`.
- Full-document SBERT: exactly `1.0000` for all 20 pairs.
- Hybrid similarity: mean `0.5838`, range `0.5618–0.6221`.

## Provenance and implementation check

The recovered generator is
`D:\FPT\Semester_8\RAG_Research\scratch\run_structural_perturbation_benchmark.py`.
It selects the first 20 `Type_A` rows from `data/dataset/pairs.csv`, copies
each source tree, swaps the `schema_class` values of its D2 and D3 children,
and keeps the text identical on both sides. All 20 source trees contained both
required layers at child positions D2=1 and D3=2.

The rerun implementation is
`experiments/clean_structural_perturbation_20.py`. It uses the same SW-BTED
cost/evaluation path and a pinned local embedding snapshot, so it does not
require Hugging Face network access.

Machine-readable outputs:

- `reports/audit/clean_structural_perturbation_results_20.csv`
- `reports/audit/clean_structural_perturbation_metrics_20.json`

## Interpretation and limitation

This reproduces the manuscript's controlled diagnostic result: SW-BTED
structural-only rejects all 20 deliberately misordered trees, while the flat
embedding and hybrid score treat every pair as similar because the underlying
text is unchanged. It is evidence for sensitivity to the tested structural
perturbation, not evidence of general real-world classification performance.

The labels are constructed by design and the sample contains only negative
perturbations. Therefore the benchmark cannot estimate recall, positive-class
sensitivity, or performance on naturally occurring section-order errors. The
manuscript should retain this limitation explicitly and should not present the
result as a standalone replacement for the 138-pair real-only evaluation.
