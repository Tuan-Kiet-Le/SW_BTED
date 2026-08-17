# SW-BTED Submission-Neutral Package

This directory defines the journal-independent submission package. The
manuscript source of truth remains:

`draft/SW_BTED_FULL_DRAFT_CITED_V2.md`

The package is intentionally neutral with respect to LaTeX/Word templates,
page limits, citation style, and journal-specific section names. Those changes
should be applied only after a target venue is selected.

## Canonical evidence

- Primary manuscript: `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`
- Canonical 138-pair metrics: `reports/FINAL_CANONICAL_RESULTS_138.md`
- Canonical predictions: `reports/audit/final_canonical_predictions_138.csv`
- Clean baseline suite: `reports/CLEAN_BASELINE_SUITE_138.md`
- Qwen3 audit: `reports/QWEN3_PREDICTION_AUDIT_REPORT.md`
- Structural perturbation: `reports/STRUCTURAL_PERTURBATION_REPRODUCTION_20.md`
- Runtime benchmark: `reports/RUNTIME_BENCHMARK_CANONICAL_138.md`
- Interpretability trace: `reports/interpretability/CANONICAL_INTERPRETABILITY_TRACE_3.md`
- Figures: `docs/submission_figures/`

## Freeze rule

No result-producing source, dataset, threshold, baseline, or pair ordering may
be changed during journal formatting. Venue-specific edits should affect only
presentation unless a new scientific experiment is explicitly recorded as a
new version.
