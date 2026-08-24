# Feedback v2 implementation report — 2026-08-22

## Completed

- Added three explicit research questions to the manuscript.
- Replaced broad metric-preservation wording with a node-level condition and separated it from APTED's minimum-cost alignment computation.
- Added operational definitions for T3/T4 distances, β values, edit weights, pre-filter, edit-budget normalization, α, and missing-domain handling.
- Replaced “statistical parity” language with “no statistically significant paired-prediction difference detected”; clarified that McNemar is not an F1-equivalence test.
- Added MCC to Table 1; the schema-matched all-positive classifier is now visibly degenerate (`MCC=0`).
- Reframed cross-domain evaluation as transfer with adaptation rather than zero-shot generalization.
- Narrowed structural-embedding claims to the tested perturbation protocol.
- Verified that all 20 perturbation pairs have byte-identical text inputs on both sides and only D2/D3 schema labels change. This rules out truncation as an explanation for the equal text-embedding scores in that paired benchmark.
- Ran an independent schema-matched BGE-small baseline. It also predicts all 138 pairs positive: F1 `0.4314 ± 0.0160`, `TP=38, FP=100, TN=0, FN=0`.

## New artifacts

- `experiments/audit_perturbation_text_identity_20.py`
- `reports/audit/perturbation_text_identity_audit_20.json`
- `reports/PERTURBATION_TEXT_IDENTITY_AUDIT_20.md`
- `reports/audit/schema_matched_embedding_baseline_bge_small_138.json`
- `reports/audit/schema_matched_embedding_pair_scores_bge_small_138.csv`
- `reports/SCHEMA_MATCHED_EMBEDDING_BASELINE_BGE_SMALL_138.md`

## Still pending for a stronger Q2 submission

1. Qwen3 schema-matched baseline. No local Qwen3 model is available; the existing Qwen3 artifact is full-document only and was run on Kaggle.
2. Add a formal paired bootstrap/permutation test if the paper needs a direct inferential statement about F1 differences.
3. Add the official Qwen3 model citation/reference.
4. Finalize the repository/reproducibility URL in the manuscript.

The truncation audit remains in the limitations section for natural-document embedding comparisons, but it is not a confound for the 20-pair structural perturbation contrast because both sides receive identical text.
