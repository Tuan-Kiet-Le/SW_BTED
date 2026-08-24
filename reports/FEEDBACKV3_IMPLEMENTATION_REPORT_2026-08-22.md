# Feedback v3 implementation report — 2026-08-22

## Completed

- Reframed the theorem as a conditional node-level proposition; the implementation no longer claims that the complete edit cost is a metric.
- Audited the canonical domain penalty matrix: symmetric, zero diagonal, and no triangle-inequality violations.
- Added the domain penalty matrix to the manuscript.
- Renamed the original 20-pair diagnostic as a controlled schema-reassignment test and removed the fair-embedding-superiority interpretation.
- Added an observable structural perturbation audit with changed serialized text inputs. Results at threshold 0.45: SW-BTED structural-only rejects 17/20 (`85.0%`); MiniLM rejects 0/20 (`0.0%`). This remains a constructed diagnostic, not natural-document evidence.
- Added weighted tree-edit-distance prior work and narrowed the novelty claim to genre grounding, four-layer representation, schema/content costs, and inspectable traces.
- Added Qwen3-Embedding citation.
- Added document-disjoint robustness subsection to Results.
- Standardized Table 1 to mean-fold F1, pooled OOF F1, and MCC; clarified that precision/recall are supplementary.
- Reframed GitBugs as exploratory transfer with adaptation because the raw split/tuning provenance is incomplete.
- Reduced the contribution list from seven bullets to five research contributions.

## New artifacts

- `experiments/audit_domain_schema_matrix.py`
- `reports/audit/domain_schema_matrix_audit.json`
- `reports/DOMAIN_SCHEMA_MATRIX_AUDIT.md`
- `experiments/observable_structural_perturbation_20.py`
- `reports/audit/observable_structural_perturbation_20.json`
- `reports/OBSERVABLE_STRUCTURAL_PERTURBATION_20.md`
- `reports/GITBUGS_ADAPTATION_PROTOCOL_AUDIT.md`

## Remaining blockers

1. Recover the raw GitBugs pair/split files and document tuning/test disjointness.
2. Replace the repository TODO with the final public URL.
3. Decide whether to retain the exploratory GitBugs section in the submission or label it explicitly as preliminary evidence.
4. The observable perturbation audit should be expanded to multiple perturbation types for a final Q2 submission.
