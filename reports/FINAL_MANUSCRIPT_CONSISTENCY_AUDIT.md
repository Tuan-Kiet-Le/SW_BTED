# Final Manuscript Consistency Audit

Date: 2026-08-14  
Manuscript: `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`

## Executive status

**Not submission-ready yet.** The main 138-pair results and the 20-pair
structural-perturbation result, and the new runtime measurement are consistent
with the current canonical artifacts. Visible placeholders and repository
provenance work remain.

## Checks completed

| Check | Status | Evidence |
|---|---|---|
| Four-layer representation | Pass | Manuscript and canonical source use Root → Domain → Intent → Terminology |
| Primary scope | Pass | 138 real-only pairs are identified as primary evidence |
| SW-BTED structural-only result | Pass | F1 `0.9498 ± 0.0253`, backed by `FINAL_CANONICAL_RESULTS_138.md` |
| Natural baseline interpretation | Pass | Manuscript says parity with strong embedding baselines, not superiority |
| Structural perturbation result | Pass | 100% vs 0%, exact p `1.9073 × 10^-6`, reproduced in `STRUCTURAL_PERTURBATION_REPRODUCTION_20.md` |
| Modern baseline | Pass with wording caveat | Qwen3 is reported, but pooled OOF rather than fold mean; this must remain explicit |
| Case A scope | Pass | `SU26SE102–SU26SE102_plag` is explicitly stated as retained |
| TODO/placeholder scan | Fail | Two unresolved content TODOs remain |
| Runtime evidence | Pass | Canonical 138-pair timing report includes environment, protocol, and pair-level output |
| Figures/tables | Pass with caveat | Figures 1–4 now have rendered PNG/SVG artifacts and captions; Table 1–4 remain Markdown tables that must be converted to the target journal format |
| Text encoding | Pass | The manuscript is valid UTF-8; the earlier mojibake was caused by a non-UTF-8 console display, not by file corruption |

## Blocking findings

### 1. Unresolved manuscript TODOs

The manuscript still contains TODO markers for:

- repository and dataset links;
- a decision about a third domain;

### 2. Figure packaging is complete; journal conversion remains

Figures 1–4 are now rendered under `docs/submission_figures/` and referenced
from the manuscript. Figure 4 is based on canonical APTED mappings for the
three explicitly identified cases; raw domain scores and representative edits
are stored under `reports/interpretability/`. The remaining work is only to
convert the Markdown tables and images into the chosen journal template.

### 3. Runtime evidence is now present

The runtime benchmark is documented in
`reports/RUNTIME_BENCHMARK_CANONICAL_138.md`. It measures the structural
alignment component on all 138 canonical pairs and clearly excludes parsing,
model loading, and embedding inference. This is sufficient for the current
revision, although an end-to-end latency experiment could be added later.

## Non-blocking clarification

Table 1 lists Hybrid F1 as `0.9744–0.9867`, while the canonical primary-results
report is centered on the structural-only result and pooled baseline metrics.
The manuscript should define whether this is a fold range, a threshold range,
or a configuration range and cite the exact output file. Ambiguous ranges are
likely to trigger a reproducibility question even if the underlying result is
correct.

## Recommended order of work

1. Remove the remaining stale/unfinished TODO markers where the corresponding
   artifact is now available.
2. Generate the actual figures and replace figure/table placeholders with
   captions and stable labels.
3. Add repository, environment, model-snapshot, and dataset-provenance links.
4. Run this audit again, then prepare the journal-specific manuscript format.

Until items 1–4 are completed, the paper has a credible Q2 research core but
should be considered **major-revision / pre-submission**, not submission-ready.
