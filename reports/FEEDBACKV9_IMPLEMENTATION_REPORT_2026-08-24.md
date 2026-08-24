# Feedback v9 implementation report

Date: 2026-08-24

## Completed manuscript changes

- Clarified that the schema-penalty matrix is a metric over domain labels, not necessarily over distinct node instances.
- Renamed the proposition to `conditional triangle-inequality preservation` and limited its claim to the component representation spaces.
- Corrected the evaluation-protocol wording so that 5-fold stratified cross-validation applies specifically to the primary 138-pair natural-document results. Supplemental protocols are referred to separately.
- Reframed the observable perturbation comparison as a score-distribution diagnostic because SW-BTED and MiniLM scores are not calibrated to the same scale.
- Softened the causal wording around topic conflation and uniform TED to remain benchmark-scoped.
- Removed internal drafting notes from the abstract, results, discussion, and references.
- Replaced placeholder table markers with submission-ready table captions.
- Added the document-disjoint Hybrid reversal to the discussion.
- Changed the conclusion terminology from `structured document similarity` to `semi-structured document similarity`.
- Preserved the existing Type_A/Type_B/Type_C provenance limitation rather than inferring undocumented semantics.

## Verification

- Confirmed the canonical Case A pair `SU26SE102, SU26SE102_plag` exists in the frozen `pairs.csv` as `label=1`, `Type_A`.
- Confirmed no `[Table 1]`–`[Table 4]` placeholders remain.
- Confirmed no internal citation/result/discussion drafting-note markers remain.
- Confirmed no HTML comment delimiters remain in the manuscript.
- Ran `git diff --check` successfully.

## Remaining release action

The manuscript still references repository commit `a291bcc`, while the working tree contains the current manuscript and audit artifacts as uncommitted changes. Before submission, create the final reproducibility commit or immutable release tag, update the manuscript's repository pointer, and run one final artifact-consistency check.
