# Feedback v6 implementation report — 2026-08-24

## Completed

- Added Hybrid pair-level and document-disjoint pooled F1 to Table 3: `1.0000` versus `0.9048`.
- Added dataset statistics: 178 unique documents, 138 pairs, 38 Type_A positives, and 100 negatives split into 50 Type_B and 50 Type_C.
- Explicitly documented that the current package preserves labels but not an independent human-annotation/pair-construction record.
- Renamed visible embedding baseline descriptions to single-vector baselines with model-specific truncation limits.
- Added the Qwen3 2048-token truncation result to the visible limitations discussion.
- Narrowed the schema-grounding principle in the abstract/introduction to “primarily grounded” with documented extensions.
- Added a clarification distinguishing standard document-oriented uniform TED baselines from the broader weighted-TED literature.
- Removed inferential “tie, p=1.0000” wording from the GitBugs table.
- Added repository pointers for T3 extraction and the Technology Equivalence Map.

## Verification

- `git diff --check` passed.
- Hybrid document-disjoint result matches the audit: pooled F1 `0.9048`.
- Canonical dataset counts match `pairs.csv`: `138 = 38 + 50 + 50`.

## Remaining pre-submission cleanup

The old Qwen/truncation wording remains inside non-rendered HTML comments in the manuscript source. It does not appear in rendered Markdown, but those legacy comments should be physically removed before final venue formatting. The raw dataset-construction/annotation archive is still unavailable, so the manuscript reports that provenance gap explicitly rather than inventing an annotation claim.
