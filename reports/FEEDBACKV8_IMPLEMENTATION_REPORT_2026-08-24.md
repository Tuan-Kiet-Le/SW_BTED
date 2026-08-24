# Feedback v8 implementation report — 2026-08-24

## Completed

- Narrowed the Introduction TED wording to standard document-oriented baselines and acknowledged broader weighted-TED prior work.
- Renamed Section 6.3 to `Why natural-document accuracy is not the primary contribution`.
- Added label-provenance and pair-dependence caveats to Section 7.
- Removed the redundant Qwen3-only truncation paragraph and retained one clean all-model truncation paragraph.
- Physically removed legacy HTML comments from the manuscript source.
- Standardized Table 1 names to Single-vector MiniLM, BGE-small, MPNet, and Qwen3.

## Verification

- No HTML comments remain in the manuscript.
- No `preregistered`, repository TODO, or old `accuracy parity` heading remains.
- No visible `full-document` baseline naming remains in the manuscript.
- `git diff --check` passes.

## Remaining submission action

The working tree contains post-v8 edits while the manuscript still points to commit `a291bcc`. Create a final commit/tag after venue formatting and replace the pointer with that final immutable revision. For double-blind venues, use an anonymized repository or supplementary artifact instead of the named public GitHub URL.
