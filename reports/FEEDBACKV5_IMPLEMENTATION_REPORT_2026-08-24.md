# Feedback v5 implementation report — 2026-08-24

## Completed

- Unified canonical `beta_T4` at `0.8` in `config.yaml`, manuscript, and frozen manifest. The existing beta audit confirms `0.8` and `1.0` are bit-identical on this dataset, but `0.8` is now the declared source-of-truth value.
- Added canonical config SHA-256: `5E13FB3F8A7000A5AD7DEBEA81F272880A02E80F49890D445290B59DC68071B3`.
- Replaced “preregistered” with “frozen canonical protocol”.
- Added manifest update date and a post-freeze supplementary-audit section.
- Added Hybrid, observable perturbation, and Qwen3 version 10 provenance pointers to the manifest.
- Replaced the repository TODO with the public repository, commit pointer, dataset path, and manifest pointer.
- Standardized Table 1 to Holm-adjusted p-values and documented the raw/adjusted audit location.
- Reduced the Abstract's emphasis on the weak all-positive diagnostic baseline and reframed the contribution around structural attribution and controlled diagnostics.
- Added explicit provenance wording for alpha `0.6` as a preliminary-development operating point.
- Added the current Qwen3 truncation result to the limitations discussion.

## Verification

- `git diff --check`: passed after cleanup.
- Canonical pair hash remains `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`.
- Qwen3 version 10 remains `F1=0.9867 ± 0.0267`, with 18/178 documents over the 2048-token cutoff.

## Remaining caution

The historical Qwen wording is retained inside an HTML comment in the manuscript source while the rendered manuscript uses the current audit wording. Before submission, the source should be flattened to remove that legacy comment entirely. GitBugs remains exploratory because its raw split and tuning manifest are unavailable.
