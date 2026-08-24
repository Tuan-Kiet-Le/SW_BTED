# Feedback v7 implementation report — 2026-08-24

## Completed

- Reframed the Abstract to mention only the evaluated software-proposal and bug-report domains.
- Changed RQ2 to cover controlled structural perturbations in plural.
- Neutralized the augmented-data exclusion wording.
- Added Sections 5.1–5.3 to Discussion 6.1 and described the observable perturbation result as diagnostic.
- Changed the main interpretability wording to “inspectable structural attribution”.
- Added a clean rendered truncation paragraph covering MiniLM, BGE-small, MPNet, and Qwen3.
- Added clarification distinguishing standard document-oriented TED baselines from the broader weighted-TED literature.

## Verification

- Dataset counts remain 178 documents and 138 pairs.
- Hybrid document-disjoint result remains pooled F1 `0.9048`.
- `git diff --check` passes.

## Submission note

The current working tree contains the post-v7 manuscript and reports, while the manuscript still points to commit `a291bcc`. A final commit/tag must be created after all review edits and then referenced in the manuscript. Double-blind venue policy must also be checked before retaining the public GitHub URL.
