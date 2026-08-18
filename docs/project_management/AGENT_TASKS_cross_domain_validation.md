# Agent Task Spec: SW-BTED Cross-Domain Validation (Bug Reports Dataset)

**Context:** SW-BTED has now been validated on the FPT capstone-proposal dataset: on the 138-pair real-only benchmark, SW-BTED's structural-only tree-alignment mode significantly and substantially outperforms a verified, independent, non-structural flat embedding baseline (F1=0.9498 vs. 0.4314, p≈2.52×10⁻²⁹). This spec extends that validation to a second, structurally different document genre — duplicate bug reports — to test the generalizability claim: **SW-BTED works on any structured document genre where an independent, expert-authored functional taxonomy already exists for the T2 domain layer.**

**Do not claim SW-BTED "works on any structured dataset" without qualification.** The correct, defensible claim is scoped: it generalizes where a real domain taxonomy exists (as it does here, per Bettenburg et al.), not to arbitrary structured text where a taxonomy would have to be invented (see: the PAN-PC dataset, which was correctly identified as *not* having this property and was scoped down to a 3-tier ablation instead of forced into a fake T2 schema).

**Do not modify the core SW-BTED cost function, β/α formula, or the triangle-inequality proof.** This is a new-dataset validation exercise, not an algorithm change.

---

## Why bug reports, and not CUAD, first

Bug-report duplicates are labeled by real developers triaging real tickets in production issue trackers — there is no synthetic-augmentation step, no GPT-paraphrase-defines-the-label problem, and no need to invent a pair-construction scheme the way CUAD would require (clause-category pairs need to be built from scratch; bug-report duplicate pairs already exist as ground truth). This sidesteps the single largest source of difficulty in the entire FPT validation. CUAD remains the logical next dataset after this one succeeds.

**T2 domain schema (already resolved in a prior task, reuse it):** Bettenburg et al.'s empirical bug-report-quality taxonomy —
- **D1 — Problem Description** (summary, observed behavior)
- **D2 — Reproduction** (steps to reproduce, expected behavior)
- **D3 — Environment/Context** (version, build, OS/platform)
- **D4 — Supporting Evidence** (stack traces, test cases, code examples)

**Source:** GitBugs (`https://github.com/av9ash/gitbugs`) or BugRepo/BugHub (LogPAI, `https://github.com/logpai/bughub`) — prefer GitBugs for size/recency and predefined duplicate-detection splits. Confirm the download link is still live before scripting against it; if dead, flag to a human rather than substituting a different dataset silently.

---

## Task A — Front-loaded baseline validity check (do this FIRST, before writing any SW-BTED-specific code)

**Why this comes first:** nearly every multi-round problem in the FPT audit traced back to a baseline being evaluated in a crippled or wrong configuration, discovered many rounds after the fact (TF-IDF's original 0.9939, B2/SBERT's tree-label-vs-full-doc scope mismatch, the mislabeled "flat baseline" that still contained a tree-alignment term). Do this diagnostic before any elaborate pipeline work, not after.

- [ ] Download and parse a sample of the bug-report dataset (fields mapped to D1-D4 per the schema above; T3 = sentences within each domain, T4 = extracted technical terms).
- [ ] Compute the **simplest possible baseline** — full-text single-document embedding (or TF-IDF) cosine similarity, no tricks, no restricted text scope — on a sample of known-duplicate and known-non-duplicate pairs.
- [ ] **Report the raw positive-class vs. negative-class similarity distributions (mean/min/max) for this simplest baseline before building anything else.** If this baseline already separates duplicates from non-duplicates well, that must be known and stated up front — it changes what SW-BTED needs to demonstrate, and changes how the paper should frame the comparison.
- [ ] Do not proceed to Task B until this distribution is reported and reviewed.

**Acceptance criteria:** A raw score distribution table for the simplest baseline, produced and reviewed before any SW-BTED-specific pipeline work begins.

---

## Task B — Genuine flat baseline, built and verified before the tree-alignment pipeline

- [ ] Implement a flat, schema-weighted domain-embedding baseline: per-domain (D1-D4) embedding similarity, weighted average, **zero tree-alignment term**. Before running it, **read the code and confirm no tree-edit-distance/`sim_struct`-equivalent term is present** — this exact mistake (a "flat" baseline that secretly retained the structural term) happened twice during the FPT validation and cost several audit rounds to catch.
- [ ] Run this flat baseline against the dataset's ground-truth duplicate labels. Report full confusion matrix (precision/recall/F1), not just aggregate accuracy.
- [ ] Any result that comes back suspiciously perfect (F1 near 1.0, near-zero variance across folds) triggers an immediate per-pair diff against any other baseline with similar-looking numbers, and a manual spot-check of a handful of pairs by hand — before it is reported as a finding, not after.

**Acceptance criteria:** A flat-baseline result with a full confusion matrix, code-verified to contain zero tree-alignment contribution, and manually spot-checked if the result looks unusually clean.

---

## Task C — Build the CapTree pipeline for bug reports

- [ ] `src/bug_report_parser.py`: map bug report fields to the 4-tier tree using the Bettenburg-derived D1-D4 schema above. Where fields aren't cleanly separated in the source data (free-text reports), use a lightweight rule-based tagger (regex for stack traces, keyword cues like "steps to reproduce:"/"expected:"/"actual:") — document the tagging method and its estimated error rate explicitly, since this is a real source of parsing noise.
- [ ] T4 terminology normalization: build `src/bug_tech_equiv_map.json` from scratch (do not reuse the capstone-domain TEM) — log misses so the map can be iteratively improved.
- [ ] Preserve the bug tracker's own marked-duplicate relationship (bug ID → duplicate-of ID) as ground truth — this is real triager-assigned labeling, not something to reconstruct or infer.
- [ ] Build positive pairs from marked duplicates. Build two categories of negative pairs, reported separately: **hard negatives** (same product/component, different actual bug) and **easy negatives** (different product/component) — this hard/easy split is the bug-report equivalent of FPT's Type B/C domain-confusion test and should be the primary metric of interest, not just aggregate F1.

---

## Task D — Run SW-BTED and compare against the flat baseline and simple baselines

- [ ] Run SW-BTED (structural-only mode first, matching how the FPT validation was actually run and verified) against: the simplest baseline (Task A), the genuine flat baseline (Task B), and standard baselines (TF-IDF, single-embedding cosine, pq-Gram if feasible).
- [ ] Report full confusion matrices and significance tests (McNemar/binomtest, Holm-Bonferroni corrected across all pairwise comparisons) for every comparison — no aggregate-F1-only claims.
- [ ] Report hard-negative vs. easy-negative performance separately for every method, not combined.
- [ ] If SW-BTED beats the flat baseline significantly on hard negatives specifically: that is the paper's cross-domain generalization claim, and it should be reported with the same rigor as the FPT novelty result (raw score distributions, verified-independent implementations, no suspiciously perfect unverified numbers).
- [ ] If SW-BTED does **not** show an advantage here: report this plainly. A negative or mixed cross-domain result is still a legitimate, useful finding about the boundaries of the method's applicability — do not suppress it or tune post-hoc to force a positive result.

---

## Non-goals / guardrails (carried forward from the FPT audit — apply from the start this time)

- Do not invent a T2 domain schema for any dataset that doesn't have an independent, citable taxonomy behind it. If no such taxonomy exists for a candidate dataset, either drop the T2 layer explicitly (3-tier mode, as an honest ablation) or don't use that dataset.
- Do not merge or average results across dataset subsets (e.g., different bug-tracker projects) without also reporting them separately — check for domain/project-specific effects before claiming a single generalization number.
- Do not report any "X beats Y" claim without a significance test in the same document.
- Do not accept a "flat" or "ablated" baseline's label at face value — verify its formula/code contains zero contribution from the component supposedly being excluded.
- A perfect or near-perfect result (F1=1.0, zero variance) requires more scrutiny before acceptance, not less — raw score distributions, threshold-independence check, and a manual spot-check, every time.
- If a task is reopened or a file resubmitted to fix one specific item, check every other number in that file against its last-known-good state before accepting the resubmission — do not assume the rest of the file is unchanged just because the requested fix looks correct.
- Any number that was previously investigated and formally discarded as invalid must never reappear without that discard being explicitly reopened and re-argued.

## Execution order

1. Task A (baseline validity check) — mandatory first step, no exceptions.
2. Task B (genuine flat baseline, code-verified).
3. Task C (SW-BTED pipeline construction for bug reports).
4. Task D (full comparison, significance-tested, hard/easy negatives reported separately).
5. Only after D: decide whether to proceed to CUAD as a third validation dataset, using the same front-loaded-validity-check approach from Task A onward.
