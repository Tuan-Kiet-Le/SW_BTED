# Agent Task Spec: SW-BTED Dataset Expansion & Evaluation Fixes

**Context:** SW-BTED is a schema-weighted bounded tree edit distance algorithm for semi-structured document similarity (originally built for capstone project registration forms, 4-tier CapTree: ROOT → DOMAIN → INTENT → TERMINOLOGY). Current results on FPT/PURE datasets show the proposed method losing to a plain TF-IDF baseline on all metrics, and ROC-AUC ceiling effects (1.0 for 4/8 methods) make the existing benchmark uninformative. This spec adds two new datasets to produce a defensible empirical story, plus the missing statistical rigor.

**Repo root:** `RAG_Research/` (existing structure — see `docs/research_notes/project_structure.md`)

**Do not modify** `05_sw_bted.py`'s core cost function or the β/(1-β) triangle-inequality guarantee. All work here is data pipeline + evaluation, not algorithm changes, unless a task explicitly says otherwise.

---

## Task 0 — Baseline statistical rigor (do this first, blocks nothing else)

**Goal:** Every existing result table (FPT, PURE) must report significance, not just point estimates.

- [ ] Add `experiments/significance_testing.py`:
  - Input: per-fold predictions for each method (must be saved per-fold if not already — check `results/updated_baselines/` for whether raw per-pair predictions exist; if only aggregate F1 is stored, re-run baselines and persist per-pair scores to `results/updated_baselines/raw_predictions/{method}_{dataset}.csv`)
  - Compute: paired McNemar's test (or bootstrap 95% CI on F1, 1000 resamples) between SW-BTED and each of {B1 TF-IDF, B2 SBERT, B7 SimCSE}
  - Output: `results/updated_baselines/significance_report.md` — a table with p-values / CIs for every pairwise comparison in the existing Table 3.2
- [ ] Do not touch the algorithm. This task is pure evaluation infrastructure.

**Acceptance criteria:** Running the script reproduces existing F1 numbers ± noise and additionally prints/saves a p-value or CI per baseline comparison.

---

## Task 1 — PAN Plagiarism Corpus (PAN-PC-11) integration

**Why:** PAN-PC has graded obfuscation levels (none / low / high paraphrase + automatic/manual/translation obfuscation). This is the experiment that explains *why* TF-IDF currently wins on FPT (verbatim-copy-dominant) and demonstrates where SW-BTED should win instead (high-obfuscation paraphrase).

**IMPORTANT — schema justification gap:** Unlike FPT/PURE (where D1-D4 come from an RE-theory functional decomposition) or the fixes in Tasks 2-3 below, generic PAN-PC prose (news/blog/wiki-style text) has **no universal, theory-grounded x-way domain taxonomy**. Do NOT invent one (e.g., via arbitrary text-tiling/paragraph boundaries) — that reintroduces the exact "ad hoc T2" problem this spec was revised to avoid. Pick ONE of the two options below and document the choice in `results/pan_pc/README.md`.

- [ ] **Option A (preferred, simplest):** Build PAN-PC as a **3-tier tree** — INTENT (sentence) and TERMINOLOGY (extracted term) only, no T2/DOMAIN layer at all. Treat "no domain layer" as a deliberate, reportable condition, not a workaround. This also produces a useful ablation: does removing the domain layer entirely hurt performance relative to FPT/PURE/CUAD/bug-reports (which all retain it)? Report this explicitly.
- [ ] **Option B (if a genre-structured alternative is preferred):** Substitute a plagiarism/text-reuse corpus built on genre-conventional text where a real schema exists, e.g., academic abstracts following the IMRaD convention (Introduction–Methods–Results–Discussion — a well-established genre norm, directly citable) or structured Wikipedia sections. This preserves the obfuscation-gradient benefit while giving T2 a real justification. Flag to a human before switching corpora — this changes Task 1's data source.

**Source:** Webis PAN-PC-11 corpus, distributed via Zenodo (search "Webis PAN-PC-11 Zenodo" for current DOI/link — corpus is free for research use, cite the corresponding PAN publication).

- [ ] `data/raw/pan_pc_11/` (or the Option B corpus) — download and extract (large; do NOT commit raw files to git, add to `.gitignore`)
- [ ] `src/pan_pc_parser.py` — new parser, analogous to `jd_parser.py`:
  - If Option A: T3 (INTENT) = sentence, T4 (TERMINOLOGY) = keyword/term extraction per sentence (reuse existing CSO/TEM/lemmatization pipeline from `03_normalizer.py` — do not fork it). No T2 node type.
  - If Option B: T2 (DOMAIN) = IMRaD section (or equivalent genre convention for the chosen corpus), T3 = sentence, T4 = term
  - Preserve PAN-PC's ground-truth XML annotations (source doc, offset spans, obfuscation type: none/random/artificial-high/artificial-low/simulated/translation) as pair metadata
- [ ] `src/pan_pc_dataset_builder.py` — build evaluation pairs stratified by obfuscation level:
  - Positive pairs: (suspicious segment, source segment) at each obfuscation level
  - Negative pairs: sample non-plagiarized segment pairs at matched topic-similarity level (avoid trivially-easy negatives — this is what caused the ceiling-effect problem on FPT/PURE)
- [ ] `experiments/run_pan_pc_evaluation.py`:
  - Run all existing baselines (B1–B7) + SW-BTED on PAN-PC, **broken out by obfuscation level** (this breakdown is the key deliverable — a single aggregate number is not useful here)
  - Output: `results/pan_pc/results_by_obfuscation_level.csv` and a summary table in `results/pan_pc/README.md`

**Acceptance criteria:** A table with rows = {none, low(artificial), high(artificial), translation}, columns = {F1 per method}, showing the lexical-method-advantage shrinking (or reversing) as obfuscation increases. If it does NOT show this pattern, report that honestly — do not tune thresholds to force the expected trend.

---

## Task 2 — Duplicate bug report dataset (BugRepo / GitBugs)

**Why:** Real (non-synthetic) human-labeled near-duplicates, semi-structured fields, orders of magnitude more pairs than FPT/PURE — fixes the small-N and synthetic-ground-truth criticisms simultaneously.

**Source options (pick one, prefer GitBugs for recency/size):**
- GitBugs: `https://github.com/av9ash/gitbugs` (Eclipse, Mozilla, Firefox, VS Code, Cassandra, etc., standardized fields, predefined train/test splits for duplicate detection)
- BugRepo/BugHub (LogPAI): `https://github.com/logpai/bughub` (Eclipse, Mozilla, Firefox, Eclipse JDT)

**T2 schema justification:** Do NOT use product/component as T2 — that's project metadata about the bug, not a functional decomposition of the report's *content*, and it isn't grounded in any theory of what a bug report is made of. Instead, base T2 on Bettenburg et al.'s well-cited empirical taxonomy of bug report information types ("What Makes a Good Bug Report?", FSE 2008 / TSE 2010), which identified the functional elements developers and reporters converge on:

| Domain | Bug report elements (Bettenburg et al.) |
|---|---|
| **D1 — Problem Description** | Summary, observed behavior |
| **D2 — Reproduction** | Steps to reproduce, expected behavior |
| **D3 — Environment/Context** | Version, build info, OS/platform |
| **D4 — Supporting Evidence** | Stack traces, test cases, code examples, screenshots (if text-described) |

This is a citable, published functional decomposition — directly analogous in kind to how D1-D4 for capstone forms come from RE theory. Use this 4-way split (not product/component) as T2.

- [ ] `data/raw/bug_reports/` — download chosen dataset
- [ ] `src/bug_report_parser.py`:
  - Map bug report fields to the 4-tier tree above:
    - T2 (DOMAIN): one of {Problem Description, Reproduction, Environment/Context, Supporting Evidence} per the Bettenburg taxonomy — a single bug report will typically contribute content to multiple D-nodes, same as a capstone form does
    - T3 (INTENT-equivalent): individual sentences within each domain section
    - T4 (TERMINOLOGY): extracted technical terms (stack traces, error codes, API/class names — this domain will need its own equivalence map, analogous to TEM but for software terms; do NOT reuse the capstone TEM as-is, build `src/bug_tech_equiv_map.json` from scratch or start from an empty map and log misses)
  - Preserve the marked-duplicate relationship (bug ID → duplicate-of ID) as ground truth
  - Note: not all bug trackers cleanly separate these fields into distinct form inputs (some are free text). Where fields aren't explicitly separated, a lightweight classifier or rule-based tagger (e.g., regex for stack traces, keyword cues like "steps to reproduce:"/"expected:"/"actual:") will be needed to assign sentences to D1-D4. Document the tagging method used — this is a real source of parsing noise, flag its estimated error rate.
- [ ] `src/bug_dataset_builder.py`:
  - Build positive pairs from marked duplicates
  - Build negative pairs by sampling same-component non-duplicates (hard negatives — same product/component, different actual bug) AND cross-component non-duplicates (easy negatives), reporting both separately
- [ ] `experiments/run_bug_report_evaluation.py`:
  - Run B1–B7 + SW-BTED
  - Report metrics separately for hard vs. easy negatives (this separation is the point — it's your new TC-TNR-equivalent test)
  - Output: `results/bug_reports/results.csv`, `results/bug_reports/README.md`

**Acceptance criteria:** F1/ROC-AUC/hard-negative-TNR reported per method, with sample size documented (should be thousands of pairs, not 200).

---

## Task 3 — CUAD (legal contracts) as the real cross-domain generalization test

**Why:** Replaces the LinkedIn JD experiment, which the paper's own text disqualifies as "trivially separable... not usable as quantitative benchmark." CUAD has genuine schema structure (contract type → clause category) and real linguistic variance across contracts (not template-identical).

**Source:** The Atticus Project, `https://www.atticusprojectai.org/cuad` (CC BY 4.0, 510 contracts, 41 expert-annotated clause categories — verify current download link is live before scripting against it, it may also be mirrored on Hugging Face datasets or Kaggle).

**T2 schema justification:** Do NOT use contract type (License, NDA, Service Agreement, etc.) as T2 — that's a document-level label, not a within-document functional domain, and it isn't how legal experts actually categorize clause content. Instead, use CUAD's own published taxonomy: the CUAD paper (Hendrycks et al. 2021) groups all 41 clause labels into three categories reflecting the importance of expert consideration:

| Domain | Description (from CUAD paper) |
|---|---|
| **D1 — General Information** | Party names, governing law, dates, renewal terms — baseline contract metadata |
| **D2 — Restrictive Covenants** | Clauses restricting the buyer's/company's ability to operate the business (e.g., non-compete, exclusivity) |
| **D3 — Revenue Risks** | Clauses that may require a party to incur additional cost or take remedial measures (e.g., liability, indemnification, uncapped liability) |

This is only 3 categories (not 4) — that's fine and expected; the point is that it's a real, citable, expert-authored functional taxonomy, not an invented one. Document this size difference from D1-D4 explicitly rather than padding to 4 artificially.

- [ ] `data/raw/cuad/` — download CUAD v1
- [ ] `src/cuad_parser.py`:
  - T2 (DOMAIN): map each of the 41 CUAD clause labels to one of {General Information, Restrictive Covenants, Revenue Risks} per CUAD's own published category assignment (check CUAD's supplementary materials/category mapping table — do not re-derive this mapping heuristically, use the dataset's own grouping)
  - T3 (INTENT-equivalent): the 41 clause categories themselves (Governing Law, Indemnification, Termination, etc.), nested under their D-parent
  - T4 (TERMINOLOGY): key terms/entities extracted per clause (party names generalized/masked, monetary terms normalized, date terms normalized — do NOT leave PII/party names as literal tokens, this is a public dataset but good practice + avoids spurious exact-match signal)
- [ ] `src/cuad_dataset_builder.py`:
  - Positive/similarity pairs: same clause-category, same contract-type, across different contracts (should be genuinely similar — same legal boilerplate function, different specific terms)
  - Negative pairs: same contract-type, different clause-category (tests topic-conflation-equivalent: two clauses from the same *kind* of contract but serving different legal purposes — this is your Type-B-equivalent hard negative)
- [ ] `experiments/run_cuad_evaluation.py` — same structure as above, output to `results/cuad/`

**Acceptance criteria:** A working cross-domain result that is NOT disqualified as trivial (i.e., document real variance in ROC-AUC/F1 — if this also hits a 1.0 ceiling, treat that as a signal the pair-construction needs harder negatives, not a result to publish as-is).

---

## Task 4 — Runtime/complexity benchmark (addresses reviewer concern on O(n³) cost)

- [ ] `experiments/run_runtime_benchmark.py`:
  - For each dataset (FPT, PURE, PAN-PC, bug reports, CUAD), measure wall-clock latency per pair for SW-BTED vs. each embedding-only baseline
  - Vary tree size (number of T3/T4 nodes) and plot latency scaling
  - Include the bounded-threshold variant explicitly, showing time saved vs. accuracy cost
  - Output: `results/runtime/latency_by_tree_size.csv` + plot

---

## Task 5 — Consolidated report

- [ ] `docs/research_notes/SW_BTED_v2_results_summary.md` — new document (do not overwrite `PROJECT_OVERVIEW.md`) summarizing:
  - Task 0 significance results overlaid on existing Table 3.2
  - Task 1 obfuscation-level breakdown (PAN-PC)
  - Task 2 hard/easy negative breakdown (bug reports)
  - Task 3 cross-domain CUAD result
  - Task 4 runtime table
  - One paragraph: does the overall narrative now support "SW-BTED wins under paraphrase/obfuscation, loses/ties under verbatim copying" — state this explicitly, whichever way the numbers land

---

## Execution order

1. Task 0 (blocks nothing, needed for all later comparisons)
2. Task 1 (PAN-PC) — highest priority, most directly fixes the core problem
3. Task 2 (bug reports) — fixes small-N + synthetic-label criticisms
4. Task 3 (CUAD) — fixes generalizability claim
5. Task 4 (runtime) — supporting evidence
6. Task 5 (report) — synthesis, do last

## Non-goals / guardrails

- Do not change the core SW-BTED cost function, β schedule, or the triangle-inequality proof to make numbers look better.
- Do not tune obfuscation-level thresholds or negative-sampling strategy post-hoc to force a particular expected result — report what the data shows.
- Do not commit raw dataset files to version control (add all `data/raw/*` paths to `.gitignore`); commit only parsers, builders, and derived small metadata files.
- If any dataset's download source is dead or changed, stop and flag it rather than substituting a different dataset silently — dataset provenance needs to be accurate for the paper's reproducibility section.
