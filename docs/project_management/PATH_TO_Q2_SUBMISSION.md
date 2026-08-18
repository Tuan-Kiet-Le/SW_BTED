# Path to Q2 Submission: SW-BTED

**Purpose:** This is the master planning document tying together everything verified across the FPT and bug-report audit threads. It states what's actually earned so far, what's still needed, and in what order — so the paper's claims never outrun the evidence behind them.

---

## 1. Verified Contributions (as of now — do not overclaim beyond this)

| # | Contribution | Status | Evidence |
|---|---|---|---|
| 1 | Metric-preserving schema-weighted cost function (β/(1−β) convex combination guarantees triangle inequality by construction) | ✅ Solid, never challenged | Theoretical proof, unaffected by any audit round |
| 2 | Structural tree-alignment significantly beats a genuinely flat embedding baseline | ✅ Verified | FPT Real-138: Struct-Only F1=0.9498 ± 0.0253 (TP=38, FP=4, TN=96, FN=0) vs. Genuine Flat Domain SBERT F1=0.4314 (TP=38, FP=100, TN=0, FN=0), p = 2.52×10⁻²⁹ (code-verified, contingency table sum=138) |
| 3 | T2 domain schema must be grounded in an independent, citable functional taxonomy (transferable design principle) | ✅ Solid | Applied successfully to RE theory (FPT), Bettenburg et al. (bug reports); correctly *not* forced onto PAN-PC, which lacks one |
| 4 | Framework generalizes via hyperparameter adaptation, not per-domain algorithm redesign | ✅ Verified, modest claim | Bug reports: one budget-ratio fix (no algorithm change) brought Hybrid Mode to statistical parity with SBERT (p=1.0) |
| 5 | Honest boundary condition: structural signal alone is insufficient, only helps when blended with embeddings | ✅ Verified, useful negative result | Bug reports: Structural-Only F1=0.6725 vs. SBERT 0.9074, even after tuning |
| 6 | Hybrid Mode, Structural-Only, and Full-Doc SBERT achieve statistical parity on FPT Real-138 | ✅ Verified | FPT Real-138: Struct-Only F1=0.9498, Full-Doc SBERT F1=0.9867, Hybrid F1=0.9744. Pairwise McNemar tests confirm statistical parity across all three (p≥0.3750). |
| 7 | Interpretability: APTED sub-tree edit traces localize structural divergence for human review | ✅ Verified | Real edit-operation traces shown (DocA/DocB node counts per domain $D_1\dots D_4$), median-sampling selection disclosed. |

---

## 2. What This Is NOT (claims to explicitly avoid)

- **NOT** "SW-BTED beats SBERT/TF-IDF" — ties on both tested domains (FPT and bug reports) once baselines are correctly scoped. State ties as ties.
- **NOT** "works on any structured dataset" — only where an independent domain taxonomy exists. PAN-PC is the counterexample that proves you understand the boundary.
- **NOT** "SBERT has systemic population-level false positive failure" — Full-Doc SBERT is statistically tied with SW-BTED on Real-138. Use domain near-miss cases solely as an interpretability illustration, not a general accuracy deficiency claim.
- **NOT** using unreconciled combined-slice claims — the historical combined-slice Hybrid F1=1.0000 figure remains formally dropped. All primary results are reported strictly on the 138 Real-only dataset.

---

## 3. Prioritized Work Plan

### Item 0 (RESOLVED & CLOSED) — Significance Test Re-verified Against Genuine Flat Domain SBERT Baseline
- [x] **Significance Test Re-verified Against Correct Baseline ($n=138$ Real-only pairs):**
  - **Comparison Target:** Structural-Only vs **Genuine Flat Domain SBERT** (Task 5's flat embedding baseline without tree alignment).
  - **Individual Baseline Confusion Matrices ($n=138$):**
    - **SW-BTED Structural-Only:** $\text{TP}=38, \text{FP}=4, \text{TN}=96, \text{FN}=0 \implies F1 = \mathbf{0.9498 \pm 0.0253}$ ($\text{Precision}=0.9048, \text{Recall}=1.0000$). Check sum: $38+4+96+0 = \mathbf{138}$.
    - **Genuine Flat Domain SBERT:** $\text{TP}=38, \text{FP}=100, \text{TN}=0, \text{FN}=0 \implies F1 = \mathbf{0.4314 \dots 0.4318}$ ($\text{Precision}=0.2754, \text{Recall}=1.0000$). Check sum: $38+100+0+0 = \mathbf{138}$.
  - **Joint $2 \times 2$ McNemar Contingency Table ($n=138$):**
    - $n_{11}$ (Both correct) = $38$
    - $n_{10}$ (Structural-Only correct, Flat SBERT wrong) = $96$
    - $n_{01}$ (Flat SBERT correct, Structural-Only wrong) = $0$
    - $n_{00}$ (Both wrong) = $4$
    - **Contingency Table Check Sum:** $n_{11} + n_{10} + n_{01} + n_{00} = 38 + 96 + 0 + 4 = \mathbf{138}$.
  - **Exact Binomial Significance Test Result:**
    - Discordant pairs: $n_{10} = 96, n_{01} = 0 \implies n_{10} + n_{01} = 96$.
    - `binomtest(0, 96, 0.5)` $\implies \mathbf{p = 2.5244 \times 10^{-29}} \approx \mathbf{2.52 \times 10^{-29}}$ (Highly statistically significant win for SW-BTED Structural-Only over Genuine Flat Domain SBERT).

---

### Item 1 (RESOLVED) — Central Framing Paragraph
- [x] Framing paragraph correctly softened to *"...can obscure the specific locus of structural divergence, which we illustrate with representative cases"* — matches tie-finding evidence. No further action.

---

### Item 2 (RESOLVED) — Interpretability Case Studies & APTED Sub-Tree Edit Traces
- [x] Real edit-operation traces shown (DocA/DocB node counts per domain), median-sampling selection disclosed, template disalignment story explained. No further action.

---

### Item 3 — Structural-perturbation benchmark (best remaining shot at a second clean win)
Build synthetic perturbed variants (reordered sections, moved subitems, near-identical terminology) of existing documents and test whether tree-alignment beats flat/embedding methods specifically under structural perturbation — the condition where it's mechanistically supposed to have an edge and hasn't been tested yet.
- [ ] Construct perturbed pairs for FPT (and bug reports, if time allows).
- [ ] Run the full baseline set + SW-BTED with the same rigor standard as every other result in this project (raw score distributions, significance tests, code-verified flat baseline).
- [ ] Report honestly regardless of outcome — this is a real experiment, not a foregone conclusion.

---

### Item 4 (RESOLVED) — Third dataset (CUAD) or explicitly bounded two-domain claim
- [x] Explicitly scoped generalizability claim to "two structurally distinct genres" (FPT Software Capstones & GitBugs Bug Reports) in the paper.

---

### Item 5 — Related work and baseline currency
- [ ] Add a paragraph positioning schema-weighted TED against recent tree-kernel and structure-aware embedding literature (not just the classical baselines already used).
- [ ] Add at least one more modern embedding baseline (a current-generation model, not necessarily beating it) so the comparison set doesn't look dated to a 2026 reviewer.

---

### Item 6 (RESOLVED) — Statistical power
- [x] Verified sample size: 138 Real-only FPT pairs + 300 Bug Report pairs = 438 total pairs with exact binomial McNemar significance testing across all comparisons.

---

### Item 7 — Reproducibility package
- [ ] Clean repo; document dataset construction explicitly, including the real/augmented split and why it was necessary.
- [ ] Include the significance-testing code directly — given how much of this project's own internal audit was about verifying exactly this kind of computation, having it open and checkable is a genuine strength to highlight, not just hygiene.

---

## 4. Execution Order

1. ~~**Item 0** (re-verify significance against Genuine Flat Domain SBERT baseline; check table sum & exact p-value)~~ — **DONE (Struct-Only TP=38/FP=4/TN=96/FN=0 vs Genuine Flat SBERT TP=38/FP=100/TN=0/FN=0; contingency sum=138, p=2.52x10⁻²⁹)**.
2. ~~**Item 1**~~ — **DONE**.
3. ~~**Item 2**~~ — **DONE**.
4. **Item 3** (structural-perturbation benchmark) — proceed in parallel with Item 5.
5. ~~**Item 4**~~ — **DONE**.
6. **Item 5** (related work + modern baseline) — parallel with Item 3.
7. ~~**Item 6**~~ — **DONE**.
8. **Item 7** (reproducibility package) — last, once all results are final.

---

## 5. Standing Guardrails (carried forward from the entire audit process — apply to all remaining work)

- Any "X beats Y" claim requires a significance test in the same place it's stated, run against the specific baseline the claim is actually about.
- A "flat" or "ablated" baseline's formula must be read and confirmed to exclude the component under test, not just labeled as excluding it.
- A perfect or near-perfect result requires more scrutiny before acceptance, not less.
- A number formally investigated and discarded must never quietly reappear.
- When a value changes between rounds, the change itself must be explained with a trace, not just presented as the new fact.
- Every confusion matrix and contingency table must be checked for basic self-consistency (does it sum to n? does it reproduce its own stated F1/precision/recall?) before being used in any significance test.
- Report negative/tied results plainly — they are more credible to reviewers than unearned wins, and this project's strongest remaining asset is that its claims are now honestly bounded.
