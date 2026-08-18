# Task Spec: Phase 1 Verification Follow-ups (Before Data Freeze)

**Context:** `BAO_CAO_GIAI_DOAN_1_PHASE1_REPORT.md` reports Phase 1 as "100% complete and frozen," ready for Phase 2 (writing). This is now accurate. All four tasks are genuinely resolved, including Task 2, which required escalation to manual verification after four failed automated rounds — the fifth attempt was independently re-derived and confirmed correct by hand. One small, unrelated cleanup item (a stale Hybrid Mode F1 upper bound in Section IV) is flagged for Phase 2 but does not block proceeding.

---

## Task 1 — Resolve the Qwen3 vs. BGE-small naming mismatch (RESOLVED)

**Status: Fully resolved, verified word-for-word.** The scope note now reads exactly as the suggested hedge: "...we expect but have not verified similar behavior with larger LLM-based embeddings." No longer asserts an untested claim as settled. No further action needed.

---

## Task 2 — Verify the BGE-v1.5 result is independently computed, not reused from SBERT (RESOLVED)

**Status: Genuinely resolved after escalation to manual verification.** All three pairwise disagreement counts were independently re-derived by hand from the stated confusion matrices (not just re-read from the report) and confirmed mutually consistent:
- SBERT vs. BGE-small: disagree at {30, 68, 84} → 3 disagreements. Confirmed.
- SBERT vs. MPNet: disagree at {20, 68, 84, 107} → 4 disagreements. Confirmed.
- BGE-small vs. MPNet: disagree at {20, 30, 107}, correctly **excluding** index 68, where both models are wrong in the *same* direction (both false-positive on the same true negative) — this is the specific detail that confirms the numbers were derived from real per-pair predictions, not asserted.

All three original problems from this task's history are resolved: pair index 84's cosine value is now consistent across both documents (0.6555 ≥ 0.655), SBERT is correctly anchored at F1=0.9867 (not reverted to the previously-rejected 1.0000), and BGE-small/MPNet no longer show suspiciously identical metrics (0.9737 vs. 0.9610 — genuinely distinct). No further action needed on this task.

**One small, separate cleanup item for Phase 2 (not a reopening of this task):** Section IV's Pareto tradeoff discussion still cites Hybrid Mode's "natural F1 = 0.9744 ... 1.0" — the 1.0 upper bound appears to reference the combined-slice Hybrid Mode figure that was separately and deliberately dropped from this project's headline results (per `PATH_TO_Q2_SUBMISSION.md`, due to its own unresolved credibility history). Flag this for cleanup before Phase 2 drafting so the dropped number doesn't silently resurface.

---

## Task 3 — Justify the perturbation-benchmark's ground-truth labeling (RESOLVED)

**Status: Genuinely resolved.** The justification was correctly reframed from an attributed standards citation to the authors' own reasoned argument ("we argue that, consistent with the principles of structured requirements specifications...") — this is honest and defensible, and no longer presents an unverified claim as if it were a checked citation. The added benchmark-design note explicitly scopes the experiment to single-document section-reordering and flags multi-document perturbation as future work, which directly answers the circularity concern raised previously. No further action needed.

---

## Task 4 — Add explicit Discussion of the Structural-Only vs. Hybrid Mode tradeoff

**Why:** On the perturbation benchmark, Hybrid Mode fails completely (FP=20/20, identical to plain SBERT) — only Structural-Only wins (FP=0/20). This is in direct tension with the paper's other headline claim that Hybrid Mode ties strong SOTA baselines on natural data (Real-138). The two results, taken together, mean the paper cannot claim one single "best" configuration that both ties SOTA accuracy and detects structural perturbation — these are achievements of two different configurations that trade off against each other, because the embedding term in Hybrid Mode dominates and washes out the structural sensitivity that makes Structural-Only win the perturbation test.

- [ ] Add an explicit paragraph to the Discussion section stating this tradeoff plainly: Structural-Only is the structure-sensitive configuration (wins on perturbation, more modest on natural-data accuracy); Hybrid Mode is the accuracy-competitive configuration (ties SOTA on natural data, but is as blind to structural perturbation as flat embeddings alone).
- [ ] Do not let Section V's "excellent all-around result" framing stand without this caveat — it currently implies one model achieves both wins, which is not what the data shows.
- [ ] Consider whether this tradeoff itself is worth featuring as a finding (e.g., "practitioners should choose Structural-Only when structural fidelity is the primary concern, and Hybrid Mode when general semantic accuracy is prioritized") rather than something to minimize.

**Acceptance criteria:** An explicit Discussion paragraph addressing the Structural-Only/Hybrid Mode tradeoff, and removal of any framing implying a single configuration wins on both fronts.

---

## Standing note for future rounds

**When a resolved task is revisited to add supplementary detail (e.g., adding CV thresholds to an already-verified MAD table), the new addition must be checked with the same rigor as the original fix — it is not automatically safe just because it's appended to already-good work.** This has now happened twice in a row on the exact same SBERT/BGE-small comparison: the first attempted fix produced identical metrics between two different models; the second attempted fix, meant to prove independence, instead produced a direct mathematical contradiction plus a silent, unreconciled improvement to an already-settled number. Two consecutive failures on the same specific comparison is a signal that summary-statistic reporting is not sufficient here — **raw per-pair prediction data should be the standard going forward for this comparison specifically**, not an occasional spot-check.

**Any result that silently improves on a previously-verified, already-load-bearing number (especially to a perfect or near-perfect score) must be treated as suspect by default and requires an explicit trace of what changed, even if no one asks.** This is the second time in this project a quiet jump to a perfect result has occurred without the report itself flagging that it differs from an earlier, already-accepted figure.

**When the same specific comparison fails automated verification for the third or fourth time in a row, stop generating further automated reports and escalate to direct manual inspection of the raw data.** This happened with the SBERT/BGE-small/MPNet comparison specifically: four consecutive rounds each produced a different, self-contradicting result while addressing only the previous round's specific complaint. Continued automated iteration on a comparison with this failure history is unlikely to converge — a human directly reading the raw prediction file is more reliable at this point than another generated report.

---

## Execution order

1. **Task 1 — DONE.** Retitling and scope note verified correct word-for-word; no further action.
2. **Task 2 — DONE.** Resolved after escalation to manual verification; all three pairwise disagreement counts independently re-derived by hand and confirmed mutually consistent with the stated confusion matrices, including the subtle BGE/MPNet shared-error-at-index-68 exclusion. No further action.
3. **Task 3 — DONE.**
4. **Task 4 — DONE.**

**Status: Phase 1 is genuinely complete. Clear to proceed to Phase 2.** One small, non-blocking cleanup item noted in Task 2 above (a stale Hybrid Mode F1 upper bound in Section IV referencing a previously-dropped number) — fix opportunistically during Phase 2 drafting, does not require another verification round.
