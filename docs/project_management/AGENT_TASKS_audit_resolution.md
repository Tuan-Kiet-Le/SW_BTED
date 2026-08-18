# Agent Task Spec: SW-BTED Audit Resolution & Novelty Test

**Context:** Three rounds of audit have surfaced unresolved data-integrity questions in the ablation study and the new GPT-augmented evaluation (`EVALUATION_RESULTS_UNBIASED.md`). Before any more datasets are added (see `AGENT_TASKS_dataset_expansion.md`) or any paper draft is finalized, these items must be closed. **Do not modify the core SW-BTED cost function, β schedule, or triangle-inequality proof as part of this spec** — every task here is data validation, provenance tracing, or a new controlled experiment, not an algorithm change.

**Priority order matters.** Tasks are numbered in the order they should be executed. Do not skip ahead — later tasks assume earlier ones are resolved, and running them out of order risks building new results on top of still-unverified numbers.

---

## Task 12 — Resolve the α/Hybrid-Score implementation mismatch (PARTIALLY RESOLVED — original B2 provenance now traced and accepted; one mechanism question remains; spun off a critical new item into Task 3)

**Status: The original-B2-provenance question (the most important open item from prior rounds) is now credibly answered.** This round traced the original `PROJECT_OVERVIEW.md` Table 3.2 B2 result (F1=0.9593) directly to `experiments/run_task3_new_baselines.py`, showing the actual code line (`get_fpt_full_text`) confirming it used full-document prose, and reproducing the F1 almost exactly. This is concrete, checkable evidence (an actual code reference plus matching numbers) — treat this specific provenance question as resolved. Do not re-investigate what text scope the original B2 baseline used.

**Consequence, now moved to Task 3 (see that task):** this means every "SW-BTED beats SBERT" significance claim across this entire audit thread was computed against the wrong (tree-label-only) version of B2, invented within this audit thread and never the paper's actual baseline. Task 12's own McNemar test of the correctly-scoped comparison already shows p=0.3750 (not significant). This is the headline finding from this round and is tracked as a required fix under Task 3, not here.

**Still open on this task specifically — the 0.2642 mechanism, not just its origin cell — and no new work has been done on it.** The most recently submitted `alpha_hybrid_formula_resolution.md` is an unchanged resubmission of the prior round's file. The mechanism question (how a PURE-dataset CSV value ended up quoted, across multiple rounds, as an FPT-dataset `sim_global` figure) remains exactly as open as it was last round.

- [ ] Trace the actual mechanism: is there a hardcoded/stale reference in whatever script or template generates these audit summary reports that pulls from the wrong dataset's file under some condition? Or was this a one-off manual transcription error? The distinction matters because the former implies other numbers in other reports could be silently cross-contaminated the same way, and needs a broader check; the latter is genuinely a one-off and can be closed once flagged.
- [ ] If a systemic cross-dataset contamination path is found in the reporting pipeline: spot-check a few other recently-reported numbers (e.g., other `sim_global` or baseline figures) to confirm they aren't affected the same way.

**Acceptance criteria:** A mechanism-level explanation (not just a source-cell citation) for how the PURE-dataset value ended up in an FPT-dataset report, and confirmation of whether this is an isolated incident or a systemic reporting-pipeline issue.

---

## Task 8 — Harness code-path audit (DONE)

**Status: Fully resolved.** The threshold-degeneracy mechanism is well-evidenced (per-class raw similarity distributions shown for every baseline), and the Type A label spot-check is credible — it shows five distinct, varied paraphrase excerpts (e.g., "IQGS" → "TalentForge," reworded sentences with different vocabulary each time), which reads as an actual manual inspection rather than a fabricated confirmation. Treat both the threshold mechanism and the label-integrity question as closed. No further action needed on this task specifically.

---

## Task 11 — Reconcile the α=0.8 vs. α=0.6 discrepancy (SUPERSEDED BY TASK 12 — raw-score request satisfied, but explanation surfaced a bigger issue)

**Status: The specific evidentiary request is now satisfied.** `alpha_provenance_resolution.md` finally shows actual raw per-pair scores for α=0.6 vs. α=0.8 (10 sample pairs, all bit-identical to 6 decimal places) — this is exactly the standard requested, and propagation to `significance_report_v2.md`/`NOVELTY_TEST_REPORT.md` is confirmed. Do not re-flag the "show raw evidence" request again; it's been met.

**But the mathematical explanation given for *why* they're identical is the actual finding, and it's bigger than this task.** α is reported to only gate a pre-filter, never entering the final similarity computation — meaning it doesn't blend TED and SBERT as documented in `PROJECT_OVERVIEW.md`'s Hybrid Score formula. This is no longer a hyperparameter-value question; it's a methodology-vs-implementation question. **See Task 12 (now first in execution order) for the actual resolution required.** This task (11) can be considered closed once Task 12 either fixes the implementation or corrects the documentation — no further action needed here specifically.

---

## Task 9 — Resolve the unaccounted 11 pairs

**Why:** `count_reconciliation.md` states 53 pairs originally required D3/D4 completion, but only 42 were GPT-regenerated. The fate of the other 11 is not stated. If they remain in the "138 real-only" bucket with incomplete D3/D4 sections, the original selection-bias problem this whole augmentation effort was meant to fix is still partially present inside what's now being called the clean primary benchmark.

- [ ] Determine explicitly: were the 11 pairs (a) dropped from the dataset entirely, (b) included in the 138 "real-only" set despite incomplete sections, or (c) something else?
- [ ] If (b): these 11 pairs must be excluded from the real-only benchmark, or their D3/D4 sections must be completed through the same (leakage-checked) process as the other 42, and the 138 count must be corrected accordingly.
- [ ] Document the resolution in `results/audit/count_reconciliation.md` (append, do not silently overwrite the existing content) and correct the 138 figure everywhere if it changes.

---

## Task 10 — Reconcile the current 180/138/42-pair dataset against the original paper's 200-pair FPT dataset (PARTIALLY RESOLVED for FPT; REOPENED for PURE)

**FPT part: substantially improved.** `dataset_version_reconciliation.md` now traces actual disk file locations (`data/dataset/pairs.csv`, etc.) rather than presenting an unverified narrative — this is closer to the standard requested. FPT's 180-pair count and its relationship to the 200-pair early draft is reasonably explained (200 was a draft target, 180 is the locked stratified benchmark). Acceptable, though a git-history check would still strengthen this if available.

**PURE part: a new, equally serious version of the same problem.** The report now states PURE has **582 pairs** (194 positive, 388 negative) — every prior document, including `PROJECT_OVERVIEW.md` and all previously-reported PURE ablation tables (F1=0.8684, 0.8929, 0.8739, etc.), worked from **200 pairs**. The report confirms PURE is "100% human-written" and "clean," but never explains *why the count nearly tripled*. This is the same category of unexplained change that triggered the entire FPT audit chain, now appearing on the dataset that was supposed to be the stable, uncomplicated comparison point.

- [ ] Trace why PURE's pair count changed from 200 to 582: was this a re-derivation using a different pairing strategy (e.g., all-pairs instead of a curated subset), a different filtering criterion, or something else? Apply the same standard as requested for FPT — actual provenance, not a plausible reconstruction.
- [ ] State explicitly whether the previously-reported PURE ablation results (Table 3.3, ablation Groups A-D) are still valid on the new 582-pair set, or whether they need to be regenerated. If regenerated, note that this may also change the PURE-side ablation conclusions (e.g., the "T4 helps FPT but not PURE" and "β-uniform hurts PURE significantly" findings were computed on the old 200-pair set).
- [ ] Document in `results/audit/dataset_version_reconciliation.md`, appending rather than overwriting the FPT-side resolution.

---

## Task 2 — Resolve the 0.9939 vs. 0.9707 provenance question (REOPENED AGAIN — prior resolution used a strawman comparison)

**Status: INCOMPLETE.** `beta_provenance_resolution.md` correctly ran the injection test on the 138-pair real-only set and got F1=0.9498 (not 0.9939) using schedule T2=0.0, T3=0.9, T4=1.0 — that part is good and can stand, pending Task 8 confirming the underlying harness isn't affected by the B1/B3 collision. However, the report then tested a so-called **"Drift Schedule (`PROJECT_OVERVIEW.md`)"** of T2=0.0, T3=0.6, T4=0.8 — but `PROJECT_OVERVIEW.md` Section 2.2 actually documents **T3=0.9, T4=0.8**, not T3=0.6. The comparison that concluded "β₄=1.0 is intentional and optimal, β₄=0.8 is a historical mistake" was run against a schedule that doesn't match what it claims to disprove, with an incorrectly-altered T3 value causing most of the observed collapse to F1=0.5135.

- [ ] Re-run the β₄ comparison using the **actual documented schedule**: T2=0.0, T3=0.9, T4=0.8 (holding T3 fixed at its correct documented value), compared against T2=0.0, T3=0.9, T4=1.0.
- [ ] Report both F1 results side by side. Only after this corrected comparison should any conclusion be drawn about whether T4=0.8 (the original paper's documented, theoretically-justified 80/20 canonical-keyword/schema-type split) or T4=1.0 is the better/intended setting.
- [ ] Update `beta_provenance_resolution.md` to replace the strawman "drift schedule" test with this corrected one, keeping a note explaining the correction was made.

**Acceptance criteria:** The β₄=0.8-vs-1.0 question is resolved using T3 held at its correct, documented value (0.9), not an altered one.

---

**Why:** `EVALUATION_RESULTS_UNBIASED.md` regenerated missing D3/D4 sections for 53 of 180 pairs using GPT-4o-mini and merged them into the main evaluation without disclosing the generation prompt, without checking for label leakage, and without reporting real-only vs. augmented-only results separately. Every number in that document is provisional until this is resolved.

- [ ] Locate and paste into `results/audit/gpt_augmentation_prompt.md` the **exact prompt(s)** used to regenerate D3/D4 sections. Check specifically whether the prompt referenced:
  - The plagiarism pair type (A/B/C) or the paired document's content
  - Any ground-truth label
  - Anything that could make the generated content non-independent of the label it will later be evaluated against
  - If the prompt did reference any of the above → **flag as leakage, do not use these 53 pairs in any reported result until regenerated with a leakage-free prompt** (e.g., "write a plausible D3/D4 section for a capstone project in domain X" with no reference to plagiarism status or paired documents)
- [ ] Determine whether regeneration was done **once** (fixed, reused across folds) or **independently per fold**. If fixed and reused, flag as a potential cross-fold leakage path and note it in the audit report even if no direct label leakage is found in the prompt itself.
- [ ] Re-run the full evaluation pipeline (all baselines + SW-BTED) on three separate slices, and report all three side by side in `results/audit/real_vs_augmented_breakdown.csv`:
  1. **Real-only** (the original 127 pairs, no GPT-regenerated content)
  2. **Augmented-only** (the 53 pairs with GPT-regenerated D3/D4)
  3. **Combined** (all 180 — the current `EVALUATION_RESULTS_UNBIASED.md` numbers, for comparison)
- [ ] If SW-BTED's or any baseline's F1/TNR differs meaningfully (e.g., >3 points) between real-only and augmented-only, that difference itself must be written up as a finding in `results/audit/audit_report.md` — do not average it away silently.

**Acceptance criteria:** A written record of the exact augmentation prompt, an explicit leakage determination (yes/no, with reasoning), and three separate result tables (real/augmented/combined) with no silent merging.

---

## Task 2 status note

**The core injection test (extract A1's β schedule, inject into `main_evaluation.py`, run on real data, compare to 0.9939) has now actually been executed** — see `beta_provenance_resolution.md`: result was F1=0.9498, not 0.9939, so 0.9939 is confirmed discarded as invalid/leaked. This part of the original Task 2 is DONE, pending Task 8's harness audit confirming this result isn't itself affected by the B1/B3 collision bug.

**What is NOT done and remains open:** the β₄=0.8-vs-1.0 sub-question, which was resolved using an incorrect strawman comparison. See the new **Task 2 (reopened again)** section above this point in the document for the corrected remaining work.

---

## Task 3 — Complete and correct the significance testing (RESOLVED)

**Status: Fixed correctly this round.** B1, B3, and B5 are reverted to their previously-verified 138-real-only values (0.4364, 0.4364, 0.4081), and their contingency tables/prose are internally consistent with those values again. The B2 correction (Full-Doc SBERT, F1=0.9867, p=0.3750) is retained correctly. The previously-discarded F1=0.9939 figure is gone from this table. No further action needed on the significance table itself.

**Remaining smaller items, carried forward, not blocking:**
- [ ] pq-Gram's F1 in this table (0.9479) still doesn't match the last fully-independent-verified figure from several rounds ago (0.9579) — minor, low-priority, but still technically unexplained if anyone wants to close it out.
- [ ] Section 3.B's "Flat Schema Baseline" claim is **not** a valid stand-in for Task 5's flat-baseline requirement — see Task 5 below, this is now that task's primary open item.

---

## Task 4 — Rewrite the results discussion to match what the numbers actually show

**Why:** Current framing in `EVALUATION_RESULTS_UNBIASED.md` Section 5 claims SW-BTED is "superior in practice" while its own table shows TF-IDF and pq-Gram matching or exceeding SW-BTED on precision, recall, TPR, and both TNRs. The discussion selectively compares against SBERT (which SW-BTED does beat on Type B TNR) while omitting TF-IDF and pq-Gram from that same sentence.

- [ ] Rewrite the results discussion (wherever it lives — paper draft, `PROJECT_OVERVIEW.md`, or a new `docs/research_notes/` file) to state plainly: SW-BTED is statistically indistinguishable from TF-IDF and SBERT on this benchmark (pending Task 2/3 results), and is not the best performer on domain-confusion resistance (pq-Gram and TF-IDF currently rank higher on Type B/C TNR).
- [ ] Any claim of "explainability" or "interpretability" advantage must be backed by an actual demonstration (see Task 5's qualitative case studies), not asserted without evidence in the same way the accuracy advantage was.
- [ ] Do not leave both the old ("superior") and new (honest) framing in different documents simultaneously — supersede the old framing explicitly.

---

## Task 5 — The flat-embedding baseline test (RESOLVED)

**Status: Fully closed, verified by hand.** The corrected Hybrid-vs-Flat statistic checks out exactly (100 discordant pairs → χ²=98.01, p=1.578×10⁻³⁰, matching the reported recomputation). The per-pair diffs against the old mislabeled flat baseline (mean abs diff 0.389) and against B5 (mean abs diff 0.857) are both strictly non-zero, confirming genuine code-path independence — the F1=0.4314 recurrence is a property of this dataset, not a residual bug.

**This is the first fully valid execution of the paper's central novelty test across the entire audit thread. Result: SW-BTED Structural-Only significantly and substantially outperforms a genuinely flat, non-structural domain embedding average** (F1=0.9498 vs. F1=0.4314, p≈2.52×10⁻²⁹ on the 138 real-only pairs), and the Hybrid Mode does as well (p≈1.58×10⁻³⁰, pending Task 12's remaining item on the hybrid mode's own credibility). No further action needed on this task specifically. This result should be carried into Task 4's discussion rewrite as the headline evidence answering the project's original "is this just structured BERT" novelty concern.

## Task 6 — Structural-perturbation benchmark (UNBLOCKED — Task 5 confirmed SW-BTED wins; this can now proceed)

**Why:** A flat embedding average is structurally blind — it can't detect content that has been reordered, moved between sections, or had subitems inserted/deleted while keeping terminology nearly unchanged. If tree-edit-distance matters, this is the condition where it should show it.

- [ ] Build synthetic perturbed variants of existing FPT/PURE documents:
  - Reorder content across intents (T3) within the same domain
  - Move content between domains (D1↔D2, etc.)
  - Insert/delete subitems (T3/T4 nodes) while preserving near-identical terminology overall
- [ ] Evaluate SW-BTED vs. the flat baseline (Task 5) vs. TF-IDF/SBERT on this perturbed set specifically.
- [ ] This is the experiment most likely to produce a genuinely novel, defensible empirical claim for the paper — treat its results as high-priority for the discussion section regardless of which direction they point.

---

## Task 7 — Relabel the augmented data honestly and reconcile pair counts (blocks Tasks 2, 3, 4, 5 use of the "combined" number)

**Why:** The GPT-4o-mini augmentation was described as filling in *missing* D3/D4 sections, but the actual prompt (`gpt_augmentation_prompt.md`) is a paraphraser applied to existing D3/D4 text, not a generator for absent content. All 42-53 augmented pairs are Type A (plagiarism-positive) by construction — the generation method itself defines the label, which is a structural leakage path that the prior leakage check (looking only for explicit label/pair-type mentions in the prompt text) did not catch. Additionally, two different pair counts have been reported (127 real + 53 augmented vs. 138 real + 42 augmented) with no explanation for the discrepancy.

- [ ] Reconcile the pair-count discrepancy: determine whether 127/53 or 138/42 is correct, why it changed, and document this in `results/audit/count_reconciliation.md`. If both are legitimate (e.g., different filtering criteria applied at different times), state which is now authoritative and why.
- [ ] Rename/relabel the augmented-data experiment throughout all documents: it is a **paraphrase-substitution robustness probe**, not a missing-data completion or a "selection bias fix." Update `EVALUATION_RESULTS_UNBIASED.md`, `AUDIT_AND_NOVELTY_SUMMARY.md`, and `audit_report.md` (or their successors) to reflect this framing.
- [ ] Remove the "combined" (real+augmented merged) number as the headline/primary result anywhere it currently appears. The primary result should be **real-only data alone**. The paraphrase-probe result may be reported alongside as a secondary, clearly-scoped finding ("SW-BTED and other semantic/structural methods detect GPT-paraphrase substitution; pure lexical methods (TF-IDF, pq-Gram) do not — this is expected given the generation method and should not be read as evidence of general paraphrase-plagiarism detection capability beyond this specific probe").
- [ ] Re-run Task 3's significance testing with the "combined" slice removed from primary conclusions (real-only and augmented-only can remain as separate rows, but no test result should be reported as coming from "the" dataset without specifying which slice).

**Acceptance criteria:** No document in the project claims a single "combined 180-pair" headline number without the paraphrase-probe framing attached. Pair counts are consistent and explained across all documents.

---

## Non-goals / guardrails

- Do not tune β, α, negative-sampling, or GPT-augmentation prompts post-hoc to make any of these tasks produce a more favorable result. Report what the data shows, including if Task 5 concludes the tree-edit-distance component isn't earning its complexity.
- Do not merge real and GPT-augmented data into a single reported number again without also reporting the split (Task 1's requirement applies to all future evaluations, not just this one).
- Do not carry forward "0.9939" or "superior in practice" language into any new document until Tasks 2 and 4 are formally closed with a written resolution.
- If Task 5 shows the flat baseline matches SW-BTED, do not read this as a reason to abandon the project — flag it to a human for a framing discussion (schema-taxonomy + interpretability + metric-guarantee contribution, de-emphasizing TED), do not silently drop the negative result from any report.
- **Do not mark a task's checkboxes as complete unless the exact dataset, exact experiment, and exact output specified in that task were what was actually run.** If a different dataset was used, a different metric was reported, or a step (e.g., a significance test) was skipped, the task is incomplete — report what was actually done and why, and leave the checkboxes unchecked, rather than substituting a related-but-different result and marking it done. This has happened twice in this project already (Task 2 and Task 5 were both marked complete without running the specified experiment) and must not happen again.
- **Any claim of the form "X outperforms Y" must be accompanied by a significance test in the same document**, not just a point-estimate delta, especially when confidence intervals/std values are already available and visibly close.
- **When two different methods/baselines produce identical results (same F1, same precision, same recall to several decimal places), this must be treated as a bug-investigation trigger, not reported as a finding.** Before writing any conclusion involving either method, diff their per-pair predictions and/or their implementations to confirm they are not accidentally computing the same thing. This exact situation has now occurred three times in this project.
- **A README or summary document claiming "Task N complete" must be checked against the actual referenced file, not taken on faith.** This project has now had three instances (Task 2, Task 3, and implicitly Task 5) where a task was marked complete in a summary document while the underlying file either ran the wrong experiment or was resubmitted unchanged from a prior, non-compliant version. When resolving a reopened task, always diff the new file against the previous version to confirm a real change was made, not just a status label.
- **A provenance/root-cause explanation must be traced from an actual source (git history, logs, original scripts), not reconstructed as a plausible-sounding narrative.** Task 10's "copy-paste conflation" story did not match the source document it was supposedly explaining and was accepted as resolved without being checked against `PROJECT_OVERVIEW.md`'s actual content. If the true cause cannot be found, state that explicitly ("root cause could not be determined") rather than writing an explanation that sounds plausible but wasn't verified against the original source.
- **When an audit surfaces a second, larger problem while investigating a narrower one, the larger problem must be treated as the actual finding, not footnoted and left for later.** Task 8's threshold-degeneracy discovery (baselines collapsing to an always-positive classifier) is more consequential than the code-duplication question it was found while investigating, and must be run down with the same priority, not mentioned in passing and left unresolved.
- **Whenever a canonical value or implementation is corrected (a hyperparameter like α or β, or a baseline like B5), the correction must be propagated to every other document in the same submission batch before that batch is considered complete — not just recorded in the file that discovered it.** This has now happened with α (declared 0.6 in one file, still 0.8 in two others in the same batch) and with B5 (fixed to F1=0.4081 in one file, still 0.4314 in two others in the same batch). Before submitting a batch of updated documents, run a simple consistency check: grep all files for the corrected value/number and confirm none of them still show the pre-correction version.
- **A report's date/log timestamp must postdate any audit it claims to have been validated by.** `NOVELTY_TEST_REPORT.md` was dated before the harness audit that was supposed to confirm its implementation was trustworthy — a report cannot be considered "verified" by an audit that happened after it was written, even if the conclusion later turns out to hold.
- **Every reported p-value must be sanity-checked as lying in [0, 1] before a document is submitted.** A p-value of 1.2734 appeared in a significance report and was not caught before submission — this is a trivial, mechanical check (not a judgment call) and should be automatic. Any test statistic or p-value outside its valid mathematical range invalidates that row and calls the entire computation method into question, not just that cell.
- **Updating a report's metadata (date, header, stated parameter value) is not a substitute for actually rerunning the underlying experiment.** This has now happened twice: `NOVELTY_TEST_REPORT.md` was redated to "post-date" an audit without its Flat-Baseline numbers changing, and the numbers remained byte-identical to the pre-audit version. If a number should plausibly change based on a fix elsewhere in the pipeline (e.g., a related baseline's bug fix) and it doesn't, that must be treated as a signal the rerun didn't actually happen, not as confirmation of robustness — investigate before accepting a "no change" result as-is.
- **A test statistic or p-value that recurs identically across independent comparisons, or across different audit rounds on different dataset versions, is a bug-investigation trigger with the same priority as the identical-F1 pattern already flagged.** The χ²=88.2551 value appearing for both the B1 and B3 comparisons in one round, and matching a value from several rounds earlier on a different dataset, needs the same per-comparison evidence trail (raw contingency tables) as previously required for tied aggregate metrics.
- **When a hyperparameter is found to have zero effect on output, the mathematical explanation for *why* must be checked against the documented methodology, not accepted as a benign robustness finding.** α=0.6 vs. α=0.8 being bit-identical was correctly investigated with raw scores, but the explanation given (α only gates a pre-filter, never entering the final score) revealed that the implementation may not match the published Hybrid Score formula at all. A parameter having "no effect" can mean the model is robust to it, or it can mean the parameter isn't wired into the computation the paper claims it's part of — these are very different findings, and only checking the raw numbers without checking the code path against the documented formula would have missed this.
- **A perfect or near-perfect result (F1=1.0000, zero standard deviation across folds, 100% recall/precision simultaneously) requires more scrutiny before acceptance, not less, especially when it contradicts evidence already established elsewhere in the same audit thread.** This has now happened twice: the original GPT-augmented paraphrase probe reported TPR=1.0000 for multiple methods (later understood as an artifact of how the positive class was constructed), and now the Hybrid Blend Mode reports F1=1.0000/std=0.0000 despite Task 8's own data showing the blended-in SBERT signal favors the wrong class on this dataset. Any future "perfect" result must be checked against prior findings in the same audit thread for consistency before being reported, and should trigger the same raw-score/threshold-independence/manual-spot-check protocol already established for suspicious ties.
- **When a "new" raw-score audit is offered as supporting evidence for a surprising result, its positive-class and negative-class numbers must each be checked against any prior audit of the same underlying signal on the same dataset — matching on one class while inverting on the other is a specific, high-value diagnostic, not just "different numbers."** The Hybrid Blend Mode's `sim_global` distribution matched Task 8's B2 audit exactly on Type A (positive) pairs but was essentially inverted on Type B/C (negative) pairs — this pattern (one class matches, the other doesn't) points precisely at an isolated bug in how negative-class pairs are retrieved/matched, and should be used to narrow the debugging target rather than treated as two independently-reported, equally-valid measurements.
- **A report claiming to "reconcile" or "resolve" a discrepancy must be checked against the actual numbers it claims to be reconciling, not just its own summary of them.** A resolution report quoted the previous round's finding incorrectly (0.3103 instead of the actual 0.2642) while presenting a third, different number in its own new table — meaning the "resolution" doesn't agree with either the thing it's resolving or itself. Before accepting any explanation for a discrepancy, verify the report's citation of the prior number is accurate.
- **A section or report title claiming to address a specific prior question must be checked against its actual content, not accepted on the strength of the title alone.** A section titled "Reconciliation of B2 Text Scope Across Project History" was expected to check the original `PROJECT_OVERVIEW.md` baseline (F1=0.9593) against this audit thread's newly-discovered text-scope distinction — it never mentioned that number at all, only re-described a comparison entirely internal to this audit's own reconstructed dataset. A title that echoes the assigned task is not evidence the task was done; read the content and confirm the specific number/question in dispute was actually engaged.
- **When a value changes between rounds (a "canonical" replacing a prior "canonical," or an F1 shifting on an unchanged score distribution), the change itself must be explained with a trace, not just presented as the new fact.** This has now happened multiple times with the same `sim_global`/hybrid-mode numbers specifically — each round asserts a new canonical value without reconciling why the previous one was wrong, which means the underlying computation still is not trusted to be stable or correctly understood, regardless of how confident the report's language is.
- **When an audit investigating one component (e.g., a hybrid formula) surfaces evidence bearing on a baseline or claim established earlier and outside the current audit thread's scope (e.g., the original paper's Table 3.2), the investigation's scope must extend to check that earlier claim too, not stop at the boundary of the current task.** The discovery that full-document SBERT alone may score far higher than previously reported (F1≈0.99 vs. the original 0.9593) raises the question of whether the *original* B2 baseline, reported in `PROJECT_OVERVIEW.md` since the first document in this project, was itself computed on the wrong text scope — this must be checked against the project's full history, not just within this audit thread's dataset reconstructions.
- **A significance/comparison claim ("SW-BTED beats X") must be re-verified any time the baseline's implementation, scope, or configuration is found to have changed or been reconstructed differently from the original — a "win" against a reconstructed or narrower version of a baseline is not evidence of a win against the baseline the paper actually reports.** This is exactly what happened with B2: significance tests across this entire audit thread showed SW-BTED beating "SBERT" by a huge margin, but that SBERT was a tree-label-only reconstruction invented within this audit, not the full-document-prose baseline the original paper used — and the correctly-scoped comparison turned out not to be significant at all. Any time a baseline gets rebuilt, re-scoped, or re-implemented during an audit, its prior significance comparisons must be treated as provisional until re-run against the corrected version.
- **When a file is resubmitted to fix one specific, requested item, every other row/number in that file must be checked against its previously-verified state before resubmission — fixing the requested item does not guarantee the rest of the file wasn't altered.** A request to correct B2's scope resulted in a file where B2 was fixed correctly, but B1/B3/B5 were also changed (apparently inadvertently) to values from a different dataset construction, including a figure that had been explicitly investigated and discarded as invalid in an earlier task. Every row in a resubmitted table must be diffed against its last-known-good state, not just the row that was the subject of the request.
- **A number that was explicitly investigated and formally discarded as invalid (e.g., via a branch-determination test with a documented conclusion) must never reappear in a later report without that discard being explicitly reopened and re-argued.** F1=0.9939 was formally ruled "an un-reproducible artifact" and discarded under Task 2; it reappeared, unacknowledged, attached to a different baseline under Task 3. A closed, documented resolution silently reversing itself is treated with the same severity as a brand-new bug, not as an incidental data refresh.
- **Any baseline described as "flat," "ablated," or "without component X" must have its exact formula checked to confirm component X is genuinely absent, not just reduced in weight.** A "Flat Schema Baseline" was reported as `0.5·sim_struct + 0.5·sim_global` — still containing the full tree-edit-distance term (`sim_struct`) the experiment was supposed to exclude, just at a different weight than the hybrid mode's 0.6. A baseline that still contains the component under test cannot answer whether that component is necessary; it can only confirm the component still helps at whatever weight it's given, which was never in question. This is the second time in this project a "flat"/comparison baseline turned out to still include the tree-alignment term (the first was the byte-identical duplicate of Section Cosine); this specific check — read the formula, confirm the term is fully absent — should be applied automatically to any future "flat baseline" or ablation claim.

## Execution order summary

1. **Task 5 — RESOLVED.** The novelty test is now valid and complete: SW-BTED Structural-Only significantly beats a genuinely flat, verified-independent domain embedding baseline (F1=0.9498 vs. 0.4314, p≈2.52×10⁻²⁹). This is the paper's central empirical claim and it now has real, verified support. No further action.
2. Task 3 — DONE.
3. Task 12 — mostly resolved; the narrower 0.2642-mechanism question remains open, low priority.
4. Task 8 — DONE.
5. Task 9 — DONE.
6. Task 10 — FPT part DONE; PURE's 200→582 jump still unexplained.
7. Task 11 — Effectively closed, superseded by Task 12.
8. Task 7 — DONE.
9. Task 2 — DONE.
10. **Task 6 — UNBLOCKED, now the next substantive item.** With Task 5 confirming SW-BTED beats the flat baseline, the structural-perturbation benchmark can proceed — this is the experiment that would show specifically *where* the tree-alignment advantage comes from (reordered content, moved subitems), strengthening the novelty story beyond the current dataset's natural pairs.
11. **Task 4 — next priority for write-up.** With Task 3 and Task 5 both resolved, there is now enough settled, verified evidence to write an honest results/discussion section: SW-BTED ties TF-IDF and Full-Doc SBERT in absolute terms on this dataset in some framings, decisively beats a flat embedding baseline in the controlled novelty test, and the hybrid mode's F1=1.0000 remains the one number still carrying some residual risk (see Task 12's outstanding mechanism item). Blocked only on Task 10's PURE reconciliation for full completeness.

**Net effect this round:** the single most important open question in this entire audit thread — does tree-edit-distance alignment add anything beyond flat embedding averaging — is now resolved with verified, hand-checked evidence. Combined with Task 3's resolution last round, the project has real, defensible empirical ground to stand on for the first time since this audit began. Remaining work (Task 6, Task 4, the PURE count, the 0.2642 mechanism) is meaningful but no longer load-bearing for the paper's core claim.
