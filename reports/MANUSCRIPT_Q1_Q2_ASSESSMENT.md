# Independent Manuscript Assessment

**Manuscript:** `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`  
**Reviewed:** 2026-08-14  
**Comparison feedback:** `docs/feedback/claudeFeedback.md`

## Executive verdict

The manuscript is not yet submission-ready for a serious Q1 venue. It is potentially defensible for a Q2 software-engineering/information-retrieval journal after a focused revision, but the current draft is better described as **promising Q2 candidate, not yet Q2-ready**.

Q1/Q2 is not an intrinsic label attached to a manuscript: it depends on the target journal, subject category, ranking database and year. Without a named venue, the most objective conclusion is a readiness assessment rather than a quartile prediction.

## Assessment by dimension

| Dimension | Current assessment | Effect on venue readiness |
|---|---|---|
| Core idea | Clear and potentially publishable: schema-weighted TED with interpretable alignment | Positive |
| Theoretical argument | Useful convex-combination proposition, but its instantiated distances need more careful qualification | Moderate concern |
| Main reproduction | Strong: current and historical four-layer paths match on 138 pairs | Positive |
| Dataset scale | 138 capstone pairs and 20 perturbation pairs are modest | Q1 concern; Q2 concern unless framed carefully |
| Baselines | Several baselines and honest parity reporting; current-generation embedding baseline still absent | Major revision |
| Ablation | No complete, clean beta/alpha/layer ablation in the manuscript | Major revision |
| Statistical validity | 5-fold CV and paired tests are good; document-level independence and repeated-pair effects need explicit analysis | Moderate/major revision |
| Reproducibility | Repository/data pointer is still TODO | Blocking for submission |
| Figures | Three figure placeholders remain | Blocking for polished submission |
| Runtime | Complexity is stated but no empirical timing/scaling evidence | Important revision |
| Writing/presentation | Strong framing and unusually honest claims; unfinished TODOs visibly reduce readiness | Major revision |

## Claude feedback: what is correct

The following points are substantively correct:

1. A clean ablation study is missing. The paper chooses alpha and per-layer beta values, but does not show sensitivity or justify that the result is not an artifact of those settings.
2. A modern embedding baseline is absent. The manuscript itself still contains a TODO for a 2025–2026 model.
3. Figures are placeholders rather than submission-quality figures.
4. The reproducibility section is incomplete and still contains a repository-link TODO.
5. There are no empirical runtime measurements despite the complexity discussion.
6. The natural-document dataset is modest, so claims should remain bounded and generalization should not be overstated.
7. Q2 is more realistic than Q1 for the current contribution shape: the strongest story is structural sensitivity, interpretability and parity with strong embedding baselines, not universal accuracy superiority.

## Claude feedback: what is outdated or needs correction

### D1–D4 citation gap

Claude's statement that the D1–D4 citation remains unresolved is outdated. The current manuscript's alignment log records the follow-up grounding pass and adds Robertson & Robertson, IIBA and PMI references. It also explicitly presents D3 as the authors' own extension rather than pretending that one external taxonomy exactly defines all four domains.

This does not mean the taxonomy is beyond criticism. The paper should still explain why the four categories are appropriate for this dataset and should report whether alternative reasonable schemas change the result.

### Citation count

The feedback says 27 citations. The current manuscript says 30 verified citations after the D1–D4 follow-up. The underlying concern about source alignment is mostly addressed, although every reference should still be checked in the final submission format.

### “Metric preservation” claim

The proof sketch is mathematically valid under its stated premise: a nonnegative convex combination of two metrics is a metric. The manuscript should not silently imply that every instantiated component is automatically a metric. In particular, `1 - cosine similarity` is not generally a metric on arbitrary vectors, and a manually specified schema-distance matrix must itself satisfy metric properties if the full metric claim is to be made literally.

Recommended wording: “the construction preserves metricity whenever the component distances used at a layer are metrics; our implementation guarantees bounded replacement costs and uses the construction as a principled cost design.” If the authors want the stronger claim, they should validate or replace the component distances with known metrics.

## Main technical risks still requiring work

1. **Ablation and parameter selection.** Run a fresh, preregistered-style grid on the fixed 138-pair split: alpha, beta schedules, removal of schema/content terms, and domain/layer removal. Threshold selection must remain inside each training fold.
2. **Data independence.** Report the number of unique documents, document reuse across pairs and a document-disjoint evaluation if possible. Pair-level random folds can put related documents or the same project into train and test.
3. **Perturbation benchmark scope.** Twenty synthetic pairs are useful as a diagnostic, not as broad evidence of real-world performance. State clearly that the benchmark tests a designed structural property.
4. **Baseline fairness.** Run at least one current embedding model with the same pair split, threshold protocol and leakage controls. Include model version and local artifact hash.
5. **Reproducibility.** Publish the exact 138-pair manifest, tree JSON, config, source commit/hash, model identifier, fold seed and commands.
6. **Runtime/scaling.** Measure parser time, embedding time, APTED time and peak memory across increasing tree sizes; compare against flat embedding inference.

## Objective readiness judgment

### Q1

**Not ready and unlikely for the current revision.** The missing ablation, unfinished reproducibility package, absent modern baseline, small primary dataset and placeholder figures would be serious first-round weaknesses. A Q1 attempt would require a stronger empirical package, clearer metric-theoretic qualification, document-disjoint validation and polished artifacts.

### Q2

**Plausible after major revision; not ready for submission today.** The core method, exact reproduction and honest positioning provide a credible Q2 foundation, especially for a venue valuing software-engineering artifacts, interpretable similarity or structured document analysis. The manuscript should not be submitted until the TODOs are removed and the ablation/baseline/runtime package is complete.

## Recommended order of work

1. Freeze the 138-pair canonical protocol and publish its manifest/hashes.
2. Run the clean alpha/beta/layer ablation.
3. Add one modern embedding baseline and rerun the exact protocol.
4. Audit document-level independence and add a document-disjoint split if feasible.
5. Add runtime/scaling experiments.
6. Generate Figures 1–3 and remove every TODO.
7. Tighten the metric-preservation wording and clarify the scope of the perturbation result.
8. Reassess against a specific target journal's aims, recent papers and quartile.

## Questions

No blocking clarification is required to begin the revision. A target journal would be needed later for a venue-specific Q1/Q2 fit assessment, because quartile status varies by journal category and ranking year.
