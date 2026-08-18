# Submission Timeline and Venue Plan

**Planning date:** 2026-08-14

## Recommended target

### Primary target: Empirical Software Engineering (Springer)

This is the best journal target for the current paper's shape. The contribution is primarily an empirical software-engineering study with a method, controlled comparisons, cross-domain evaluation, interpretability evidence and reproducibility concerns. Current public ranking profiles report ESE as Q1 by Scopus/CiteScore and Q2 in the 2026 JCR software-engineering category; the exact quartile depends on the ranking system and category. The journal has no conference-style fixed annual deadline, which gives enough time to finish the missing experiments instead of submitting an unfinished draft.

This should be treated as an ambitious Q2/JCR-Q2 target, not a guaranteed acceptance. The paper will need a much stronger empirical package than the current manuscript before submission.

### Conference alternative: SANER 2027 Research Track

SANER is a good topical fit for software analysis, parsing, empirical software engineering and AI for software engineering. Its 2027 Research Track requires a full paper of at most 10 pages plus up to 2 reference-only pages, uses double-anonymous review, and explicitly evaluates novelty, significance, soundness, open science/verifiability and presentation. The official page lists the mandatory abstract deadline as 21 September 2026 and the paper deadline as 25 September 2026 AoE. That leaves about six weeks from the planning date.

SANER is a viable fast-track option only if the paper is reframed explicitly as a software-analysis problem. The current capstone-proposal framing may need a stronger connection to requirements analysis, compliance checking, software artifact evolution or maintenance. Otherwise, a journal submission to ESE is safer.

## Time estimate

The main APTED evaluation is not the time bottleneck. Previous local runs completed approximately as follows:

| Workload | Observed/estimated compute time |
|---|---:|
| Canonical 138-pair structural comparison | ~15 seconds |
| Full-document MiniLM embedding reconciliation | ~40 seconds after model was cached |
| Candidate hybrid harness | ~4–5 minutes |
| Modern Qwen embedding inference for ~178 unique documents | likely minutes to a few hours including setup, batching and reruns on Kaggle GPU |

The time-consuming work is experimental design, debugging, audit, writing and figure production rather than raw APTED computation.

### Realistic focused schedule

| Phase | Work | Estimated elapsed time |
|---|---|---:|
| 1 | Freeze manifests, hashes and protocol; prepare Kaggle notebook | 0.5–1 day |
| 2 | Clean alpha/beta/layer ablation with fold-safe threshold tuning | 1–2 days |
| 3 | Qwen3-Embedding-4B baseline on Kaggle, export embeddings and rerun metrics | 1–2 days |
| 4 | Document-disjoint audit, leakage checks and runtime/scaling benchmark | 1–2 days |
| 5 | Generate figures and tables; resolve metric-theory wording | 1–2 days |
| 6 | Reorganize manuscript to target format, reproducibility package and final audit | 2–3 days |

**Focused total:** roughly 7–12 working days.  
**Safer total with failed runs and review cycles:** 3–4 weeks.

For SANER's 25 September deadline, this is technically possible but leaves little buffer. For ESE, the safer plan is 3–6 weeks before submission.

## Kaggle plan for the modern embedding baseline

Use Qwen3-Embedding-4B only as a baseline, not to replace the current MiniLM result. The official model card identifies it as a 4B-parameter embedding model with 32K context, 100+ language support and configurable embedding dimensions. Qwen's published speed/memory table reports roughly 8–12 GB GPU memory for Transformers BF16 at representative sequence lengths, with lower memory for FP8/AWQ; a Kaggle 16GB GPU should be treated as the practical target, with conservative truncation and batch size 1–4.

The notebook should:

1. Load exactly the 178 unique documents in the real-only 138-pair slice.
2. Use one explicit English task instruction for all documents.
3. Record model revision, dtype, quantization, max length, pooling/output dimension and batch size.
4. Export embeddings plus a SHA-256 manifest, not only final F1 values.
5. Run the same pair order, folds, threshold selection and statistical tests as MiniLM.
6. Keep Kaggle inference separate from local APTED scoring; only the embedding artifact needs to move between environments.

The 40-hour monthly quota should be more than sufficient for this dataset if the notebook is efficient. Reserve time for one pilot, one full run and one verification run rather than spending the quota on broad uncontrolled tuning.

## Decision rule

- If ablation, modern baseline and document-disjoint audit are complete within two weeks: prepare an ESE submission and optionally a SANER submission if the framing meets its software-evolution/analysis scope.
- If those experiments expose instability or leakage: do not submit yet; revise the dataset/protocol first.
- Do not claim Q1 readiness solely because one F1 value is high. The decisive remaining evidence is robustness, independence, modern-baseline fairness and reproducibility.
