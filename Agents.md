# SW-BTED Agent Working Instructions

## 1. Source of truth

This project is currently a **four-layer** SW-BTED implementation. Do not implement or migrate to the six-layer specification in this file's previous version.

The authoritative research artifacts are:

- Manuscript: `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`
- Primary evaluation scope: **138 real-only document pairs**
- Current tree representation: `Root → Domain → Intent → Terminology`
- Main result to reproduce: structural-only SW-BTED F1 ≈ `0.9498`

The 180-pair branch containing 42 GPT-generated sections is audit material only. It must not replace the 138-pair real-only result as the manuscript's primary evidence.

`RAG_Research` is the historical experiment workspace. Its files may contain multiple incompatible runs. Never assume that the newest file or newest result is the one used by the manuscript.

## 2. Immediate objective: provenance and reproducibility

Before modifying code, identify the exact source files, configuration, dataset files, experiment script, and archived result that produced each number reported in the manuscript.

### 2.1 Timestamp-first triage

Start the investigation with filesystem timestamps. For every candidate source, experiment script, configuration file, dataset artifact, and result file:

1. Record `LastWriteTime`, file size, and absolute path.
2. Sort candidates by modification time around the timestamp of the manuscript result or archived evaluation run.
3. Inspect the closest candidates first, especially files modified immediately before the result files.
4. Compare timestamps against the experiment history directory and report-generation files.
5. Treat timestamps as a search heuristic only; a file is accepted as provenance only after code references, configuration, dataset inputs, and output schema also match.

Do not assume that the newest source file produced the newest result. A later edit may have happened after the experiment, and an older archived script may have produced the manuscript result.

The agent must first create a provenance table with at least:

| Manuscript item | Required provenance |
|---|---|
| 138-pair dataset | pair file, labels, document/tree files |
| SW-BTED result | source file, config, alpha/beta, threshold, budget |
| Baseline results | implementation, text/tree input scope, threshold protocol |
| Ablation results | script, variant definition, output file |
| Perturbation benchmark | generator, labels, evaluation output |
| Statistical tests | prediction file, test script, correction method |

Do not overwrite or copy source files until this table is complete.

## 3. Four-layer model

The manuscript's current representation is:

```text
T1 ROOT
└── T2 DOMAIN
    └── T3 INTENT
        └── T4 TERMINOLOGY / NORMALIZED KEYWORD
```

The implementation may use project-specific names such as `CapstoneNode`, `schema_class`, `IntentMatching`, and `TerminologyVerification`. Preserve the names used by the result-producing code when reproducing historical results; rename only after reproducibility is established.

## 4. Reproduction order

Run only after the user reviews the provenance table and approves execution.

1. Validate Python version, dependencies, spaCy model, sentence-transformer model, APTED, CSO data, and working directory.
2. Validate dataset counts, pair labels, document IDs, and tree counts.
3. Run a small smoke test on fixed pairs and save its output separately.
4. Re-run the exact 138-pair experiment with the recovered source and configuration.
5. Compare regenerated metrics against the manuscript, including per-fold values and thresholds.
6. Run leakage and pre-filter diagnostics.
7. Only then compare the recovered source with the current workspace source.

The 180-pair audit branch, GPT-generated sections, bug-report branch, and unrelated job-description branch must be run separately and clearly labeled.

## 5. Rules for interpreting results

- A result is **reproduced** only when code, data, configuration, and evaluation protocol are all identified and the regenerated value is within an explicitly reported tolerance.
- Do not claim superiority over a baseline when the manuscript's own corrected tests show statistical parity.
- Keep full-document SBERT, genuine flat/domain SBERT, TF-IDF, Section Cosine, Standard TED, and pq-Gram distinct. Always record their input scope.
- The root pre-filter is a computational shortcut, not a ground-truth classifier. Report how many positive and negative pairs fall below the threshold.
- Treat the structural-perturbation benchmark as a controlled diagnostic benchmark. State that its labels are constructed by design.
- Do not claim that `1 - cosine` is automatically a mathematical metric without checking the assumptions required by the proof.

## 6. Source selection policy

The canonical four-layer source set is expected to include only files that are proven to participate in the 138-pair manuscript run, typically:

```text
src/01_parser.py
src/02_keyword_extractor.py
src/03_ontology_lookup.py
src/03_normalizer.py
src/04_tree_builder.py
src/05_sw_bted.py
src/baselines.py
src/06_evaluate.py
experiments/main_evaluation.py
experiments/ablation_study.py
config.yaml
```

This is a candidate list, not an authorization to copy every file. A file is copied into the canonical workspace only after provenance verification.

Files from experimental branches must be placed in an explicitly named archive or marked as non-canonical. Do not silently replace a result-producing file with a newer variant.

## 7. Manuscript review checklist

Before submission, verify:

- no unresolved TODOs or placeholder figures/tables;
- 138-pair primary results are consistent across abstract, tables, discussion, and conclusion;
- 180-pair/GPT results are labeled as secondary audit evidence;
- every reported metric has a reproducible output file;
- baseline input scope and threshold selection are explicit;
- per-fold metrics, confidence intervals, and corrected significance tests are available;
- metric-preservation claims match the actual distance functions used;
- perturbation labels and limitations are disclosed;
- repository structure, environment, model versions, and dataset provenance are documented.

## 8. Stop conditions

Stop and report before running or changing anything if:

1. the 138-pair source and result cannot be uniquely identified;
2. a result depends on GPT-generated content but is presented as real-only evidence;
3. the recovered code produces materially different results and the cause is unknown;
4. source files from different experimental branches are being mixed;
5. a destructive overwrite or deletion would be needed.

The first deliverable is therefore a provenance report and a proposed canonical source manifest. Implementation changes come only after that report is reviewed.
