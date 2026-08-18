# Manuscript vs clean canonical results reconciliation

Date: 2026-08-14

## Current clean primary results

These values come from the clean embedding vectors and the canonical SW-BTED run on 138 pairs. All embedding baselines use five-fold stratified CV, seed 42, a 0.005 threshold grid, and train-fold-only threshold selection.

| Method | Clean mean F1 ± std | Precision | Recall | TP | FP | TN | FN | Exact McNemar p vs SW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SW-BTED Structural-Only | 0.9498 ± 0.0253 | — | — | 38 | 4 | 96 | 0 | — |
| Full-document SBERT | 0.9867 ± 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 | 0.375 |
| BGE-small-v1.5 | 0.9882 ± 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 | 0.375 |
| MPNet-base-v2 | 0.9882 ± 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 | 0.375 |

The clean values supersede the old manually anchored BGE/MPNet values. They do not change the qualitative conclusion: SW-BTED is not superior in natural-document F1, but is statistically at parity under the paired test.

## Required manuscript edits

### 1. Section 5.1 results table

Current lines 160–162 report:

```text
BGE-small-v1.5  0.9737 ± 0.0267
MPNet-base-v2   0.9610 ± 0.0275
```

Replace with:

```text
BGE-small-v1.5  0.9882 ± 0.0235
MPNet-base-v2   0.9882 ± 0.0235
```

Keep SBERT at `0.9867 ± 0.0267` and the natural-document parity wording. The p-values can remain `0.3750` if the manuscript reports the exact paired comparison used in the clean reconciliation, but the calculation/provenance file must be cited internally in the repository.

### 2. Threshold protocol

The methods section currently says thresholds are selected per fold but does not state the grid resolution. Add:

```text
For all embedding baselines, the threshold was selected on the training fold over
the grid 0.00, 0.005, ..., 1.00; the test fold was never used for threshold tuning.
```

### 3. Raw vector provenance

Do not cite `reports/audit/raw_prediction_vectors_138.json` as a clean raw-output artifact. Its generating script manually overwrote selected scores. Use:

```text
reports/audit/clean_raw_embedding_vectors_138.json
reports/audit/clean_embedding_evaluation_138.json
```

The manuscript should not report SBERT cosine `0.6555` as a reproducible model output. The clean value for `SU26SE087–SP26SE001` is `0.5151844621`.

### 4. Case-study provenance

Section 5.5 should identify the pairs explicitly:

```text
Case A: SU26SE102–SU26SE102_plag
```

The pair is retained in the 138-pair evaluation because neither document occurs in `plag_regen_sections.json`. The manuscript should state that exclusion is based on explicit regen-key membership, not merely the `_plag` suffix.

### 5. Claims that can remain unchanged

The following conclusions remain supported by the clean reconciliation:

- SW-BTED Structural-Only: `0.9498 ± 0.0253`.
- SW-BTED is statistically at parity with strong natural-document embedding baselines.
- The structural-perturbation result is a controlled diagnostic benchmark.
- Structural interpretability is the principal qualitative contribution rather than raw natural-document F1 superiority.

## Do not edit until final review

This file is a change map only. The manuscript itself has not been modified. Before editing it, the remaining baseline rows (TF-IDF, Section Cosine, Standard TED, pq-Gram, and genuine flat/domain SBERT) should be checked against their own clean prediction files so that one final table is updated consistently.
