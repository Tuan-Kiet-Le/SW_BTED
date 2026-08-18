# Qwen3 Prediction Audit

## Verdict

Claude's verification request was appropriate. The downloaded Kaggle artifact was independently audited using the same pair order, 138-pair filter, five-fold split (`shuffle=True`, `random_state=42`) and train-fold-only threshold selection.

There is no evidence that Qwen3 reused the MiniLM embedding cache:

- Qwen3 cosine scores and independently recomputed MiniLM cosine scores differ substantially.
- Pair index 84 (zero-based) has Qwen3 cosine `0.584966` and MiniLM cosine `0.515184`.
- The Qwen3-only error is a specific Type_B pair, not a copied prediction vector.

One protocol nuance was also exposed: the independent MiniLM recomputation from
the downloaded data gives F1 `1.0000` with a 0.01 threshold grid, whereas the
historical manuscript path reports `0.9867`. Re-running both score-rounding
variants shows that four-decimal rounding does not cause the difference. The
decisive difference is the historical 0.005 threshold grid: it selects 0.595 in
one fold and admits `SP26SE068`–`SU26SE063` (cosine `0.5978018`) as a false
positive. The manuscript comparison must freeze score precision, threshold-grid
resolution, and text-construction details explicitly.

## Qwen3 out-of-fold confusion matrix

Using the fold-specific threshold selected on each training fold:

```text
TP = 38
FP = 1
TN = 99
FN = 0
```

Pooled out-of-fold metrics are:

```text
F1        = 0.9870129870
Precision = 0.9743589744
Recall    = 1.0000000000
```

The manuscript-style mean across five test folds is:

```text
F1 = 0.9866666667 ± 0.0266666667
```

The small difference between pooled F1 and mean-fold F1 is expected: they aggregate predictions differently.

## Fold thresholds and metrics

| Fold | Threshold source | Threshold | F1 | Precision | Recall |
|---:|---|---:|---:|---:|---:|
| 1 | Train fold | 0.66 | 1.0000 | 1.0000 | 1.0000 |
| 2 | Train fold | 0.66 | 1.0000 | 1.0000 | 1.0000 |
| 3 | Train fold | 0.66 | 1.0000 | 1.0000 | 1.0000 |
| 4 | Train fold | 0.63 | 0.9333 | 0.8750 | 1.0000 |
| 5 | Train fold | 0.66 | 1.0000 | 1.0000 | 1.0000 |

No threshold was selected on the corresponding test fold.

## Pair index 84, zero-based

```text
doc_a       = SU26SE087
doc_b       = SP26SE001
type        = Type_B
label       = 0
Qwen3       = 0.5849661827
MiniLM      = 0.5151844621
Qwen thresh = 0.66
Qwen pred   = 0
```

## Qwen3 error

The only Qwen3 out-of-fold error is:

```text
doc_a       = SU26SE048
doc_b       = SU26SE087
type        = Type_B
label       = 0
Qwen3       = 0.6538107991
fold thresh = 0.63
prediction  = 1
MiniLM      = 0.5147178173
```

This explains why Qwen3 and the manuscript's MiniLM result can share the same mean-fold F1 while producing different raw similarities. They have different score geometries but converge to a similar one-error classification outcome.

## Artifacts

- `reports/qwen3_prediction_audit.json`
- `reports/qwen3_pair_prediction_audit.csv`
- Original Kaggle artifact: `kaggle/qwen3_results/qwen3_results/qwen3_pair_scores.csv`
- Original provenance: `kaggle/qwen3_results/qwen3_results/provenance_manifest.json`
- Audit runner: `experiments/audit_qwen3_predictions.py`

## Manuscript action

The Qwen3 result may be reported as a modern-baseline parity result, but the paper should specify that `0.9867 ± 0.0267` is the mean of five fold F1 scores. If a confusion matrix is reported, use the out-of-fold matrix above and label it explicitly as pooled OOF predictions.
