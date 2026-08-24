# GitBugs adaptation-protocol audit

## Current evidence

The available cross-domain report records 300 pairs and reports:

- SBERT full-text: F1 `0.9074 ± 0.0304`;
- SW-BTED structural-only: F1 `0.6725 ± 0.0194`;
- SW-BTED hybrid after adaptation (`α=0.6`): F1 `0.9141 ± 0.0348`;
- hybrid confusion matrix: `TP=91, FP=8, TN=192, FN=9`.

## Provenance gap

The current workspace does not contain a canonical raw GitBugs pair file, split manifest, or a machine-readable record specifying:

1. which hyperparameter was tuned beyond the stated `α=0.6`;
2. source and target values/search space;
3. the pairs used for tuning;
4. the held-out test pairs;
5. whether hard/easy negatives were kept separate during tuning and evaluation.

Therefore the `0.9141` result is retained as an exploratory cross-domain artifact, but it is not promoted here to a confirmed held-out generalization result.

## Required evidence before submission

Provide the raw GitBugs pair/split files and a runner that records the adaptation parameter, search space, tuning split, test split, per-category metrics, confusion matrices, and paired significance tests. Until then, manuscript wording should remain “exploratory transfer with adaptation.”
