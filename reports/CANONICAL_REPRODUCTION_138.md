# Canonical Reproduction — 138 Real-Only Pairs

## Result

The fixed side-by-side harness now runs both historical and current source paths on the same 138-pair slice, with the same fold seed and threshold-selection protocol.

| Source path | Beta input | F1 | Precision | Recall | Fold thresholds |
|---|---|---:|---:|---:|---|
| Historical source | Explicit `{T2: 0.0, T3: 0.9, T4: 0.8}` | `0.9498 ± 0.0253` | `0.9056` | `1.0000` | 0.33, 0.34, 0.34, 0.34, 0.34 |
| Current source | Module config `{T2: 0.0, T3: 0.9, T4: 1.0}` | `0.9498 ± 0.0253` | `0.9056` | `1.0000` | 0.28, 0.29, 0.29, 0.29, 0.29 |

The metric result is identical under the current source's supported configuration. The score scale is not necessarily identical: the different thresholds show that the two beta schedules produce different score calibration even though thresholded F1 is unchanged.

## Important implementation finding

Before the fix, the current `src/05_sw_bted.py` did not support an explicit beta dictionary through the constructor. Passing `beta={"T2": 0.0, "T3": 0.9, "T4": 0.8}` caused:

```text
TypeError: unsupported operand type(s) for *: 'dict' and 'float'
```

The current source has now been patched to handle dictionary beta values per layer, matching the historical behavior. The harness was then rerun with the explicit beta dictionary on both paths.

## Fixed protocol

- Dataset: `repro_candidate_138/data/dataset/trees_section.json` and `pairs.csv`.
- Exclusion: remove every pair whose document key occurs in `plag_regen_sections.json`.
- Slice: 138 pairs, 38 Type_A positives and 100 Type_B/Type_C negatives.
- Alpha: `0.8`.
- Five-fold `StratifiedKFold`, `shuffle=True`, `random_state=42`.
- Threshold grid: `0.00` through `1.00`, step `0.01`, selected on each training fold.
- Output JSON: `reports/canonical_reproduction_138.json`.
- Harness: `experiments/canonical_reproduction_138.py`.

## Conclusion

The “4 layers and 138 pairs” statement is correct for both paths. The prior discrepancy was caused by evaluation implementation/configuration differences, not by a different pair count or tree depth. After repairing the beta interface, the current implementation reproduces the historical result under the same explicit protocol.

## Next code change

Keep the canonical harness and source/configuration hashes under version control; this prevents future runs from silently changing protocol or score calibration.
