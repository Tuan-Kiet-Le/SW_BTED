# Regression Test Report

**Date:** 2026-08-14

## Result

All current regression tests pass:

```text
Ran 6 tests in 0.002s
OK
```

Python bytecode compilation also passed for `src`, `experiments` and `tests`.

## Coverage added

- Parser bullet cleanup and group-header detection.
- Four-domain tree creation.
- Preservation of `raw_text`, `normalized_text` and `feature_label`.
- Per-layer beta dictionary handling.
- Replace-cost boundedness by delete-plus-insert cost.
- Protection of the canonical 138-pair F1 result.

## Files

- `tests/test_regression.py`
- `experiments/canonical_reproduction_138.py`
- `reports/canonical_reproduction_138.json`

## Remaining validation

The tests cover the cost engine and representative parser behavior. They do not yet validate every DOCX template, spaCy model output, ontology lookup, or a newly generated dataset end-to-end. The next meaningful test is a small DOCX fixture suite covering empty sections, alternate headings, multiple actors, mixed Vietnamese/English text, and unknown technologies.
