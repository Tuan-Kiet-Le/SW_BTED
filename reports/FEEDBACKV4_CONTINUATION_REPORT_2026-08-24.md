# Feedback v4 continuation report — 2026-08-24

## GitBugs

The workspace contains summary reports and implementation code, but no canonical raw GitBugs pair file, split manifest, or machine-readable tuning/test record. The result `F1=0.9141 ± 0.0348` therefore remains exploratory transfer evidence. It is not promoted to confirmed held-out generalization.

## Qwen3

The successful Kaggle version 10 artifact records model ID, `max_length=2048`, device, dtype, pair counts, fold thresholds, and actual tokenizer lengths. It found 18/178 documents above the configured cutoff. The manuscript now reports Qwen3 as a 2048-token truncation protocol.

## Files changed

- `kaggle/run_qwen3_embedding_baseline.py`
- `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`
- `reports/QWEN3_TOKENIZER_AUDIT_STATUS.md`
- `reports/QWEN3_VERSION10_AUDIT_REPORT_2026-08-24.md`

## Next external action

Provide the original GitBugs pair/split files if the cross-domain result is to remain in the main manuscript.
