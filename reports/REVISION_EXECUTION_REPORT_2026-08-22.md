# Revision execution report — 2026-08-22

## Scope

Executed the approved high-priority revisions against the canonical four-layer, 138-pair benchmark. Existing canonical result files were preserved; new audits were written separately.

## Provenance checked

| Item | Artifact | Result |
|---|---|---|
| Pair file | `repro_candidate_138/data/dataset/pairs.csv` | 138 pairs; SHA-256 `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56` |
| Canonical source | `reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md` | Four-layer primary protocol confirmed |
| Primary result | `reports/audit/final_canonical_results_138.json` | Structural-only F1 ≈ 0.9498 |
| Document-disjoint audit | `experiments/document_disjoint_robustness_138.py` | Re-run completed successfully |

## Executed audits

### 1. Document-disjoint robustness

Re-ran the connected-component `GroupKFold(5)` audit with train-fold-only threshold selection at step 0.005.

| Method | Mean fold F1 | Std | Pooled F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| SW-BTED | 0.9160 | 0.0928 | 0.9157 | 38 | 7 | 93 | 0 |

The result is supplemental, not a replacement for the canonical pair-level result. Its higher variance is now disclosed in the manuscript.

### 2. Embedding input/truncation audit

The exact canonical `full_texts.json` strings were tokenized against the cached model snapshots.

| Model | Effective max length | Documents over limit |
|---|---:|---:|
| MiniLM | 256 | 159/178 (89.3%) |
| BGE-small | 512 | 103/178 (57.9%) |
| MPNet | 384 | 124/178 (69.7%) |

This is a limitation of the stated embedding protocol. It does not invalidate the historical result, but it limits claims about unlimited-document semantics.

### 3. Schema-matched embedding baseline

Ran an independent flat baseline using the same D1–D4 text blocks, MiniLM pooling, unweighted mean of four domain cosine scores, 5-fold stratified CV, and train-fold-only threshold grid 0.005.

Result: **F1 = 0.4314 ± 0.0160; TP=38, FP=100, TN=0, FN=0.** This matches the established genuine flat-domain baseline and confirms that schema decomposition without tree alignment does not explain the structural result.

## Manuscript changes applied

- Narrowed the theorem claim from whole-framework “metric-preserving” to the unscaled node-level convex-combination proposition.
- Added the schema-matched baseline protocol and result.
- Added document-disjoint robustness results and the embedding truncation limitation.
- Removed the third-domain TODO and replaced it with an explicit future-work limitation.
- Replaced overbroad “genuine structural interpretability” language with “inspectable structural attribution through subtree edit traces.”

## New artifacts

- `experiments/audit_embedding_input_scope_138.py`
- `reports/audit/embedding_input_scope_audit_138.json`
- `reports/audit/EMBEDDING_INPUT_SCOPE_AUDIT_138.md`
- `experiments/schema_matched_embedding_baseline_138.py`
- `reports/audit/schema_matched_embedding_baseline_138.json`
- `reports/audit/schema_matched_embedding_pair_scores_138.csv`
- `reports/SCHEMA_MATCHED_EMBEDDING_BASELINE_138.md`

The manuscript updated is `draft/SW_BTED_FULL_DRAFT_CITED_V2.md`.

## Remaining blockers before submission

1. Replace the manuscript repository TODO with the final public repository URL and reproducibility pointer.
2. Decide whether the document-disjoint audit is presented as a prominent secondary result or becomes the primary estimate; that is a methodological decision for the supervisor/reviewers.
3. If claiming comparison with current-generation embeddings, run them under the same explicit long-document protocol; otherwise retain the limitation wording.
