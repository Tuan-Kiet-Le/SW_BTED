# Frozen Scientific Manifest — SW-BTED v2

Freeze date: 2026-08-17  
Primary scope: 138 real-only pairs  
Pair-file SHA-256: `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`

## Frozen decisions

- Representation: four layers, Root → Domain → Intent → Terminology.
- Primary split: 5-fold `StratifiedKFold`, shuffle=True, seed=42.
- Threshold selection: training fold only, grid step `0.005`.
- Primary SW-BTED configuration: `beta_T2=0.0`, `beta_T3=0.9`, `beta_T4=0.8`;
  structural-only result reported separately from hybrid mode.
- Clean lexical protocol: full-document fields from canonical `full_texts.json`.
- Historical tree-label lexical values are legacy/input-scope ablations and are
  not mixed with the clean baseline table.
- Qwen3: raw downloaded pair scores, name-matched by `(doc_a, doc_b, label,
  type)`, threshold re-evaluated on the `0.005` grid.
- Perturbation benchmark: 20 constructed D2↔D3 swaps, secondary diagnostic.
- Document-disjoint robustness: supplemental GroupKFold by pair-graph
  connected component; not a replacement for primary pair-level results.

## Primary result map

| Evidence | Canonical result | Artifact |
|---|---:|---|
| SW-BTED structural-only | F1 `0.9498 ± 0.0253` | `reports/FINAL_CANONICAL_RESULTS_138.md` |
| Clean TF-IDF | F1 `0.9867 ± 0.0267` | `reports/CLEAN_BASELINE_SUITE_138.md` |
| Clean Section Cosine | F1 `0.6837 ± 0.0894` | `reports/CLEAN_BASELINE_SUITE_138.md` |
| Standard TED | F1 `0.4364 ± 0.0162` | `reports/CLEAN_BASELINE_SUITE_138.md` |
| Structural perturbation | 100% vs 0% | `reports/STRUCTURAL_PERTURBATION_REPRODUCTION_20.md` |
| Qwen3, 0.005 grid | pooled F1 `0.9870` | `reports/qwen3_threshold_protocol_audit_005.json` |
| Document-disjoint SW-BTED | pooled F1 `0.9157` | `reports/DOCUMENT_DISJOINT_ROBUSTNESS_138.md` |
| Beta ablation | documented schedule F1 `0.9498 ± 0.0253` | `reports/ABLATION_138_CLEAN_005_REPORT.md` |

## Non-canonical historical values

Legacy tree-label TF-IDF `0.4364` and legacy tree-label Section Cosine `0.4081`
remain useful for provenance analysis. They must be labeled as a different
input scope, not presented as the clean full-document baseline results.
