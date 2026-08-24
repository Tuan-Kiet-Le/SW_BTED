# Frozen Scientific Manifest — SW-BTED v2

Primary protocol freeze date: 2026-08-17
Manifest last updated: 2026-08-24
Post-freeze audits do not alter the frozen primary dataset, split, or canonical configuration.
Primary scope: 138 real-only pairs  
Pair-file SHA-256: `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`
Canonical config: `config.yaml`
Canonical config SHA-256: `5E13FB3F8A7000A5AD7DEBEA81F272880A02E80F49890D445290B59DC68071B3`

## Frozen decisions

- Representation: four layers, Root → Domain → Intent → Terminology.
- Primary split: 5-fold `StratifiedKFold`, shuffle=True, seed=42.
- Threshold selection: training fold only, grid step `0.005`.
- Primary SW-BTED configuration: `beta_T2=0.0`, `beta_T3=0.9`, `beta_T4=0.8`;
  structural-only result reported separately from hybrid mode.
- Clean lexical protocol: full-document fields from canonical `full_texts.json`.
- Historical tree-label lexical values are legacy/input-scope ablations and are
  not mixed with the clean baseline table.
- Qwen3: version 10 raw downloaded pair scores, name-matched by `(doc_a, doc_b,
  label, type)`, threshold re-evaluated on the `0.005` grid; tokenizer audit
  records the configured 2048-token truncation protocol.
- Perturbation benchmark: 20 constructed D2↔D3 swaps, secondary diagnostic.
- Document-disjoint robustness: supplemental GroupKFold by pair-graph
  connected component; not a replacement for primary pair-level results.

## Post-freeze supplementary audits

- Hybrid operating-point audit: `alpha=0.6`, fixed historical configuration;
  pair-level F1 `1.0000`, document-disjoint pooled F1 `0.9048`; see
  `reports/HYBRID_PROVENANCE_AUDIT_138.md`.
- Observable structural perturbation: 20 constructed pairs with a separate
  threshold-sensitivity analysis; see
  `reports/OBSERVABLE_PERTURBATION_THRESHOLD_SENSITIVITY_20.md`.
- Qwen3 version 10 tokenizer audit: 2048-token truncation protocol; see
  `reports/QWEN3_VERSION10_AUDIT_REPORT_2026-08-24.md`.

## Primary result map

For the canonical lexical baselines, TF-IDF and Section Cosine vectorizers are fitted independently within each cross-validation fold using training-fold documents only; test-fold documents are transformed with the corresponding training-fitted vectorizers. Thresholds are selected using training-fold scores on the `0.005` grid. The machine-readable fold-local artifact is `reports/audit/fold_local_lexical_suite_138.json`.

| Evidence | Canonical result | Artifact |
|---|---:|---|
| SW-BTED structural-only | F1 `0.9498 ± 0.0253` | `reports/FINAL_CANONICAL_RESULTS_138.md` |
| Fold-local TF-IDF | F1 `0.9867 ± 0.0267` | `reports/FOLD_LOCAL_LEXICAL_SUITE_138.md` and `reports/audit/fold_local_lexical_suite_138.json` |
| Fold-local Section Cosine | F1 `0.6837 ± 0.0894` | `reports/FOLD_LOCAL_LEXICAL_SUITE_138.md` and `reports/audit/fold_local_lexical_suite_138.json` |
| Standard TED | F1 `0.4364 ± 0.0162` | `reports/CLEAN_BASELINE_SUITE_138.md` |
| Structural perturbation | 100% vs 0% | `reports/STRUCTURAL_PERTURBATION_REPRODUCTION_20.md` |
| Qwen3, 0.005 grid | pooled F1 `0.9870` | `reports/QWEN3_VERSION10_AUDIT_REPORT_2026-08-24.md` |
| Document-disjoint SW-BTED | pooled F1 `0.9157` | `reports/DOCUMENT_DISJOINT_ROBUSTNESS_138.md` |
| Beta ablation | documented schedule F1 `0.9498 ± 0.0253` | `reports/ABLATION_138_CLEAN_005_REPORT.md` |

## Non-canonical historical values

Legacy tree-label TF-IDF `0.4364` and legacy tree-label Section Cosine `0.4081`
remain useful for provenance analysis. They must be labeled as a different
input scope, not presented as the clean full-document baseline results.
