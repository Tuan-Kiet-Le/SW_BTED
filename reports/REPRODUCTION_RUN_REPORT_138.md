# SW-BTED 138-Pair Reproduction Run Report

Date: 2026-08-14  
Scope: four-layer manuscript reproduction  
Primary dataset: 138 real-only pairs

## Executive result

The recovered candidate pipeline **runs to completion in a staging copy**, but it does **not reproduce the manuscript's reported F1 = 0.9498**.

The staging run produced:

```text
SW-BTED F1 = 0.9467 ± 0.0869
SBERT F1   = 0.9529 ± 0.0767
pq-Gram F1 = 0.9579 ± 0.0386
```

The manuscript reports SW-BTED structural-only F1 `0.9498 ± 0.0253`. The difference is not accepted as reproducible yet because the run also shows source/configuration drift and a different fold-level variance.

## What was executed

The candidate was staged at:

`repro_candidate_138/`

The staging dataset was constructed from:

- `RAG_Research/data/dataset/pairs.csv`
- `RAG_Research/data/processed/plag_regen_sections.json`

All pairs involving one of the 42 regenerated-document keys were excluded. The resulting dataset was independently verified as:

| Class | Count |
|---|---:|
| Type A / positive | 38 |
| Type B | 50 |
| Type C | 50 |
| Total | 138 |

The candidate source was copied from the timestamp-near RAG source chain, with two staging-only execution adjustments:

1. `ProcessPoolExecutor` was changed to `ThreadPoolExecutor` because the environment rejected Windows multiprocessing pipes with `WinError 5`.
2. `baselines.py` was pointed to the locally cached `all-MiniLM-L6-v2` snapshot because external Hugging Face access was unavailable.

No source in the main workspace or `RAG_Research` was overwritten.

## Environment checks

Available:

- Python 3.13.5
- numpy 2.4.6
- pandas 3.0.3
- scikit-learn 1.8.0
- scipy 1.17.1
- apted 1.0.3
- PyYAML 6.0.3
- sentence-transformers 5.5.1
- spaCy 3.8.14
- `en_core_web_sm` and `en_core_web_trf`
- local `all-MiniLM-L6-v2` model snapshot

The first run attempted network model resolution and was blocked. The final run used the local model snapshot in offline mode.

## Final staging metrics

| Method | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| SW-BTED | 0.9467 ± 0.0869 | 0.9083 | 1.0000 | 0.9861 |
| TF-IDF | 0.9223 ± 0.1083 | 0.8694 | 1.0000 | 1.0000 |
| Full-document SBERT | 0.9529 ± 0.0767 | 0.9178 | 1.0000 | 1.0000 |
| Standard TED | 0.3425 ± 0.1922 | 0.2180 | 0.8000 | 0.1046 |
| pq-Gram | 0.9579 ± 0.0386 | 1.0000 | 0.9214 | 0.9971 |
| Section Cosine | 0.5015 ± 0.2323 | 0.5470 | 0.6857 | 0.8125 |

The complete generated outputs are under:

`repro_candidate_138/results/`

## Fold configuration

The run selected the following configuration:

```text
alpha = 0.8 on all five folds
beta  = per_layer on all five folds
thresholds = 0.22, 0.23, 0.23, 0.23, 0.22
```

This is materially different from the manuscript's structural-only presentation and confirms that the recovered evaluation harness is a hybrid/configuration-search harness, not a simple fixed structural-only reproduction.

## Important findings

### 1. The source candidate is close, but not proven canonical

The candidate chain correctly reproduces the 138-pair composition and produces a SW-BTED score close to `0.9498`. However, it does not reproduce the manuscript exactly.

### 2. The candidate harness mixes modes

The evaluation script searches alpha values and combines structural and global similarity. Its selected alpha is `0.8`, while the manuscript separately reports structural-only and hybrid modes. Therefore the exact manuscript result likely came from a different evaluation path, a different output snapshot, or a later baseline/harness correction.

### 3. Existing baseline numbers are not from one unambiguous run

The timestamp-near audit file matches SW-BTED `0.9498`, but its baseline values differ from the manuscript's Table 1. The current reproduction also differs from both in several baseline rows. This confirms source/result drift across experiment branches.

### 4. Pre-filter diagnostic remains valid

The previously recorded diagnostic reports:

```text
positive pairs below 0.25: 0%
negative pairs below 0.25: 31%
```

This supports treating the pre-filter as a computational shortcut with measurable class-dependent coverage, not as an automatic ground-truth classifier.

### 5. No final structural perturbation rerun was claimed

The current run reproduced the 138-pair evaluation harness. The separate 20-pair structural perturbation benchmark was not rerun in this step because its generator and exact source provenance were not yet uniquely identified.

## Verification status

| Requirement | Status |
|---|---|
| Timestamp-first triage | Complete |
| Environment/dependency inspection | Complete |
| 138-pair dataset reconstruction | Complete: 38/50/50 |
| Smoke test of SW-BTED cost engine | Complete |
| Full candidate 138-pair run | Complete in staging |
| Exact manuscript metric reproduction | **Failed / unresolved** |
| Leakage/pre-filter outputs | Partially available; pre-filter diagnostics identified |
| Exact canonical source manifest | **Not established** |
| Structural perturbation rerun | Pending |

## Conclusion

The code candidate is executable and scientifically close to the manuscript result, but the manuscript's exact numbers are not yet reproducible from a single recovered source/configuration package. The next required task is a read-only reconciliation of the manuscript's structural-only output, the hybrid evaluation harness, and the corrected baseline trace—especially the source of `0.9498`, `0.9867`, `0.9479`, and `0.4081`—before any source is declared canonical or the manuscript is revised.
