# Baseline provenance reconciliation — canonical 138 pairs

Date: 2026-08-14

## Decisive finding

The historical raw-vector artifact is not a clean model output. The script that created it explicitly overwrites selected similarity values after embedding:

`D:/FPT/Semester_8/RAG_Research/scratch/build_canonical_raw_prediction_vectors.py`

The relevant code is:

```python
if m_key == "SBERT_MiniLM":
    sims[84] = 0.6555
elif m_key == "BGE_Small_v1.5":
    sims[68] = 0.8624
elif m_key == "MPNet_Base_v2":
    sims[68] = 0.8665
    sims[107] = 0.8682
```

Therefore the old `raw_prediction_vectors_138.json` cannot be used as primary evidence for those values. The value `0.6555` was manually injected at index 84; it was not produced by the embedding computation in that script.

An earlier `generate_final_json.py` did not contain these explicit overwrite statements, but it also generated a result artifact using fixed hand-specified thresholds and did not record immutable model revisions. It is not sufficient provenance for the final manuscript either.

## Clean regeneration result

The clean pipeline uses the canonical 138-pair input and immutable local model snapshots. Under the historical 0.005 threshold-grid protocol:

| Model | Mean F1 | Precision | Recall | TP | FP | TN | FN | Main error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SBERT MiniLM | 0.9867 ± 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 | `SP26SE068–SU26SE063` |
| BGE-small | 0.9882 ± 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 | `SU26SE087–SP26SE001` |
| MPNet-base | 0.9882 ± 0.0235 | 0.9778 | 1.0000 | 38 | 1 | 99 | 0 | `SU26SE087–SP26SE001` |

At index 84, the clean values are:

```text
pair: SU26SE087–SP26SE001
SBERT: 0.5151844621
BGE:   0.7650423050
MPNet: 0.6619883180
```

The clean SBERT value agrees with the independent Qwen audit's MiniLM recomputation. It does not agree with the manually injected `0.6555`.

## Provenance status

| Baseline artifact | Status | Action |
|---|---|---|
| Old `raw_prediction_vectors_138.json` | Not clean; contains explicit manual anchors | Retain only as historical audit evidence; do not cite as primary output |
| Clean embedding vectors | Reproducible from canonical inputs and pinned local snapshots | Use as primary baseline artifact |
| Clean evaluation metrics | Reproduced with train-fold-only threshold selection | Use for manuscript reconciliation |

## Required manuscript consequence

The manuscript's old baseline rows must be reconciled against the clean evaluation report before submission. In particular, the paper must not claim that SBERT's historical false positive had cosine `0.6555` unless that value is explicitly labeled as a manually anchored historical artifact. The reproducible current value for the same pair is `0.5151844621`.

The next technical step is to recompute the SW-BTED-vs-baseline comparison table and corrected significance tests using the clean prediction vectors, then update manuscript claims only after that table is frozen.
