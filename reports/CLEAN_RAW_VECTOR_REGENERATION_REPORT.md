# Clean regeneration report: 138-pair embedding vectors

Date: 2026-08-14

## Execution

A new script, `experiments/regenerate_clean_embedding_vectors_138.py`, regenerated the three local embedding baselines directly from:

- `repro_candidate_138/data/dataset/pairs.csv`
- `repro_candidate_138/data/dataset/trees_section.json`
- `repro_candidate_138/data/dataset/full_texts.json`

The script used immutable local Hugging Face snapshots and wrote new files without overwriting the historical audit artifact:

- `reports/audit/clean_raw_embedding_vectors_138.csv`
- `reports/audit/clean_raw_embedding_vectors_138.json`

Text construction was:

```text
tree label + " " + full_texts section values in JSON order
```

Cosine was computed directly from the generated embeddings. The canonical pair-file hash is recorded in the JSON artifact.

## Direct result for index 84

The regenerated row is:

```json
{
  "doc_a": "SU26SE087",
  "doc_b": "SP26SE001",
  "label": "0",
  "type": "Type_B",
  "index": 84,
  "SBERT_MiniLM": 0.5151844620704651,
  "BGE_Small_v1.5": 0.7650423049926758,
  "MPNet_Base_v2": 0.6619883179664612
}
```

This agrees with the independent MiniLM audit (`0.5151844621`) and disagrees with the historical raw-vector artifact's SBERT value (`0.6555`). The BGE and MPNet values are effectively the same at this row after normal floating-point precision.

## Comparison with the old raw artifact

The old file is not a reliable source for regenerated SBERT vectors:

| Model | Maximum absolute difference | Mean absolute difference | Rows within 1e-4 |
|---|---:|---:|---:|
| SBERT MiniLM | 0.1769540 | 0.0215812 | 100/138 |
| BGE small | 0.2171477 | 0.0109333 | 100/138 |
| MPNet base | 0.5668267 | 0.0143728 | 99/138 |

The old artifact therefore contains multiple mismatches, not only the index-84 anomaly. Possible causes include a different historical model revision, different text/tree inputs for some rows, ordering/copy errors, or manual artifact construction. The exact cause is not inferred from the vector file alone.

## Consequence for the `0.6555` claim

The earlier statement that “the original raw artifact directly proves index 84 has SBERT cosine `0.6555` for `SU26SE087–SP26SE001`” is technically true as a statement about the contents of that old JSON, but it is not evidence that a clean current SBERT run produces `0.6555`.

The clean reproducible value for the canonical 138-pair protocol is `0.5151844621`. Therefore the old `0.6555` should be removed from primary manuscript evidence unless its producing script, model revision, and exact input artifacts can be recovered.

## Case A status

The separate Case A concern remains resolved at the dataset-membership level:

- `SU26SE102–SU26SE102_plag` is row 0 of the 180-pair file.
- Neither document is in `plag_regen_sections.json`.
- The pair is therefore retained in the 138-pair filtered dataset.

The manuscript should still state the Case A pair ID explicitly and link its alignment output.

## Required next action

Recompute baseline metrics from the clean vectors using the frozen historical threshold protocol, then update the manuscript and comparison tables from those clean outputs. Do not use `reports/audit/raw_prediction_vectors_138.json` as primary numerical provenance.
