# Second-pass check: raw index 84 and Case A data membership

Date: 2026-08-14

## 1. Direct evidence for zero-based index 84

The original file is:

`reports/audit/raw_prediction_vectors_138.json`

The relevant object is directly under `disagreement_pair_details.sbert_vs_bge`, not reconstructed from the summary report:

```json
{
  "index": 84,
  "doc_a": "SU26SE087",
  "doc_b": "SP26SE001",
  "type": "Type_B",
  "true_label": 0,
  "sbert_sim": 0.6555,
  "sbert_pred": 1,
  "bge_sim": 0.765,
  "bge_pred": 0
}
```

The same original JSON contains, at array position 84 in `similarity_scores`:

```text
SBERT_MiniLM[84] = 0.6555
BGE_Small_v1.5[84] = 0.765
MPNet_Base_v2[84] = 0.662
```

The source CSV independently confirms the same object. Since the header occupies physical line 1, zero-based pair index 84 is physical line 86:

```csv
"SU26SE087","SP26SE001","0","Type_B"
```

The Qwen3 score CSV has the corresponding row:

```csv
SU26SE087,SP26SE001,Type_B,0,0.5849661827087402
```

Thus the original 138-pair artifact directly identifies index 84 as `SU26SE087`–`SP26SE001`, and its stored `0.6555` is attached to that pair. A subsequent clean regeneration, however, produces `0.5151844621` for SBERT at the same row and finds multiple old-vector mismatches. The pair identity is confirmed, but the old `0.6555` value is not reproducible under the current frozen protocol. The separate `SP26SE082`–`SP26SE082_plag` pair is at zero-based index 26 of the 180-pair source, not index 84 of the 138-pair source.

## 2. The `_plag` suffix is not the exclusion rule

The actual exclusion operation in the historical evaluation is:

```python
real_pairs = pairs[
    ~(pairs['doc_a'].isin(regen_keys) | pairs['doc_b'].isin(regen_keys))
]
```

where `regen_keys` comes from:

`RAG_Research/Data/processed/plag_regen_sections.json`

The suffix `_plag` alone is not used as the filter. Direct counts from the 180-pair source are:

| Check | Count |
|---|---:|
| Pairs containing `_plag` | 80 |
| Such pairs excluded by explicit `regen_keys` membership | 42 |
| Such pairs retained in the 138-pair evaluation | 38 |

The relevant direct rows are:

```text
zero-based row 0:
SU26SE102, SU26SE102_plag, label=1, Type_A
regen membership: SU26SE102=False, SU26SE102_plag=False

zero-based row 26:
SP26SE082, SP26SE082_plag, label=1, Type_A
regen membership: SP26SE082=False, SP26SE082_plag=True
```

Therefore:

- `SP26SE082`–`SP26SE082_plag` is excluded from the 138-pair real-only evaluation.
- `SU26SE102`–`SU26SE102_plag` is not excluded by the current provenance rule and is retained in the 138-pair evaluation.

## 3. Case A implication

The manuscript's Section 5.5 describes Case A as a true plagiarism pair but does not print the document IDs in the manuscript text. If the intended Case A pair is indeed `SU26SE102`–`SU26SE102_plag`, the current dataset evidence says it is a retained pair in the 138-pair evaluation, not one of the 42 explicitly regenerated/excluded pairs.

The concern that every `_plag` pair is automatically a GPT-augmented leakage case is therefore false for this repository. The project contains two distinct categories:

1. Original/plagiarism pairs whose `_plag` documents are not present in `plag_regen_sections.json` and remain in the 138-pair set.
2. The 42 pairs whose `_plag` documents are present in `plag_regen_sections.json` and are removed from the primary evaluation.

However, there is still a documentation weakness: Section 5.5 should name the exact pair IDs and cite the case-study output file. Otherwise readers cannot independently verify that Case A uses a retained pair.

## Final verdict

The new challenge is valuable because it required checking the original artifacts. The direct artifacts support the current index-84 identity, and they reject the proposed blanket `_plag` exclusion rule. The remaining action is documentation/provenance hardening: explicitly identify Case A's pair IDs and distinguish retained historical plagiarism examples from the 42 regenerated documents.
