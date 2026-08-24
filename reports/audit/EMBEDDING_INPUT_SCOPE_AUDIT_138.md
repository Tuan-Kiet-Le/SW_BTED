# Embedding input-scope audit — canonical 138 pairs

Dataset: `138` pairs, `178` participating documents.
Pair hash: `d20a34fc8997baa272a6d8f89bf5553922842c6032b11d08b4f831dcb4bc2a56`.

The audit measures untruncated tokenizer lengths for the exact full-document strings used by the clean embedding scripts. Historical `encode()` calls did not pass an explicit max length; the model tokenizer limit therefore governs truncation.

| Model | Tokenizer max | Effective encode max | Median | P95 | Max observed | Docs over effective limit | Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| SBERT_MiniLM | 512 | 256 | 606 | 2581 | 4235 | 159 | 0.893 |
| BGE_Small_v1.5 | 512 | 512 | 606 | 2581 | 4235 | 103 | 0.579 |
| MPNet_Base_v2 | 512 | 384 | 606 | 2581 | 4235 | 124 | 0.697 |

Interpretation: truncation is a protocol limitation when the fraction above the model limit is non-zero. It does not invalidate the historical result, but the manuscript should report the limit and affected-document count, and schema-matched experiments should use the same explicit tokenizer/truncation policy.