# Schema-matched embedding baseline — canonical 138 pairs

This is a flat baseline: it embeds D1–D4 separately, averages their cosine similarities, and uses no tree alignment or edit operations.

Protocol: MiniLM snapshot `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, model max length `256`, SentenceTransformer pooling, 5-fold stratified CV (seed 42), train-fold-only threshold grid 0.005.

| F1 mean | F1 std | Precision | Recall | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4314 | 0.0160 | 0.2751 | 1.0000 | 38 | 100 | 0 | 0 |

Interpretation: this isolates the contribution of schema decomposition without SW-BTED's structural edit distance. It is a comparison baseline, not evidence that the embedding model is universally weak or strong.