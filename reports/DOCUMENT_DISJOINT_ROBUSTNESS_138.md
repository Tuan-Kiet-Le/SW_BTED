# Document-Disjoint Robustness Audit — Canonical 138 Pairs

Protocol: `GroupKFold(5)` by connected components of the pair-document graph;
178 documents and 43 connected components. Thresholds use a `0.005` grid
selected on training groups only. Pair-file SHA-256 is the frozen canonical
hash in `CANONICAL_SCIENTIFIC_MANIFEST_138.md`.

| Method | Mean F1 | Std | Pooled F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| SW-BTED | 0.9160 | 0.0928 | 0.9157 | 38 | 7 | 93 | 0 |
| TF-IDF | 0.9846 | 0.0308 | 0.9870 | 38 | 1 | 99 | 0 |
| Standard TED | 0.5534 | 0.2002 | 0.4318 | 38 | 100 | 0 | 0 |
| pq-Gram | 0.9448 | 0.0530 | 0.9474 | 36 | 2 | 98 | 2 |
| Section Cosine | 0.7130 | 0.2168 | 0.6387 | 38 | 43 | 57 | 0 |
| Genuine Flat Domain SBERT | 0.5534 | 0.2002 | 0.4318 | 38 | 100 | 0 | 0 |

This is a supplemental robustness audit, not a replacement for the primary
stratified pair-level result. Its main purpose is to test whether document
identity overlap across pairs is driving the headline result.
