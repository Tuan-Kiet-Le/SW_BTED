# Clean baseline suite — canonical 138 pairs

Protocol: 5-fold stratified CV, seed 42, threshold grid 0.005, train-fold-only selection.

| Method | F1 | Std | Precision | Recall | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.9867 | 0.0267 | 0.9750 | 1.0000 | 38 | 1 | 99 | 0 |
| Standard TED | 0.4364 | 0.0162 | 0.2792 | 1.0000 | 38 | 98 | 2 | 0 |
| pq-Gram | 0.9479 | 0.0478 | 0.9528 | 0.9464 | 36 | 2 | 98 | 2 |
| Section Cosine | 0.6837 | 0.0894 | 0.5263 | 1.0000 | 38 | 38 | 62 | 0 |
| Genuine Flat Domain SBERT | 0.4314 | 0.0160 | 0.2751 | 1.0000 | 38 | 100 | 0 | 0 |