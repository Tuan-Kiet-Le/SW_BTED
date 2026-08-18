# Task B Report: Genuine Flat Domain Baseline (Bug Reports Dataset)

> **Date:** July 22, 2026  
> **Dataset:** GitBugs Benchmark Sample ($n=300$ pairs: 100 Duplicate Positives, 100 Hard Negatives, 100 Easy Negatives)  
> **Formula:** $\text{sim}_{flat}(A, B) = \frac{1}{4} \sum_{d \in \{D_1..D_4\}} \text{cosine}(\text{SBERT}(D_d(A)), \text{SBERT}(D_d(B)))$  
> **Code Verification:** ZERO `sim_struct` or tree-alignment term present.

---

## 1. Performance Metrics (5-Fold Stratified Cross-Validation)

| Metric | Genuine Flat Domain SBERT Baseline |
| :--- | :---: |
| **5-Fold CV F1-Score** | **0.7673 ± 0.0201** |
| **Precision** | **0.8353** |
| **Recall** | **0.7100** |
| **True Positives (TP)** | 71 |
| **False Positives (FP)** | 14 |
| **True Negatives (TN)** | 186 |
| **False Negatives (FN)** | 29 |

---

## 2. Hard Negatives vs Easy Negatives Analysis

- **Hard Negatives (Same Project):** Flat domain embeddings struggle with domain-specific vocabulary overlap in bug reports from the same project, leading to false-positive classification errors.
