# Task A Report: Front-Loaded Baseline Validity Check (Bug Reports Dataset)

> **Date:** July 22, 2026  
> **Dataset:** GitBugs Benchmark Sample ($n=300$ pairs: 100 Duplicate Positives, 100 Hard Negatives Same-Project, 100 Easy Negatives Cross-Project)  
> **Taxonomy:** Bettenburg et al. ($D_1 \dots D_4$)

---

## 1. Raw Similarity Distributions (Simplest Baselines)

### A. Full-Text SBERT Cosine Similarity (`sim_sbert_fulltext`)
| Pair Type | Sample Count | Mean Similarity | Min Similarity | Max Similarity |
| :--- | :---: | :---: | :---: | :---: |
| **Duplicate Positives (Ground Truth)** | 100 | **0.6818** | 0.1090 | 1.0000 |
| **Hard Negatives (Same Project)** | 100 | **0.2255** | -0.1189 | 0.7518 |
| **Easy Negatives (Cross Project)** | 100 | **0.1569** | -0.0794 | 0.4869 |

---

### B. Full-Text TF-IDF Cosine Similarity (`sim_tfidf_fulltext`)
| Pair Type | Sample Count | Mean Similarity | Min Similarity | Max Similarity |
| :--- | :---: | :---: | :---: | :---: |
| **Duplicate Positives (Ground Truth)** | 100 | **0.4590** | 0.0000 | 1.0000 |
| **Hard Negatives (Same Project)** | 100 | **0.0331** | 0.0000 | 0.3242 |
| **Easy Negatives (Cross Project)** | 100 | **0.0118** | 0.0000 | 0.0961 |

---

## 2. Key Diagnostic Findings & Baseline Observations

1. **Hard Negative Domain Overlap:**  
   Hard Negatives (Same Project) have significantly higher SBERT similarity (Mean = `0.2255`) than Easy Negatives (Mean = `0.1569`), confirming that Same-Project bug reports suffer from domain terminology overlap.
2. **Separation Margin:**  
   Duplicate Positives (Mean = `0.6818`) overlap with Hard Negatives (Max = `0.7518`), demonstrating that flat embeddings alone encounter false-positive risks on Hard Negatives.

---

## 3. Next Steps (Task B)

With the simplest baseline distributions front-loaded and documented, we proceed to **Task B: Implementing the Genuine Flat Domain Baseline ($D_1 \dots D_4$) with ZERO `sim_struct` term**.
