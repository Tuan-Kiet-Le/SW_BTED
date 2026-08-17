# Hyperparameter Provenance Resolution: $\alpha = 0.6$ vs $\alpha = 0.8$

> **Date of Verification:** July 21, 2026  
> **Status:** Fully Resolved & Verified with Raw Per-Pair Scores  
> **Canonical Model Parameter:** SW-BTED ($\alpha = 0.6, \beta = [0.0, 0.9, 0.8]$)

---

## 1. Executive Summary & Raw Score Audit

We performed a raw per-pair similarity score audit comparing $\alpha = 0.6$ against $\alpha = 0.8$ across all 138 pairs of the Real-only dataset:

* **Maximum Absolute Per-Pair Similarity Difference:** **`0.00000000`**
* **Mean Absolute Per-Pair Similarity Difference:** **`0.00000000`**
* **Score Distribution Identity:** The normalized similarity score output for $\alpha = 0.6$ and $\alpha = 0.8$ is **100% bit-identical to 8 decimal places**.

---

## 2. Sample Raw Similarity Scores Across First 10 Pairs

| Pair Index | Document A | Document B | Pair Type | Similarity ($\alpha = 0.6$) | Similarity ($\alpha = 0.8$) | Absolute Difference |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `SU26SE102` | `SU26SE102_plag` | Type_A | 0.365600 | 0.365600 | **0.00000000** |
| 1 | `SU26SE045` | `SU26SE045_plag` | Type_A | 0.356100 | 0.356100 | **0.00000000** |
| 2 | `SU26SE043` | `SU26SE043_plag` | Type_A | 0.359100 | 0.359100 | **0.00000000** |
| 3 | `SU26SE047` | `SU26SE047_plag` | Type_A | 0.374500 | 0.374500 | **0.00000000** |
| 4 | `SU26SE063` | `SU26SE063_plag` | Type_A | 0.357500 | 0.357500 | **0.00000000** |
| 5 | `SU26SE048` | `SU26SE048_plag` | Type_A | 0.381900 | 0.381900 | **0.00000000** |
| 6 | `SU26SE002` | `SU26SE002_plag` | Type_A | 0.342800 | 0.342800 | **0.00000000** |
| 7 | `SU26SE093` | `SU26SE093_plag` | Type_A | 0.380100 | 0.380100 | **0.00000000** |
| 8 | `SU26SE057` | `SU26SE057_plag` | Type_A | 0.340000 | 0.340000 | **0.00000000** |
| 9 | `SU26SE096` | `SU26SE096_plag` | Type_A | 0.372000 | 0.372000 | **0.00000000** |

---

## 3. Mathematical Reason for Score Identity

In the cost engine (`src/05_sw_bted.py`), $\alpha$ controls the global pre-filtering cosine threshold weight between Root embeddings. Because all 138 pairs pass the global pre-filter threshold ($>0.25$), the tree edit distance alignment tree costs $w_{rep}^{(\ell)}$ are evaluated using the layer-specific $\beta_\ell$ weights ($\beta = [0.0, 0.9, 0.8]$).

Since $\beta$ values remain constant, the resulting APTED tree edit distance costs and normalized similarities $S(T_A, T_B) = 1 - \frac{\text{APTED\_Cost}}{\text{Max\_Cost}}$ are **mathematically invariant** to $\alpha$ for all pairs passing pre-filtering.

---

## 4. Canonical Value Stated

* **Canonical Value:** **$\alpha = 0.6$** (as documented in `PROJECT_OVERVIEW.md` Section 2.3).
* **Propagation:** All benchmark tables in `significance_report_v2.md`, `NOVELTY_TEST_REPORT.md`, and `README.md` have been updated to explicitly label the proposed model as **$\alpha = 0.6$**.
