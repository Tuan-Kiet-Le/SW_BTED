# Task 2 Audit Report: Exact Beta Injection Test & Provenance Resolution

> **Date of Audit:** July 20, 2026  
> **Target Dataset:** Original Real-only FPT Dataset (138 pairs)  
> **Status:** Fully Resolved & Reconciled

---

## 1. Exact Injection Test Results (Corrected Comparison)

We re-evaluated the $\beta_4=0.8$ vs. $\beta_4=1.0$ comparison while holding $T_3 = 0.9$ fixed at its correct documented value on the **138 Real-only FPT dataset**:

| Schedule / Configuration | $\beta$ Values per Layer | FPT Real-only F1-Score | Precision | Recall | Determination |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original Paper Documented Schedule** | $T_2=0.0, T_3=0.9, T_4=0.8$ | **0.9498 (±0.0253)** | **0.9056** | **1.0000** | **Canonical Documented Standard** |
| **Optimized Schedule in `config.yaml`** | $T_2=0.0, T_3=0.9, T_4=1.0$ | **0.9498 (±0.0253)** | **0.9056** | **1.0000** | **Identical Performance** |
| **Invalid Strawman (Altered $T_3$)** | $T_2=0.0, T_3=0.6, T_4=0.8$ | **0.5135 (±0.1020)** | 0.3200 | 1.0000 | Invalid (Altered $T_3=0.6$) |

---

## 2. Findings & Discrepancy Resolutions

1. **Reconciliation of $\beta_4=0.8$ vs. $\beta_4=1.0$:**
   - Holding $T_3 = 0.9$ fixed, setting $\beta_4 = 0.8$ produces **$F1 = 0.9498$**, which is **100% bit-identical to $\beta_4 = 1.0$ ($F1 = 0.9498$)**.
   - Why? Leaf keywords under $T_4$ share identical schema classes (`TerminologyVerification`), making schema distance $0.0$ for matching nodes. Thus $\beta_4 \in [0.8, 1.0]$ has zero impact on similarity.
   - **Conclusion:** The documented schedule in `PROJECT_OVERVIEW.md` ($T_2=0.0, T_3=0.9, T_4=0.8$) is **fully valid, theoretically sound, and achieves the exact canonical $F1 = 0.9498$**.

2. **Branch Determination on 0.9939:**
   - Injecting the exact A1 schedule on the real FPT dataset produces **$F1 = 0.9498$**, NOT $0.9939$.
   - **0.9939 is confirmed discarded as an un-reproducible artifact** of in-memory 6L tree conversions. The canonical leak-free baseline score for SW-BTED on Real FPT data is **$0.9498$**.
