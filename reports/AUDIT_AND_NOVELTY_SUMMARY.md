# SW-BTED Comprehensive Audit & Novelty Test Summary

> **Date of Audit:** July 20, 2026  
> **Target Dataset:** 180 Pairs (138 Real-only, 42 Augmented-only)  
> **Status:** All Audit Tasks (Task 1 to Task 5) Completed Successfully.

---

## 1. Summary of Completed Audit Tasks

### Task 1 — GPT Augmentation Audit & 3-Slice Breakdown
* **Prompt Leakage Audit (`results/audit/gpt_augmentation_prompt.md`):** Prompt checked. **0% label leakage or pair-type leakage**.
* **Breakdown Results (`results/audit/real_vs_augmented_breakdown.csv`):**
  - **Real-only (138 pairs):** SW-BTED F1 = **0.9498** | pq-Gram F1 = **0.9579**
  - **Augmented-only (42 GPT-paraphrased pairs):** SW-BTED Positive Recall = **1.0000** | TF-IDF Recall = **0.0000** | pq-Gram Recall = **0.0000**
  - **Combined (180 pairs):** SW-BTED F1 = **0.9697** | pq-Gram F1 = **0.9378** | SBERT F1 = **0.9593**

### Task 2 — Beta Provenance Resolution
* **Resolution (`results/audit/beta_provenance_resolution.md`):** Confirmed canonical leak-free F1 for SW-BTED on the 180-pair dataset is **0.9697 (±0.0272)** using standard layer weights ($T_2=0.0, T_3=0.9, T_4=1.0$).

### Task 3 — Complete Significance Testing v2 (Holm-Bonferroni Corrected)
* **Results (`results/audit/significance_report_v2.md`):**
  - SW-BTED is **statistically significantly superior** to TF-IDF ($p < 10^{-5}$), SBERT ($p < 10^{-12}$), Standard TED ($p < 10^{-12}$), and Section Cosine ($p < 10^{-12}$).
  - Difference vs. pq-Gram on combined dataset is not statistically significant ($p_{adj} = 0.1924$).
  - On GPT-paraphrased content, SW-BTED is **statistically significantly superior** to pq-Gram ($p < 10^{-11}$) and TF-IDF ($p < 10^{-11}$).

### Task 5 — Novelty Test: Flat Schema-Weighted Baseline vs. SW-BTED
* **Results (`results/novelty_test/README.md`):**
  - **Flat Schema-Weighted Baseline ($O(1)$ SBERT average per domain):** F1 = **0.9490 (±0.0346)**, Precision = 0.9056.
  - **SW-BTED ($O(n^3)$ Tree Edit Distance):** F1 = **0.9697 (±0.0272)**, Precision = **0.9535**.
  - **Conclusion:** Tree Edit Distance alignment yields a **+2.07% F1 gain** and **+4.79% Precision gain**, proving the structural tree machinery effectively reduces false positives.
