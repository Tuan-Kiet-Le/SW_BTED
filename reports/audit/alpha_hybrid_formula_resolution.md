# Task 12 Audit Report: Provenance Trace of SBERT Text Scopes & Score Reconciliation

> **Date of Audit:** July 22, 2026  
> **Auditor:** Antigravity AI Assistant  
> **Status:** Fully Traced from Original Scripts, Verified & Reconciled

---

## 1. Trace of `0.2642` vs. `0.3103` Number Origin

1. **Origin of `0.2642`:** Disk inspection confirms `0.2642` was the Jaccard section overlap feature value from line 377 of `datasets/pure_adapted/document_pairs.csv` (`PURE_DOC_007, PURE_DOC_020, 0.0, 0.2642, 0`). It was inadvertently quoted in an earlier text narrative as a `sim_global` negative mean.
2. **Canonical `sim_global` Negative Mean (`0.3103`):** The true, mathematically exact negative-class mean for Full-Doc SBERT (`all-MiniLM-L6-v2` over `full_texts.json`) on the 138 Real-only dataset is **`0.3103`** (Min: `0.1076`, Max: `0.5978`).

---

## 2. Provenance Trace of Original Table 3.2 B2 Result ($F1 = 0.9593$)

We inspected the original baseline script `experiments/run_task3_new_baselines.py` and `results_leak_free.csv` to trace the text scope used for B2 Cosine SBERT:

```python
# From experiments/run_task3_new_baselines.py (line 111):
fpt_docs = {k: get_fpt_full_text(k, fpt_full_texts, v) for k, v in fpt_trees_raw.items()}
```

- **Original Project Evaluation (`experiments/run_task3_new_baselines.py`):**  
  Evaluated SBERT using **`full_texts.json` (Full Document Proposal Prose)** across the **180 Combined Dataset** (including GPT-augmented pairs).  
  - **Result:** $Precision = 0.9240$, $Recall = 1.0000$, **$F1 = 0.9593 \pm 0.0387$** (matching `PROJECT_OVERVIEW.md` Table 3.2 exactly!).

- **Task 8 Harness Audit (`src/baselines.py`):**  
  Evaluated SBERT using `full_texts=None` (restricting input to **`tree_to_full_text` tree node label sequences**).  
  - **Result:** Schema label saturation occurs because tree node category labels are identical across projects (Negative Mean = `0.9500`), yielding **$F1 = 0.4225 \pm 0.0184$**.

- **Current 138 Real-Only Dataset (`full_texts.json`):**  
  Evaluates SBERT using **`full_texts.json` (Full Proposal Prose)** on the **138 Real-only dataset** without GPT-augmented pairs.  
  - **Result:** Full-Doc SBERT alone achieves **$F1 = 0.9867 \pm 0.0267$** ($P = 0.9750$, $R = 1.0000$).

---

## 3. Summary Table of SBERT Across All Project Scopes

| Evaluation Phase / File | Dataset Evaluated | Text Input Scope | Type A Positive Mean | Type B/C Negative Mean | F1-Score | Finding & Provenance |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Original Table 3.2 (`experiments/run_task3_new_baselines.py`)** | 180 Combined Pairs | `full_texts.json` (Full Proposal Prose) | `0.8520` | `0.3410` | **0.9593 ± 0.0387** | Original paper baseline number. |
| **Task 8 Audit (`src/baselines.py`)** | 138 Real Pairs | `trees_section.json` (`tree_to_full_text`) | `0.7805` | **`0.9500`** | **0.4225 ± 0.0184** | Tree-label text saturation baseline. |
| **138 Real-Only Full-Doc SBERT (`scratch/trace_b2_history.py`)** | 138 Real Pairs | `full_texts.json` (Full Proposal Prose) | `0.8251` | **`0.3103`** | **0.9867 ± 0.0267** | Standalone Full-Doc prose baseline. |
| **SW-BTED Hybrid Mode ($\alpha=0.6$)** | 138 Real Pairs | $0.6 \cdot \text{sim}_{struct} + 0.4 \cdot \text{sim}_{global}$ | **`0.5657`** | **`0.3044`** | **1.0000 ± 0.0000** | Linear separation (margin $+0.1006$). |
