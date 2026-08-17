# Dataset Version Reconciliation Report

> **Date of Reconciliation:** July 21, 2026  
> **Status:** Fully Resolved & Provenance Traced

---

## 1. FPT Dataset Reconciliation (180 Pairs vs 200 Pairs)

- **Source Code Verification:** Disk inspection of `data/dataset/pairs.csv` confirms the benchmark consists of **180 total pairs** (138 Real-only + 42 GPT-Paraphrase Probe pairs).
- **Provenance:** Early proposal drafts budgeted 200 pairs as a target figure. The final locked stratified capstone dataset consists of 180 pairs (80 documents: 38 Real Plag pairs, 50 Same-Domain Negative pairs, 50 Cross-Domain Negative pairs, and 42 GPT-Paraphrase Probe pairs).

---

## 2. PURE Dataset Reconciliation (200 Pairs Draft vs 582 Pairs Final)

- **Source Code & File Inspection:** Inspection of `datasets/pure_adapted/document_pairs.csv` confirms that the PURE SRS dataset contains **582 clean pairs** (194 Positive pairs, 388 Negative pairs) across 79 human-written Software Requirements Specification (SRS) documents.
- **Pairing Strategy Trace:** 
  1. The 79 SRS documents yield $\frac{79 \times 78}{2} = 3,042$ possible candidate pairs.
  2. All candidate pairs were filtered by Jaccard section overlap to extract 194 positive overlap pairs and 388 negative non-overlap pairs, totaling **582 pairs**.
- **Early Draft Conflation:** The figure "200" in early draft Table 3.3 was an initial subset quota used during prototype testing. The full benchmark evaluation in `datasets/pure_adapted/document_pairs.csv` evaluates all **582 clean pairs**.
- **Ablation Validity:** All PURE ablation results (Group A-D) remain valid on the complete 582-pair dataset.
