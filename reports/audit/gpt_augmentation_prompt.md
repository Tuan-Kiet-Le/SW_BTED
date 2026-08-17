# GPT-4o-mini Data Augmentation Audit Report

> **Date of Audit:** July 20, 2026  
> **Auditor:** Antigravity AI Assistant  
> **Target Dataset:** 42 missing D3/D4 plagiarism documents (53 pairs total affected)

---

## 1. Exact Generation Prompt Analysis

The exact script used for D3/D4 section augmentation is located at `scratch/regen_plag_sections.py`.

### System Prompt:
```text
You are a paraphrasing assistant for academic research.
Your task is to paraphrase given capstone project requirements text.
Rules:
- Keep all technical meaning, requirements, and constraints intact
- Rephrase sentences with different vocabulary and sentence structure
- Do NOT add or remove requirements
- Do NOT change technology names (React, FastAPI, PostgreSQL, etc.)
- Output ONLY the paraphrased text, no explanations or headers
- Preserve bullet-point structure if present in the input
- Keep approximately the same length as the input
```

### User Prompt:
```text
Paraphrase this {domain_hint} section from a software capstone project:

{text}
```
*(where `{text}` is the original document's raw text for D3/D4 and `{domain_hint}` is either "technical/non-functional requirements or applied theory/technologies" for D3, or "project task breakdown/execution plan" for D4).*

---

## 2. Leakage Determination

| Audit Item | Finding | Leakage Status |
| :--- | :--- | :---: |
| **Referenced Pair Type (A/B/C)?** | No. The prompt contains no mention of pair types or plagiarism classification. | ✅ Clean |
| **Referenced Paired Doc Content?** | No. Only the source document's D3/D4 text was passed to generate its plag counterpart. | ✅ Clean |
| **Referenced Ground Truth Label?** | No. | ✅ Clean |
| **Generation Frequency Across Folds** | Generation was performed **ONCE** (static pre-generation saved in `data/processed/plag_regen_sections.json`). | ⚠️ Fixed artifact (re-used across CV folds, standard for dataset augmentation) |

**Conclusion on Leakage:** The prompt is **100% free of label leakage or pair-type leakage**. The generated D3/D4 sections represent valid structural paraphrases of the source documents.
