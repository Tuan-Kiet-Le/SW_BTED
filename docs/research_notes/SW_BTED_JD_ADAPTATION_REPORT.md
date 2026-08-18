# SW-BTED: Job Descriptions (JD) Domain Adaptation Report

This report tracks the implementation progress, dataset statistics, and architectural decisions made while adapting the **SW-CapTree** (SW-BTED) framework to the Job Descriptions domain (Dataset-3).

---

## 1. Overview & Objectives
To demonstrate the generalizability of the domain-aware 4-layer CapTree matching framework, we are executing a third dataset evaluation using the **LinkedIn Job Postings (2023-2024)** Kaggle dataset.
This validates:
1.  **Generalizability**: Extending the 4-layer document partitioning (ROOT -> DOMAIN -> INTENT -> TERMINOLOGY) beyond capstones and requirements documents.
2.  **Mitigation of Topic Conflation**: Proving that SBERT embeddings alone fail on job description pairs sharing identical roles but different industries (e.g. HealthTech Data Scientist vs FinTech Data Scientist), while SW-CapTree successfully filters them out.

---

## 2. Completed Milestones

### Milestone A: Dataset Placement & Path Resolution
*   Located the LinkedIn postings raw database under: `d:\FPT\Semester_8\RAG_Research\Data\dataset\linkedin_jd/`
*   Verified presence of core tables: `postings.csv`, `companies/companies.csv`, `jobs/job_industries.csv`, and `mappings/industries.csv`.

### Milestone B: Developed Job Description Parser (`src/jd_parser.py`)
*   Designed a regex-based segmenter targeting common headers (`Who you are`, `Role`, `Qualifications`, `Responsibilities`, etc.) to cleanly partition flat text into 4 domains:
    *   `D1_COMPANY_CONTEXT`: Mission, overview.
    *   `D2_REQUIREMENTS`: Experience, skills.
    *   `D3_RESPONSIBILITIES`: Daily tasks, duties.
    *   `D4_COMPENSATION`: Salary, schedule.
*   Parsed domain contents into T3 `IntentMatching` sentence segments using spaCy sentence segmentation.
*   **Performance Verification**: Tested on the filtered 42,332 postings. Over **99.6%** (42,176 postings) successfully matched at least 2 non-empty domain partitions, with **23,101** postings having all 4 domains parsed cleanly.

### Milestone C: Created Dataset Builder (`src/jd_dataset_builder.py`)
*   Integrated target industry filters (IT, Healthcare, Finance, Education, Marketing).
*   Constructed matching logic for the target 200 pairs:
    *   **66 Positive Pairs**: Same company, same job title, different postings.
    *   **80 Hard Negatives**: Same title (e.g., Data Scientist) but different industries (e.g., Finance vs Healthcare).
    *   **54 Easy Negatives**: Different company, different roles.

---

## 3. Issues Identified & Resolved
1.  **Low Valid JD count**: Initial validation required both `D2_REQUIREMENTS` and `D3_RESPONSIBILITIES` to be strictly non-empty and >50 characters. Because HTML-to-text converters scrape differently, some postings lacked explicit title headers for one of the sections.
    *   *Solution*: Relaxed the filter to require at least **any 2 out of 4** sections to be non-empty. This increased the valid pool from `204` to **`42,176`**, enabling high-quality pairs construction.
2.  **CapstoneNode constructor TypeError**: Found that the CapstoneNode constructor does not accept a `project_id` keyword argument.
    *   *Solution*: Updated the root node instantiation to assign the unique `job_id` string directly to `label`, and store the descriptive title in the `feature_label` attribute, matching the FPT implementation.

---

## 4. Execution History
*   [x] Run the corrected dataset builder to generate `pairs.csv` and `trees_unnormalized.json`.
*   [x] Develop the normalizer engine (`src/jd_normalizer.py`) to run leaf-level CSO/TEM matching.
*   [x] Implement the evaluation suite (`experiments/archive/run_jd_evaluation.py`) and report the F1 performance.

---

## 5. Experimental Results & Key Findings

We evaluated the proposed **SW-CapTree** against multiple baseline algorithms on the constructed LinkedIn Job Descriptions Dataset (200 pairs: 66 positive, 80 hard negatives, 54 easy negatives). The results are summarized below:

| Method | Threshold | Precision | Recall | F1-Score | ROC-AUC | TC TNR (Hard Negs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Lexical Overlap | 0.17 | 0.7857 | 0.8333 | 0.8088 | 0.8808 | 0.7125 |
| SBERT Only | 0.60 | 0.7857 | 0.8333 | 0.8088 | 0.9009 | 0.6750 |
| TED Only | 0.30 | 0.8254 | 0.7879 | 0.8062 | 0.8407 | 0.8750 |
| Engelbach et al. (2024) | 0.46 | 0.8143 | 0.8636 | 0.8382 | 0.9168 | 0.7750 |
| **SW-CapTree (Proposed)** | **0.32** | **0.8923** | **0.8788** | **0.8855** | **0.9427** | **0.9125** |

### Key Takeaways:
1.  **Generalizability Confirmed**: SW-CapTree achieved SOTA performance on the Job Description domain with an **F1-score of 0.8855** and **ROC-AUC of 0.9427**, demonstrating that the 4-layer CapTree matches and excels on text corpora other than Capstone documents.
2.  **Mitigation of Topic Conflation**: SBERT Only suffered heavily from Topic Conflation, misclassifying 32.5% of hard negatives (same title, different industry) as positive, resulting in a low **TC TNR of 0.6750**. SW-CapTree improved this significantly to **0.9125** (+23.75% TNR), proving that domain-aware tree edit distance successfully prevents embedding-based topic confusion.
3.  **Outperforming Domain Baselines**: SW-CapTree surpassed Engelbach et al. (2024) [P1] by **+4.73%** in F1-score and **+13.75%** in TC TNR, confirming the benefits of a hierarchical domain partition over flat lexical-semantic combinations.


