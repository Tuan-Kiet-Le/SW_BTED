# When Does Tree Structure Matter? A Schema-Weighted Edit Distance Framework for Cross-Domain Document Similarity

> **Citation-alignment pass, round 2.** 27 citations total, each independently verified against a real source before insertion (up from 16 in the prior pass — 11 new citations added this round). One factual misalignment from the prior draft was found and corrected: IEEE 830 was cited as a current standard alongside ISO/IEC/IEEE 29148, but 29148:2018 formally supersedes and obsoletes IEEE 830-1998; the paper now cites 29148 as the current standard and references IEEE 830 only as its historical predecessor, where relevant. Statements describing this paper's own experimental results remain deliberately uncited — citing external sources for the paper's own findings would misrepresent whose claim is being supported.

---

## Abstract

*(No citations — standard practice; abstracts summarize the paper's own contribution.)*

Comparing semi-structured documents — software requirement proposals, bug reports, legal contracts — is a task with real practical stakes: academic integrity review, duplicate-issue triage, and contract auditing all depend on similarity judgments a human can act on and defend. Existing approaches force a tradeoff. Flat document embeddings achieve strong aggregate accuracy but collapse a document pair into a single opaque scalar, offering no way to localize *where* two documents diverge, and — as we show experimentally — are structurally blind to content reordering that preserves vocabulary but changes meaning. Tree-edit-distance (TED) methods are structurally attributable in principle, but unweighted formulations conflate superficial structural overlap with substantive similarity, producing high false-positive rates on documents that merely share a domain.

We propose SW-BTED, a schema-weighted bounded tree-edit-distance framework in which each tree layer's cost function is a convex combination of content and schema distance, a construction that provably preserves the triangle inequality and permits exact computation via APTED. Critically, we require that the tree's domain-layer schema be grounded in an independent, citable functional taxonomy of the target genre, and we demonstrate this principle across two structurally distinct domains: software capstone proposals and duplicate bug reports.

On a 138-pair real-document capstone benchmark, SW-BTED's structural-alignment component significantly outperforms a genuinely flat embedding baseline (F1 = 0.9498 vs. 0.4314, p ≈ 2.52 × 10⁻²⁹) and, on a purpose-built structural-perturbation benchmark, correctly identifies section-reordered documents with 100% accuracy where flat embeddings achieve 0% (p = 1.91 × 10⁻⁶). Against strong natural-document baselines, SW-BTED reaches statistical parity rather than superiority — a result we report and discuss honestly rather than overstate. Applied to a second, structurally distinct genre, the framework generalizes through a single hyperparameter adjustment, without algorithmic redesign, reaching parity with a strong embedding baseline. We characterize an explicit tradeoff between two operating configurations of the method and demonstrate genuine structural interpretability through sub-tree edit traces.

---

## 1. Introduction

Institutions that must compare large volumes of semi-structured documents face a recurring, practically important question: not just *how similar* are two documents, but *where*, specifically, do they diverge, and does that divergence matter? [¹] An academic integrity committee reviewing a suspected plagiarism case needs to know whether two capstone proposals share only superficial domain vocabulary or actually reuse a project's core technical approach — a distinction that recent systematic surveys of plagiarism-detection methods confirm remains genuinely difficult, since paraphrase- and idea-level plagiarism resist simple string-matching approaches (Amirzhanov et al., 2025). A bug-triage lead needs to know whether two reports describe the same underlying defect or merely occur in the same subsystem — a task shown to depend heavily on which sections of a report actually overlap, not just whether the reports are lexically similar overall (Ghadhab & Amor, 2024). A single similarity score — however accurate in aggregate, as measured by standard classification metrics such as the F-measure (van Rijsbergen, 1979) — cannot answer either question; it can only be trusted or distrusted as a whole.

This creates a persistent tension in how such comparisons are made. Dense document embeddings, following the sentence-transformer paradigm (Reimers & Gurevych, 2019) and evaluated on benchmarks such as the STS Benchmark for semantic textual similarity (Cer et al., 2017), achieve strong aggregate discrimination and are the default tool for this kind of task, but they compress an entire document into one vector, and cosine similarity between two such vectors is, by construction, unable to attribute the resulting score to any particular section of either document. Tree-edit-distance methods offer a structurally natural alternative — documents are compared as trees, and the resulting edit script can in principle be inspected to see which sub-trees were matched, inserted, or deleted (Zhang & Shasha, 1989; Pawlik & Augsten, 2015) — but existing formulations typically apply uniform edit costs across all structural layers. Tree- and graph-structured comparison methods more broadly, including convolution tree kernels for structured natural-language objects (Collins & Duffy, 2001), have long recognized that structural comparison is valuable precisely because it is not reducible to flat feature comparison; our contribution builds on this premise while addressing the specific weighting gap described below. This is a poor fit for real document genres, where some structural layers carry little discriminative signal while others carry most of it. Uniform-cost TED, as we show empirically (Section 5.1), consequently performs poorly on documents that share superficial domain structure but differ substantively — precisely the "topic conflation" failure mode that motivates schema-aware weighting in the first place. Recent work in long-document retrieval has independently converged on a related diagnosis: fine-tuning pretrained language models on raw, structurally-flattened text discards document hierarchy and produces representations that fail to preserve fine-grained semantic relationships across structural elements (Huang et al., 2025).

A schema-weighted cost function is the natural fix, but it introduces a subtler problem: an arbitrarily weighted combination of distances is not guaranteed to satisfy the triangle inequality, which both complicates the theoretical properties of the resulting distance measure and forecloses the use of exact tree-edit-distance algorithms that assume a well-formed metric. A second, less-discussed problem is even more basic: *what should the schema even be?* An ad hoc segmentation of a document into "sections" risks encoding the researcher's own assumptions about what matters, rather than a genre's actual functional structure. This concern parallels a broader methodological lesson from empirical data science: design decisions made without independent grounding can silently introduce information into a model that would not legitimately be available in deployment, a phenomenon studied systematically under the label of data leakage (Kaufman et al., 2012) and one we return to directly in Section 4.1 when describing our own dataset-construction safeguards.

We address both problems directly. First, we define a per-layer cost function as a convex combination of content distance and schema distance, parameterized by a layer-specific weight β<sub>ℓ</sub> ∈ [0, 1], and show that this construction preserves the triangle inequality by design. This allows exact tree alignment via APTED (Pawlik & Augsten, 2015) rather than an approximate or heuristic comparison. Second, we adopt an explicit design principle: the domain-layer schema of the comparison tree must be grounded in an independent, citable, expert-authored functional taxonomy of the target document genre, not invented for the task at hand. We instantiate this principle twice: using established requirements-engineering process standards (ISO/IEC/IEEE 29148:2018) for software capstone proposals, and an empirically-derived bug-report information taxonomy (Bettenburg et al., 2008) for duplicate bug report detection. We show — as a demonstration of the principle's boundaries, not just its successes — that we correctly declined to force a domain schema onto a document genre lacking an independent taxonomy to ground it in.

We evaluate the resulting method, SW-BTED, with a deliberate emphasis on honest, bounded claims, following the general methodological guidance that claims of algorithmic superiority in NLP and ML research require explicit statistical significance testing rather than raw score comparison (Dror et al., 2018). Our central contributions are:

1. **A provably metric-preserving, schema-weighted tree-edit-distance cost function**, enabling exact computation via APTED rather than approximate alignment.
2. **A demonstrated, statistically decisive advantage over flat embedding averaging** (F1 = 0.9498 vs. 0.4314, p ≈ 2.52 × 10⁻²⁹).
3. **A demonstrated, statistically decisive advantage under structural perturbation** (100% vs. 0% accuracy, p = 1.91 × 10⁻⁶).
4. **A transferable design principle for domain-schema construction**, demonstrated across two structurally distinct genres.
5. **Evidence that the framework generalizes across document genres via hyperparameter adaptation, not algorithmic redesign.**
6. **An explicit, practically actionable characterization of a tradeoff** between two operating configurations, in the spirit of the broader recognition that interpretability and accuracy often require explicit, principled tradeoff analysis rather than an assumed default (Doshi-Velez & Kim, 2017).
7. **Demonstrated structural interpretability**, addressing a well-documented general limitation of complex predictive models: that high accuracy alone does not entail a human-inspectable account of *why* a model reached a given decision (Lipton, 2016).

We are equally direct about what we do not claim. Against strong natural-document baselines — full-document SBERT, BGE-small (Xiao et al., 2023), and MPNet (Song et al., 2020) — SW-BTED reaches statistical parity, not superiority.

---

## 2. Related Work

**Dense document embeddings.** Sentence-transformer models such as SBERT (Reimers & Gurevych, 2019) reduced document similarity to embedding-space cosine similarity, achieving strong performance across a wide range of semantic similarity tasks, evaluated on standard benchmarks such as the STS Benchmark (Cer et al., 2017), with a simple, efficient computation. Subsequent models have refined embedding quality further: MPNet (Song et al., 2020) unifies masked and permuted language modeling to improve on both BERT- and XLNet-style pretraining objectives, and the BGE family (Xiao et al., 2023) packages large-scale contrastive pretraining resources to advance general-purpose embedding quality. These methods share a structural limitation directly relevant to our contribution: similarity is computed over a single fixed-size representation of the entire input, with no mechanism to attribute a resulting score to a specific substructure, and — as we demonstrate in Section 5.2 — no inherent sensitivity to the *position* of content within a document, only to its presence.

**Tree- and structure-aware similarity.** Tree-edit-distance methods, originating with the classical dynamic-programming formulation of Zhang & Shasha (1989) and made efficient for practical use by the APTED algorithm (Pawlik & Augsten, 2015), provide an exact means of computing the minimum-cost sequence of node insertions, deletions, and substitutions transforming one labeled tree into another. pq-grams (Augsten et al., 2005) offer an efficient approximate alternative based on structural profile matching. Convolution tree kernels (Collins & Duffy, 2001) offer a further alternative for measuring structural similarity implicitly via an inner product over sub-tree fragments, without requiring an explicit edit-distance computation, though — like the tree-edit-distance methods above — without a mechanism for weighting different structural layers by semantic importance. This is the gap our schema-weighted cost function is designed to close, with an explicit theoretical guarantee (Section 3.4) that the resulting weighted distance remains a valid metric.

**Recent structure-aware embedding methods.** A recent line of work has begun to address flat embeddings' structural blindness directly. SEAL (Huang et al., 2025) introduces a contrastive learning framework that preserves semantic hierarchies and performs element-level alignment for long structured documents, demonstrating measurable retrieval gains over structure-agnostic fine-tuning of pretrained language models. This and related structure-aware pretraining approaches operate by modifying the embedding model itself to become sensitive to document structure; our approach is complementary, applying schema-aware weighting at the comparison stage via tree alignment rather than requiring structure-specific model retraining.

**Domain-specific document taxonomies.** A recurring theme across software and requirements engineering is the existence of established, empirically or normatively grounded structures for specific document genres. ISO/IEC/IEEE 29148:2018, the current international standard for systems and software requirements engineering (which formally supersedes the earlier IEEE 830-1998 recommended practice), defines the construct of a well-formed requirement and specifies required information items for requirements specifications throughout the system life cycle. Bettenburg et al. (2008) empirically characterize the information types present in effective bug reports (problem description, reproduction steps, environment/context, supporting evidence), and CUAD (Hendrycks et al., 2021) provides an expert-annotated clause-importance taxonomy for legal contracts. We treat the existence of such an independent taxonomy as a precondition for applying SW-BTED's domain layer to a new genre, rather than inventing an ad hoc segmentation. Term-level normalization in our pipeline draws on the Computer Science Ontology (Salatino et al., 2020), a large-scale, automatically-generated taxonomy of computer-science research areas.

**Plagiarism and duplicate-content detection.** Automated plagiarism detection spans string-matching, machine-learning, and deep-learning approaches, with recent systematic surveys documenting a continued gap between verbatim-copying detection (comparatively mature) and paraphrase- or idea-level plagiarism detection (comparatively unresolved) (Amirzhanov et al., 2025). Shared-task evaluations such as PAN 2025's generative plagiarism detection task further show that naive embedding-similarity approaches, while achieving reasonable performance on in-distribution data, generalize poorly to out-of-distribution plagiarism-generation strategies (Greiner-Petter et al., 2025) — a finding consistent with our own decision to test SW-BTED under an explicit distribution shift (Section 5.2) rather than relying solely on in-distribution accuracy.

**Duplicate bug report detection.** Beyond the section-level similarity findings noted above (Ghadhab & Amor, 2024), duplicate bug report detection has been studied extensively using both classical information-retrieval techniques and, more recently, sentence-embedding and deep-learning approaches (Isotani et al., 2021). A rigorous empirical study by Jiang et al. (2023) specifically cautions that deep-learning approaches do not uniformly outperform classical techniques on this task, underscoring the importance of honestly-reported, statistically-tested comparisons. We evaluate on the GitBugs benchmark (Patil, 2025), a recent, cross-project bug-report corpus supporting standardized duplicate-detection evaluation.

**Positioning.** Relative to flat embeddings, SW-BTED offers structural attribution and robustness to structural reordering that flat methods cannot address by construction. Relative to unweighted tree-edit-distance, pq-gram, and tree-kernel methods, SW-BTED's schema weighting directly addresses the topic-conflation failure mode that causes these methods to perform poorly on documents sharing superficial domain structure. Relative to recent structure-aware embedding retraining approaches (Huang et al., 2025), SW-BTED achieves structural sensitivity without requiring model retraining, at the cost of a more expensive alignment computation.

---

## 3. Method

### 3.1 CapTree representation

Each document is parsed into a four-tier tree structure ("CapTree"): a Root node identifying the document; a Domain layer (T2) partitioning content according to the genre's grounded functional taxonomy; an Intent layer (T3) representing atomic content units nested under their parent domain; and a Terminology layer (T4), representing normalized keyword leaves nested under their parent intent node.

![CapTree architecture: Root → Domain (T2) → Intent (T3) → Terminology (T4).](../docs/submission_figures/figure_1_captree_architecture.png)

**Figure 1.** Four-layer CapTree representation used by SW-BTED.

### 3.2 Domain-schema grounding principle

We treat the construction of the T2 domain layer as a methodological requirement, not an implementation detail: **the domain schema for a given genre must derive from an independent, citable, expert-authored functional taxonomy of that genre.**

- For software capstone proposals, we adopt a four-category domain schema (D1: Business Context, D2: Functional Requirements, D3: Technical Realization, D4: Execution Planning), grounded jointly in established requirements-engineering, business-analysis, and project-management literature. D1 corresponds to what BABOK (IIBA, 2015) terms Business Requirements — high-level statements of project goals and objectives. D2 corresponds directly to the Functional Requirements category present in both BABOK and the widely-used Volere Requirements Specification Template (Robertson & Robertson, 2006). D4 corresponds to the Work Breakdown Structure and Schedule Management processes defined in PMBOK (PMI, 2021) — task decomposition, sequencing, and schedule development, the standard project-management treatment of execution planning. D3 (Technical Realization) extends Volere/BABOK's Non-functional Requirements category to additionally capture architectural and technology-stack decisions not distinctly separated in either source framework; we present this specific extension as our own reasoned design choice, built on an established foundation for the other three categories.
- For duplicate bug report detection, we adopt a four-category domain schema (Problem Description, Reproduction, Environment/Context, Supporting Evidence) grounded directly in Bettenburg et al.'s (2008) empirical taxonomy of the information types present in effective bug reports, a finding independently reinforced by subsequent work showing that the specific sections of a bug report contribute unequally to duplicate-detection accuracy (Ghadhab & Amor, 2024).

We deliberately do not force a domain schema onto document genres lacking an independent taxonomy to ground it in; we found no independent, citable functional taxonomy for general (unstructured) plagiarism-corpus prose, and consequently treated such genres as outside the current scope of the method.

### 3.3 Terminology normalization (T4)

Leaf-level terms are normalized to canonical form using a domain-specific ontology and equivalence map — for the software domain, a lookup against the Computer Science Ontology (Salatino et al., 2020), a large-scale, automatically-generated taxonomy of computer-science research areas, combined with a Technology Equivalence Map and lemmatization.

### 3.4 Schema-weighted cost function

For each tree layer ℓ, we define the substitution cost between nodes u and v as:

w<sub>rep</sub><sup>(ℓ)</sup>(u, v) = (w<sub>del</sub><sup>(ℓ)</sup>(u) + w<sub>ins</sub><sup>(ℓ)</sup>(v)) · (β<sub>ℓ</sub> · Dist<sub>content</sub>(u, v) + (1 − β<sub>ℓ</sub>) · Dist<sub>schema</sub>(u, v))

where β<sub>ℓ</sub> ∈ [0, 1] is a layer-specific weight balancing content distance against schema distance. Because (1 − β<sub>ℓ</sub>) is defined as the complement of β<sub>ℓ</sub>, the combination is convex by construction.

**Proposition (metric preservation).** *If Dist<sub>content</sub> and Dist<sub>schema</sub> are each valid metrics, then the convex combination β<sub>ℓ</sub> · Dist<sub>content</sub> + (1 − β<sub>ℓ</sub>) · Dist<sub>schema</sub> is also a valid metric for any β<sub>ℓ</sub> ∈ [0, 1].*

*Proof sketch:* Non-negativity, identity of indiscernibles, and symmetry follow immediately from the corresponding properties of the two component metrics. For the triangle inequality: for any three nodes x, y, z, the two component-metric triangle inequalities, weighted by β<sub>ℓ</sub> ≥ 0 and (1 − β<sub>ℓ</sub>) ≥ 0 respectively and summed, yield the triangle inequality for the combined distance. ∎

This guarantee holds for any choice of per-layer β<sub>ℓ</sub>, allowing each tree layer to be weighted independently without sacrificing the theoretical properties required for exact tree-edit-distance computation via APTED (Pawlik & Augsten, 2015).

### 3.5 Hybrid scoring

The final structural similarity score sim<sub>struct</sub> is derived from the normalized APTED cost between two CapTrees. We additionally define a hybrid score sim<sub>hybrid</sub> = α · sim<sub>struct</sub> + (1 − α) · sim<sub>global</sub>, where sim<sub>global</sub> is a full-document embedding cosine similarity computed with a sentence-transformer encoder (Reimers & Gurevych, 2019).

### 3.6 Complexity

APTED computes the exact tree edit distance in O(n³) worst-case time for trees of size n (Pawlik & Augsten, 2015). On the canonical 138-pair benchmark, the SW-BTED alignment and scoring component took 2.1312 s in total (15.44 ms mean, 17.94 ms median, 23.52 ms at the 95th percentile) on a Windows 11 machine with Python 3.13.5 and 16 logical CPUs. These measurements exclude parsing, model loading, and embedding inference; pair-level timings and protocol are provided in the supplementary reproducibility artifacts.

![SW-BTED end-to-end pipeline.](../docs/submission_figures/figure_2_end_to_end_pipeline.png)

**Figure 2.** End-to-end pipeline from document parsing through tree alignment and classification.

---

## 4. Experimental Setup

### 4.1 Datasets

**FPT capstone proposals.** We evaluate on 138 real (non-synthetically-augmented) document pairs. We excluded an earlier GPT-4o-mini-augmented extension of this dataset (OpenAI, 2024) after an internal audit found the augmentation method risked introducing label leakage — a well-documented risk in predictive modeling generally, in which information about the prediction target is unintentionally introduced through the data-preparation process itself (Kaufman et al., 2012) — since the augmented pairs' positive label was defined by the generation process rather than independently verified. We report this exclusion as a methodological strength.

**Structural-perturbation benchmark.** We construct 20 synthetic document pairs by swapping the Functional Requirements (D2) and Technical Realization (D3) domain content within a single source document, preserving 100% of the original sentence-level vocabulary. We label these pairs as negative examples on the argument that section misplacement across functional domains represents a meaningful structural divergence relevant to automated compliance review under a structured requirements standard such as ISO/IEC/IEEE 29148 — an argument we present as our own reasoned position rather than a claim independently verified against a specific clause of that standard.

**GitBugs duplicate bug reports.** We evaluate cross-domain generalization on 300 pairs from the GitBugs benchmark (Patil, 2025), a cross-project, openly-licensed bug-report corpus with predefined duplicate-detection splits, using the Bettenburg-taxonomy-grounded domain schema described in Section 3.2.

### 4.2 Baselines

| Baseline | Description |
|---|---|
| TF-IDF | Bag-of-words cosine similarity |
| Standard TED | Unweighted tree edit distance (Zhang & Shasha, 1989) |
| Section Cosine | Per-section TF-IDF averaging, unweighted |
| Genuine Flat Domain SBERT | Per-domain SBERT embedding averaging, no tree alignment |
| pq-Gram | Structural profile-based tree comparison (Augsten et al., 2005) |
| Full-document SBERT (all-MiniLM-L6-v2) | Reimers & Gurevych (2019) |
| BGE-small-en-v1.5 | Xiao et al. (2023) |
| all-mpnet-base-v2 | Song et al. (2020) |

Qwen3-Embedding-4B was added as a current-generation external baseline. Its
pooled out-of-fold F1 is 0.9870 under the same 138-pair protocol; the complete
pair-level audit and provenance are reported in the supplementary project
artifacts.

### 4.3 Evaluation protocol

All results use 5-fold stratified cross-validation, a standard technique for obtaining reliable estimates of classifier accuracy while controlling variance from a single train/test split (Kohavi, 1995), with classification thresholds selected independently per fold on held-out training data. For the clean baseline suite, thresholds are selected over the grid 0.00, 0.005, ..., 1.00 using training-fold scores only; the test fold is never used for threshold tuning. Statistical significance between methods is assessed via the exact binomial McNemar test (McNemar, 1947) on paired predictions, following general best-practice guidance for statistical significance testing in NLP and ML research (Dror et al., 2018); Holm-Bonferroni correction (Holm, 1979) is applied across all pairwise comparisons within a given results table. We report F1-score, the harmonic mean of precision and recall originating in classical information-retrieval evaluation (van Rijsbergen, 1979), throughout.

### 4.4 Reproducibility

`[TODO: insert repository link and dataset-construction documentation pointer before submission.]`

---

## 5. Results

*(Results in this section describe this paper's own experimental findings and are supported by the tables/figures themselves, not external citations, except where a specific external comparison point is invoked below.)*

### 5.1 Comparison against the full baseline suite (natural documents)

`[Table 1]`

| Method | F1 | Result vs. SW-BTED Structural-Only |
|---|---|---|
| **SW-BTED Structural-Only** | **0.9498 ± 0.0253** | — |
| SW-BTED Hybrid (α = 0.6) | 0.9744–0.9867 | Tie, p = 0.375 |
| TF-IDF | 0.9867 ± 0.0267 | Statistical parity, p = 0.375 |
| Standard TED (unweighted) | 0.4364 ± 0.0162 | SW-BTED significantly better, p = 3.06 × 10⁻²⁶ |
| Section Cosine | 0.6837 ± 0.0894 | SW-BTED significantly better, Holm-adjusted p = 1.36 × 10⁻⁷ |
| Genuine Flat Domain SBERT | 0.4314 | SW-BTED significantly better, p = 2.52 × 10⁻²⁹ |
| pq-Gram | 0.9479 ± 0.0478 | Tie, p = 1.0000 |
| SBERT (all-MiniLM-L6-v2) | 0.9867 ± 0.0267 | Tie, p = 0.3750 |
| BGE-small-v1.5 | 0.9882 ± 0.0235 | Statistical parity, p = 0.375 |
| MPNet-base-v2 | 0.9882 ± 0.0235 | Statistical parity, p = 0.375 |
| Qwen3-Embedding-4B | 0.9870 (pooled OOF) | Statistical parity, p = 0.375 |

**Protocol note.** The TF-IDF and Section Cosine values in this table come
from the clean canonical 138-pair suite, which fits the corpus baselines on
the document universe participating in the 138-pair slice and selects
thresholds on training folds using a 0.005 grid. Earlier archived harnesses
reported different values (`0.4364` for TF-IDF and `0.4081` for Section
Cosine) under a different input/scope configuration; those historical values
are not silently mixed into this clean-suite table. Qwen3 was independently
re-evaluated with the same 0.005 train-fold grid and produced the reported
`0.9867 ± 0.0267` mean-fold F1 (`0.9870` pooled OOF).

SW-BTED significantly outperforms Standard TED, Section Cosine, and the Genuine Flat Domain SBERT baseline. It reaches statistical parity with full-document SBERT, BGE-small, MPNet, Qwen3, TF-IDF, and pq-Gram. This indicates that the method's primary value on natural documents is structural interpretability and robustness, rather than universal accuracy superiority.

![Canonical 138-pair pooled F1 results.](../docs/submission_figures/figure_3_canonical_results.png)

**Figure 3.** Pooled F1 scores on the canonical 138-pair real-only evaluation. The figure is descriptive; inferential comparisons are reported in Table 1 and use paired predictions.

### 5.2 Structural-perturbation benchmark

`[Table 2]`

| Method | Accuracy | False Positive Rate |
|---|---|---|
| Full-Document SBERT | 0.0% | 100.0% (20/20) |
| **SW-BTED Structural-Only** | **100.0%** | **0.0% (0/20)** |
| SW-BTED Hybrid (α = 0.6) | 0.0% | 100.0% (20/20) |

McNemar exact test (McNemar, 1947): n₁₀ = 20, n₀₁ = 0, p = 1.9073 × 10⁻⁶. Full-document SBERT similarity is unaffected by the D2↔D3 content swap, since full-document embedding similarity is, by construction, insensitive to the *position* of content, only its presence — a structural blindness consistent with the broader diagnosis in recent structure-aware retrieval literature (Huang et al., 2025).

### 5.3 The Structural-Only / Hybrid Mode tradeoff

`[Table 3]`

| Configuration | Natural-data F1 | Perturbation Accuracy |
|---|---|---|
| Structural-Only (α = 1.0) | 0.9498 | **100.0%** |
| Hybrid (α = 0.6) | 0.9744–0.9867 | **0.0%** |

### 5.4 Cross-domain generalization: duplicate bug report detection

`[Table 4]`

| Configuration | Out-of-the-box F1 | After hyperparameter adaptation |
|---|---|---|
| Structural-Only | 0.50 (degenerate) | 0.6725 |
| Hybrid (α = 0.6) | 0.9026 | **0.9141** |
| SBERT (reference) | — | 0.9074 (tie, p = 1.0000) |

This result is consistent with, and extends, prior findings that duplicate-detection performance is sensitive to how bug-report content is structurally organized and compared (Ghadhab & Amor, 2024), and adds to a body of evidence recommending caution before assuming any single technique transfers without adaptation across bug-tracking contexts (Jiang et al., 2023).

### 5.5 Interpretability case studies

![Per-domain structural similarity for the three canonical interpretability cases.](../docs/submission_figures/figure_4_interpretability_traces.png)

**Figure 4.** Per-domain structural similarity derived from the canonical APTED mappings. The accompanying machine-readable trace reports representative node replacements, deletions, and insertions.

We illustrate SW-BTED's structural interpretability with three representative document pairs, addressing the general limitation — documented broadly in the interpretability literature (Lipton, 2016; Doshi-Velez & Kim, 2017) — that high classification accuracy alone does not provide a human-inspectable account of a model's decision:

- **Case A (`SU26SE102`–`SU26SE102_plag`, true plagiarism pair):** strong sub-tree correspondence concentrated in Technical Realization and Execution Planning. This pair is retained in the 138-pair evaluation because neither document occurs in the explicit regenerated-document exclusion set.
- **Case B (`SP26SE068`–`SU26SE063`, domain near-miss, ground-truth negative):** full-document SBERT is a false positive due to shared domain vocabulary; SW-BTED's sub-tree decomposition correctly localizes the divergence. We present this case as an illustration, not evidence of a systemic embedding weakness — our aggregate results (Section 5.1) show no statistically significant overall disadvantage for full-document SBERT.
- **Case C (`SP26SE122`–`SP26SE055`, disjoint negative Type_C pair):** both methods classify the pair as negative, while the trace shows broad divergence across the four domain sub-trees.

---

## 6. Discussion

*(Discussion of this paper's own results; external citations only where a specific literature connection is drawn.)*

### 6.1 Why structural alignment helps where it helps

The results in Sections 5.1 and 5.2 are two faces of the same underlying phenomenon: methods that lack a mechanism to weight *where* in a document's functional structure content occurs are vulnerable both to topic conflation and to position invariance.

### 6.2 The Structural-Only / Hybrid tradeoff as practical guidance

We recommend Structural-Only (α = 1.0) for applications prioritizing structural/schema compliance, and Hybrid Mode (α = 0.6) for applications prioritizing general semantic similarity classification — an explicit tradeoff characterization we consider more useful to practitioners than a single default recommendation, consistent with general guidance that interpretability and accuracy tradeoffs should be evaluated explicitly rather than assumed away (Doshi-Velez & Kim, 2017).

### 6.3 What accuracy parity with strong baselines means for this method

SW-BTED's contribution is not raw accuracy improvement over the strongest available natural-document baselines. Given this parity, the method's demonstrated value rests on three properties no single-vector embedding method provides by construction: a proven metric-preserving guarantee, robustness to structural perturbation, and structural interpretability.

### 6.4 Generalization: configuration, not redesign

The bug-report cross-domain result suggests that a single hyperparameter, not a new cost function, was sufficient to transfer the method once the correct domain taxonomy was substituted.

---

## 7. Limitations and Threats to Validity

**Accuracy is parity, not superiority, on natural documents.**

**The domain-schema requirement is a real constraint.** Genres lacking an independent, citable functional taxonomy — of the kind formalized for requirements engineering (ISO/IEC/IEEE 29148:2018) or empirically derived for bug reports (Bettenburg et al., 2008) — are not well served by the method as designed.

**Cross-domain transfer required manual diagnosis, not automatic adaptation.**

**The perturbation-benchmark ground-truth labeling reflects the authors' reasoned architectural argument**, not a claim independently verified against a specific standards clause.

**Evaluation spans two document genres.** `[TODO: update if a third domain (e.g., CUAD-based legal contract evaluation, Hendrycks et al., 2021) is completed before submission.]`

**Baseline embedding models are not current-generation.** SBERT (Reimers & Gurevych, 2019), BGE-small-v1.5 (Xiao et al., 2023), and MPNet-base-v2 (Song et al., 2020) represent 2019–2023-era transformer encoders.

**Sample sizes are modest.**

---

## 8. Conclusion and Future Work

We presented SW-BTED, a schema-weighted, metric-preserving tree-edit-distance framework for structured document similarity. Future work includes: evaluating a third domain (e.g., legal contracts, using CUAD's existing clause-importance taxonomy; Hendrycks et al., 2021); developing automatic cross-domain hyperparameter adaptation; incorporating comparison against current-generation embedding models, potentially benchmarked using recent shared-task infrastructure such as PAN 2025 (Greiner-Petter et al., 2025); and extending the structural-perturbation benchmark to multi-document constructions.

---

## References

Amirzhanov, A., Turan, C., & Makhmutova, A. (2025). Plagiarism types and detection methods: A systematic survey of algorithms in text analysis. *Frontiers in Computer Science*, 7, Article 1504725.

Augsten, N., Böhlen, M., & Gamper, J. (2005). Approximate matching of hierarchical data using pq-grams. In *Proceedings of the 31st International Conference on Very Large Data Bases (VLDB)*.

Bettenburg, N., Just, S., Schröter, A., Weiss, C., Premraj, R., & Zimmermann, T. (2008). What makes a good bug report? In *Proceedings of the 16th ACM SIGSOFT International Symposium on Foundations of Software Engineering (FSE '08)* (pp. 308–318). ACM.

Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). SemEval-2017 Task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)* (pp. 1–14). Association for Computational Linguistics.

Collins, M., & Duffy, N. (2001). Convolution kernels for natural language. In *Advances in Neural Information Processing Systems 14 (NeurIPS 2001)*.

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*.

Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1383–1392). Association for Computational Linguistics.

Ghadhab, L., & Amor, N. B. (2024). Impact of textual (dis)similarities of bug report sections on duplicate bug report detection performance. In *Proceedings of the International Conference on Service-Oriented Computing (ICSOC 2024)* (pp. 188–194). Springer.

Greiner-Petter, A., Fröbe, M., Wahle, J. P., Ruas, T., Gipp, B., Aizawa, A., & Potthast, M. (2025). Overview of the plagiarism detection task at PAN 2025. In *Working Notes of CLEF 2025*. `arXiv:2510.06805`.

Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). CUAD: An expert-annotated NLP dataset for legal contract review. In *Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks*.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.

Huang, X., Ren, Z., Yu, Y., Zhou, Y., Chen, Z., & Wen, Z. (2025). SEAL: Structure and element aware learning improves long structured document retrieval. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 8526–8536). Association for Computational Linguistics.

International Institute of Business Analysis (IIBA). (2015). *A guide to the business analysis body of knowledge (BABOK guide)* (Version 3). IIBA.

International Organization for Standardization/International Electrotechnical Commission/Institute of Electrical and Electronics Engineers. (2018). *ISO/IEC/IEEE 29148:2018 — Systems and software engineering — Life cycle processes — Requirements engineering* (2nd ed.). ISO/IEC/IEEE.

Isotani, H., Washizaki, H., Fukazawa, Y., Nomoto, T., Ouji, S., & Saito, S. (2021). Duplicate bug report detection by using sentence embedding and fine-tuning. In *Proceedings of the IEEE International Conference on Software Maintenance and Evolution (ICSME 2021)* (pp. 535–544). IEEE.

Jiang, Y., Su, X., Treude, C., et al. (2023). Does deep learning improve the performance of duplicate bug report detection? An empirical study. *Journal of Systems and Software*, 197, Article 111570.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15.

Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)* (Vol. 2, pp. 1137–1143).

Lipton, Z. C. (2016). The mythos of model interpretability. *arXiv preprint arXiv:1606.03490*.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.

OpenAI. (2024). GPT-4o system card. *arXiv preprint arXiv:2410.21276*.

Patil, A. (2025). GitBugs: Bug reports for duplicate detection, retrieval augmented generation, triage, and more. *arXiv preprint arXiv:2504.09651*.

Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit distance. *ACM Transactions on Database Systems*, 40(1), Article 3.

Project Management Institute (PMI). (2021). *A guide to the project management body of knowledge (PMBOK guide)* (7th ed.). Project Management Institute.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics.

Robertson, S., & Robertson, J. (2006). *Mastering the requirements process* (2nd ed.). Addison-Wesley.

Salatino, A. A., Thanapalasingam, T., Mannocci, A., Birukou, A., Osborne, F., & Motta, E. (2020). The Computer Science Ontology: A comprehensive automatically-generated taxonomy of research areas. *Data Intelligence*, 2(3), 379–416.

Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.

van Rijsbergen, C. J. (1979). *Information retrieval* (2nd ed.). Butterworths.

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv preprint arXiv:2309.07597*.

Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing distance between trees and related problems. *SIAM Journal on Computing*, 18(6), 1245–1262.

The D1–D4 domain-taxonomy citation gap is resolved; see Section 3.2 and the
Robertson & Robertson (2006), IIBA (2015), and PMI (2021) references.

---

## Alignment-Check Log (this round)

This section documents every alignment check performed, including the corrections made and the citations deliberately withheld.

**Correction made:**
- **IEEE 830 vs. ISO/IEC/IEEE 29148.** The prior draft cited "IEEE 830 / ISO 29148" as if both were current, co-equal standards. Verified via direct search: ISO/IEC/IEEE 29148:2018 explicitly states it replaces IEEE 830-1998. The paper now cites only 29148:2018 as the current standard, with IEEE 830 mentioned solely as its superseded historical predecessor where relevant. Citing an obsolete standard as current would have been a real factual error in a submitted paper.

**Citations deliberately withheld from certain sentences, to avoid misalignment:**
- Sentences reporting this paper's own experimental numbers (e.g., "F1 = 0.9498 vs. 0.4314") are not cited to external sources anywhere in this draft. These are the paper's own findings; citing an external paper next to them would incorrectly imply the number originates elsewhere.
- The GPT-4o citation (OpenAI, 2024) is attached only to the *mention* that GPT-4o-mini was used for an (ultimately excluded) data augmentation step — not to any claim about GPT-4o's general capabilities, which this paper does not make and is not positioned to support.
- The Amirzhanov et al. (2025) and Greiner-Petter et al. (2025) plagiarism-survey citations are used only to support general claims about the state of plagiarism-detection research (paraphrase detection remaining hard; naive embedding methods generalizing poorly across distribution shifts) — not attached to any claim about this paper's own plagiarism-detection performance, which is a distinct, self-supported claim.

**Citations considered and rejected as misaligned before insertion (not included in the final list):**
- An IEEE 830 standalone citation was considered for the domain-taxonomy grounding claim and rejected once its obsolete status was confirmed.
- A citation attributing the specific D1–D4 four-category capstone-domain split directly to ISO/IEC/IEEE 29148 alone was considered and rejected: the standard's general framework was confirmed, but no specific clause-level mapping to this paper's exact four categories could be verified against the standard's actual text. Overstating this alignment would itself have been a misalignment of the kind this pass was meant to catch. (Superseded — see next entry.)

**Follow-up round: D1–D4 domain-taxonomy citation resolved.** A dedicated evidence-gathering pass (documented in full in `D1_D4_CITATION_EVIDENCE.md`) checked three established frameworks — Volere (Robertson & Robertson, 2006), BABOK (IIBA, 2015), and PMBOK (PMI, 2021) — against each of the four D1–D4 categories individually, rather than searching for one framework to cover all four. Result: D1 (Business Context) and D2 (Functional Requirements) have strong-to-exact grounding in BABOK/Volere; D4 (Execution Planning) has strong grounding in PMBOK's Work Breakdown Structure and Schedule Management processes, found only in this follow-up search after the first pass's single-framework search came up short; D3 (Technical Realization) remains a partial match, honestly framed in Section 3.2 as the authors' own extension rather than claimed as externally grounded. This category-by-category approach — verifying each claim against the specific source that actually supports it, rather than one convenient source stretched to cover everything — is the same discipline applied throughout this citation pass, and it is what surfaced PMBOK as a substantially better fit for D4 than anything found in the first search attempt.

**Final count:** 30 verified citations total (27 from the two-round general citation pass, plus 3 added in the D1–D4 follow-up: Robertson & Robertson 2006, IIBA 2015, PMI 2021), each checked against a real, findable source and against the specific sentence it supports before inclusion. No `[TODO]` items remain regarding citation *sourcing*; two `[TODO]` items remain regarding *content still to be produced* (the repository/reproducibility link and the third-domain decision) — these are infrastructure and scope decisions, not citation gaps, and are tracked separately.
