# When Does Tree Structure Matter? A Schema-Weighted Edit Distance Framework for Semi-Structured Document Similarity

---

## Abstract

Comparing semi-structured documents — software requirement proposals and bug reports — is a task with real practical stakes: academic integrity review and duplicate-issue triage both depend on similarity judgments a human can act on and defend. Common flat-embedding and unweighted-TED approaches expose a practical tradeoff. Flat document embeddings achieve strong aggregate accuracy but collapse a document pair into a single scalar that does not by itself localize where the documents diverge. Tree-edit-distance (TED) methods are structurally attributable in principle, but unweighted formulations conflate superficial structural overlap with substantive similarity, producing high false-positive rates on documents that merely share a domain.

We propose SW-BTED, a schema-weighted tree-edit-distance framework in which each tree layer's node-level cost combines content and schema distance under a formal sufficient condition for preserving the triangle inequality, while APTED computes the minimum-cost alignment under the implemented edit-cost model. Critically, we ground each domain-layer schema primarily in independent, citable functional taxonomies while documenting justified extensions, and we demonstrate this principle across two structurally distinct domains: software capstone proposals and duplicate bug reports.

On a 138-pair real-document capstone benchmark, SW-BTED obtains F1 = 0.9498, while several lexical and neural single-vector baselines obtain approximately 0.99; paired tests do not detect a statistically significant difference against those strongest baselines. Controlled schema-reassignment and observable perturbation diagnostics show that the tree method responds to tested functional-organization changes, while providing inspectable structural attribution through subtree edit traces. A second, structurally distinct genre is used for an exploratory transfer study with target-domain hyperparameter adaptation; its split provenance is reported separately from the primary benchmark. We characterize an explicit tradeoff between two operating configurations of the method.

---

## 1. Introduction

Institutions that must compare large volumes of semi-structured documents face a recurring, practically important question: not just *how similar* are two documents, but *where*, specifically, do they diverge, and does that divergence matter? An academic integrity committee reviewing a suspected plagiarism case needs to know whether two capstone proposals share only superficial domain vocabulary or actually reuse a project's core technical approach — a distinction that recent systematic surveys of plagiarism-detection methods confirm remains genuinely difficult, since paraphrase- and idea-level plagiarism resist simple string-matching approaches (Amirzhanov et al., 2025). A bug-triage lead needs to know whether two reports describe the same underlying defect or merely occur in the same subsystem — a task shown to depend heavily on which sections of a report actually overlap, not just whether the reports are lexically similar overall (Ghadhab & Amor, 2024). A single similarity score — however accurate in aggregate, as measured by standard classification metrics such as the F-measure (van Rijsbergen, 1979) — does not by itself answer either question; without additional attribution, it must be interpreted as a document-level judgment.

This creates a persistent tension in how such comparisons are made. Dense document embeddings, following the sentence-transformer paradigm (Reimers & Gurevych, 2019) and evaluated on benchmarks such as the STS Benchmark for semantic textual similarity (Cer et al., 2017), achieve strong aggregate discrimination and are the default tool for this kind of task, but they compress an entire document into one vector, and cosine similarity between two such vectors is, by construction, unable to attribute the resulting score to any particular section of either document. Tree-edit-distance methods offer a structurally natural alternative — documents are compared as trees, and the resulting edit script can in principle be inspected to see which sub-trees were matched, inserted, or deleted (Zhang & Shasha, 1989; Pawlik & Augsten, 2015) — but standard document-oriented TED baselines commonly use uniform or structure-agnostic edit costs, although weighted TED formulations exist in the broader literature. Tree- and graph-structured comparison methods more broadly, including convolution tree kernels for structured natural-language objects (Collins & Duffy, 2001), have long recognized that structural comparison is valuable precisely because it is not reducible to flat feature comparison; our contribution builds on this premise while addressing the specific weighting gap described below. This is a poor fit for real document genres, where some structural layers carry little discriminative signal while others carry most of it. In our benchmark, uniform-cost TED performs poorly on documents that share superficial domain structure but differ substantively — a result consistent with the "topic conflation" failure mode that motivates schema-aware weighting in this study. Recent work in long-document retrieval has independently converged on a related diagnosis: fine-tuning pretrained language models on raw, structurally-flattened text discards document hierarchy and produces representations that fail to preserve fine-grained semantic relationships across structural elements (Huang et al., 2025).

A schema-weighted cost function is the natural fix, but an unconstrained weighting scheme makes the mathematical properties of the resulting node-level distance difficult to characterize. A second, less-discussed problem is even more basic: *what should the schema even be?* An ad hoc segmentation of a document into "sections" risks encoding the researcher's own assumptions about what matters, rather than a genre's actual functional structure. This concern parallels a broader methodological lesson from empirical data science: design decisions made without independent grounding can silently introduce information into a model that would not legitimately be available in deployment, a phenomenon studied systematically under the label of data leakage (Kaufman et al., 2012) and one we return to directly in Section 4.1 when describing our own dataset-construction safeguards.

We address both problems directly. First, we define a per-layer node-level combination of content distance and schema distance, parameterized by a layer-specific weight β<sub>ℓ</sub> ∈ [0, 1], and state the conditions under which that combination satisfies the triangle inequality. APTED is then used to compute the minimum-cost alignment under the implemented edit-cost model. Second, we adopt an explicit design principle: each domain-layer schema is grounded primarily in an independent, citable functional taxonomy, while justified extensions are documented explicitly. We instantiate this principle twice: using established requirements-engineering process standards (ISO/IEC/IEEE 29148:2018) for software capstone proposals, and an empirically-derived bug-report information taxonomy (Bettenburg et al., 2008) for duplicate bug report detection. We show — as a demonstration of the principle's boundaries, not just its successes — that we did not force a domain schema onto a document genre lacking a suitable independently grounded taxonomy.

We evaluate the resulting method, SW-BTED, with a deliberate emphasis on honest, bounded claims, following the general methodological guidance that claims of algorithmic superiority in NLP and ML research require explicit statistical significance testing rather than raw score comparison (Dror et al., 2018). Our central contributions are:

1. **A four-layer CapTree and a genre-grounding design principle** for semi-structured document comparison.
2. **A schema/content-weighted edit-cost formulation** with a formal sufficient condition for preserving the triangle inequality and an inspectable APTED alignment.
3. **A rigorous evaluation across natural pairs and controlled structural diagnostics**, including document-disjoint robustness and an observable perturbation audit.
4. **An exploratory second-genre transfer study** with domain-specific adaptation, without changing the core alignment algorithm.
5. **Inspectable subtree attribution and an explicit structural-only/hybrid operating tradeoff.**

We are equally direct about what we do not claim. Against strong natural-document baselines — single-vector SBERT, BGE-small (Xiao et al., 2023), MPNet (Song et al., 2020), and Qwen3 under model-specific truncation — SW-BTED has lower raw F1 in this benchmark, but the paired McNemar tests do not detect a statistically significant difference. This is not an equivalence claim.

### Research questions

We organize the empirical study around three research questions:

- **RQ1:** How does SW-BTED perform on natural semi-structured document pairs relative to lexical, unweighted structural, schema-matched, and single-vector embedding baselines evaluated under their stated truncation protocols?
- **RQ2:** How does SW-BTED respond to controlled structural perturbations, and which schema layers contribute to the resulting decisions?
- **RQ3:** What changes are required to transfer the method to a second document genre, and does transfer require algorithmic redesign or only domain-specific adaptation?

---

## 2. Related Work

**Weighted tree edit distance.** Weighted tree-edit-distance formulations predate this work, including node-weighted and approximate approaches such as Torsello and Hancock (2001), while recent theory studies weighted tree edit distance and its computational complexity (Das et al., 2023). We therefore do not claim to introduce weighted tree edit distance itself. SW-BTED's narrower contribution is the combination of genre-grounded functional taxonomies, a four-layer semi-structured document tree, layer-specific schema/content costs, and inspectable edit traces for document similarity.

Here, references to uniform costs concern standard document-oriented TED baselines, not the full prior-art family of weighted TED formulations. Existing weighted formulations establish that non-uniform costs are possible; SW-BTED ties those costs explicitly to genre-grounded functional layers.

**Dense document embeddings.** Sentence-transformer models such as SBERT (Reimers & Gurevych, 2019) reduced document similarity to embedding-space cosine similarity, achieving strong performance across a wide range of semantic similarity tasks, evaluated on standard benchmarks such as the STS Benchmark (Cer et al., 2017), with a simple, efficient computation. Subsequent models have refined embedding quality further: MPNet (Song et al., 2020) unifies masked and permuted language modeling to improve on both BERT- and XLNet-style pretraining objectives, and the BGE family (Xiao et al., 2023) packages large-scale contrastive pretraining resources to advance general-purpose embedding quality. These methods share a structural limitation directly relevant to our contribution: similarity is computed over a single fixed-size representation of the entire input, with no mechanism to attribute a resulting score to a specific substructure. Their response to document-level reorganization depends on the encoder and input-construction protocol rather than on an explicit functional alignment.

**Tree- and structure-aware similarity.** Tree-edit-distance methods, originating with the classical dynamic-programming formulation of Zhang & Shasha (1989) and made efficient for practical use by the APTED algorithm (Pawlik & Augsten, 2015), provide an exact means of computing the minimum-cost sequence of node insertions, deletions, and substitutions transforming one labeled tree into another. pq-grams (Augsten et al., 2005) offer an efficient approximate alternative based on structural profile matching. Convolution tree kernels (Collins & Duffy, 2001) offer a further alternative for measuring structural similarity implicitly via an inner product over sub-tree fragments, without requiring an explicit edit-distance computation, though — like the tree-edit-distance methods above — without the explicit genre-grounded layer-weighting mechanism studied here. This is the gap our schema-weighted cost function is designed to address, with a formal sufficient condition for preserving the triangle inequality (Section 3.4) rather than a claim that the complete implemented edit-cost function is a metric.

**Recent structure-aware embedding methods.** A recent line of work has begun to address flat embeddings' structural blindness directly. SEAL (Huang et al., 2025) introduces a contrastive learning framework that preserves semantic hierarchies and performs element-level alignment for long structured documents, demonstrating measurable retrieval gains over structure-agnostic fine-tuning of pretrained language models. This and related structure-aware pretraining approaches operate by modifying the embedding model itself to become sensitive to document structure; our approach is complementary, applying schema-aware weighting at the comparison stage via tree alignment rather than requiring structure-specific model retraining.

**Domain-specific document taxonomies.** A recurring theme across software and requirements engineering is the existence of established, empirically or normatively grounded structures for specific document genres. ISO/IEC/IEEE 29148:2018, the current international standard for systems and software requirements engineering (which formally supersedes the earlier IEEE 830-1998 recommended practice), defines the construct of a well-formed requirement and specifies required information items for requirements specifications throughout the system life cycle. Bettenburg et al. (2008) empirically characterize the information types present in effective bug reports (problem description, reproduction steps, environment/context, supporting evidence), and CUAD (Hendrycks et al., 2021) provides 41 expert-annotated clause types for legal contract review. We treat the existence of such an independent taxonomy as a precondition for applying SW-BTED's domain layer to a new genre, rather than inventing an ad hoc segmentation. Term-level normalization in our pipeline draws on the Computer Science Ontology (Salatino et al., 2020), a large-scale, automatically-generated taxonomy of computer-science research areas.

**Plagiarism and duplicate-content detection.** Automated plagiarism detection spans string-matching, machine-learning, and deep-learning approaches, with recent systematic surveys documenting a continued gap between verbatim-copying detection (comparatively mature) and paraphrase- or idea-level plagiarism detection (comparatively unresolved) (Amirzhanov et al., 2025). Shared-task evaluations such as PAN 2025's generative plagiarism detection task further show that naive embedding-similarity approaches, while achieving reasonable performance on in-distribution data, generalize poorly to out-of-distribution plagiarism-generation strategies (Greiner-Petter et al., 2025) — a finding consistent with our own decision to test SW-BTED under an explicit distribution shift (Section 5.2) rather than relying solely on in-distribution accuracy.

**Duplicate bug report detection.** Beyond the section-level similarity findings noted above (Ghadhab & Amor, 2024), duplicate bug report detection has been studied extensively using both classical information-retrieval techniques and, more recently, sentence-embedding and deep-learning approaches (Isotani et al., 2021). A rigorous empirical study by Jiang et al. (2023) specifically cautions that deep-learning approaches do not uniformly outperform classical techniques on this task, underscoring the importance of honestly-reported, statistically-tested comparisons. We include an exploratory GitBugs transfer artifact (Patil, 2025), while keeping its split and adaptation provenance separate from the canonical evaluation evidence.

**Positioning.** Relative to flat embeddings, SW-BTED offers structural attribution and, in the tested controlled benchmarks, sensitivity to explicit structural changes. Relative to unweighted tree-edit-distance, pq-gram, and tree-kernel methods, SW-BTED contributes an explicit schema-weighting mechanism; this paper does not evaluate every member of those method families and therefore does not claim a family-wide performance ranking. Relative to recent structure-aware embedding retraining approaches (Huang et al., 2025), SW-BTED achieves the tested structural sensitivity without requiring model retraining, at the cost of a more expensive alignment computation.

---

## 3. Method

### 3.1 CapTree representation

Each document is parsed into a four-tier tree structure ("CapTree"): a Root node identifying the document; a Domain layer (T2) partitioning content according to the genre's grounded functional taxonomy; an Intent layer (T3) representing the parser's atomic requirement/content units nested under their parent domain; and a Terminology layer (T4), representing normalized keyword leaves nested under their parent intent node. In the canonical implementation, T3 content distance is cosine distance between stored intent embeddings, while T4 content distance is exact equality after normalization.

The exact T3 extraction rules, Technology Equivalence Map, and associated normalization resources are versioned in the repository under `repro_candidate_138/src/01_parser.py`, `repro_candidate_138/src/03_normalizer.py`, and `repro_candidate_138/src/tech_equivalence.py`; their current source commit is recorded in the reproducibility manifest.

![CapTree architecture: Root → Domain (T2) → Intent (T3) → Terminology (T4).](../docs/submission_figures/figure_1_captree_architecture.png)

**Figure 1.** Four-layer CapTree representation used by SW-BTED.

### 3.2 Domain-schema grounding principle

We treat the construction of the T2 domain layer as a methodological requirement, not an implementation detail: **domain schemas should be primarily grounded in independently established, citable functional taxonomies; any extensions must be explicitly documented and justified.**

- For software capstone proposals, we adopt a four-category domain schema (D1: Business Context, D2: Functional Requirements, D3: Technical Realization, D4: Execution Planning), grounded jointly in established requirements-engineering, business-analysis, and project-management literature. D1 corresponds to what BABOK (IIBA, 2015) terms Business Requirements — high-level statements of project goals and objectives. D2 corresponds directly to the Functional Requirements category present in both BABOK and the widely-used Volere Requirements Specification Template (Robertson & Robertson, 2006). D4 corresponds to the Work Breakdown Structure and Schedule Management processes defined in PMBOK (PMI, 2021) — task decomposition, sequencing, and schedule development, the standard project-management treatment of execution planning. D3 (Technical Realization) extends Volere/BABOK's Non-functional Requirements category to additionally capture architectural and technology-stack decisions not distinctly separated in either source framework; we present this specific extension as our own reasoned design choice, built on an established foundation for the other three categories.
- For duplicate bug report detection, we adopt a four-category domain schema (Problem Description, Reproduction, Environment/Context, Supporting Evidence) grounded directly in Bettenburg et al.'s (2008) empirical taxonomy of the information types present in effective bug reports, a finding independently reinforced by subsequent work showing that the specific sections of a bug report contribute unequally to duplicate-detection accuracy (Ghadhab & Amor, 2024).

We deliberately do not force a domain schema onto document genres lacking an independent taxonomy to ground it in; we did not identify an independent, citable functional taxonomy suitable for the general unstructured plagiarism-corpus prose considered during development, and consequently treated such genres as outside the current scope of the method.

### 3.3 Terminology normalization (T4)

Leaf-level terms are normalized to canonical form using a domain-specific ontology and equivalence map — for the software domain, a lookup against the Computer Science Ontology (Salatino et al., 2020), a large-scale, automatically-generated taxonomy of computer-science research areas, combined with a Technology Equivalence Map and lemmatization. The canonical implementation uses schema distance 0 for equal domain/schema classes and the documented domain-pair penalty matrix otherwise; missing domain children are charged their insertion/deletion cost.

### 3.4 Schema-weighted cost function

For each tree layer ℓ, we define the substitution cost between nodes u and v as:

w<sub>rep</sub><sup>(ℓ)</sup>(u, v) = (w<sub>del</sub><sup>(ℓ)</sup>(u) + w<sub>ins</sub><sup>(ℓ)</sup>(v)) · (β<sub>ℓ</sub> · Dist<sub>content</sub>(u, v) + (1 − β<sub>ℓ</sub>) · Dist<sub>schema</sub>(u, v))

where β<sub>ℓ</sub> ∈ [0, 1] is a layer-specific weight balancing content distance against schema distance. Because (1 − β<sub>ℓ</sub>) is defined as the complement of β<sub>ℓ</sub>, the combination is convex by construction.

For the canonical four-layer run, the configured values are β<sub>T2</sub> = 0.0, β<sub>T3</sub> = 0.9, and β<sub>T4</sub> = 0.8; Root substitutions have zero cost. Domain insertion and deletion weights are 2.0, Intent insertion and deletion weights are 1.0, and terminology weights use the stored TF–IDF-derived leaf weight with the implementation's 0.5 scaling. The pre-filter threshold is 0.25 and the edit-budget ratio ρ is 0.80. These are fixed configuration values for the canonical run, not values re-estimated on the test folds.

The canonical domain-penalty matrix is:

| | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| D1 | 0.0 | 0.8 | 0.9 | 0.9 |
| D2 | 0.8 | 0.0 | 0.5 | 0.7 |
| D3 | 0.9 | 0.5 | 0.0 | 0.6 |
| D4 | 0.9 | 0.7 | 0.6 | 0.0 |

This finite matrix is symmetric, has zero diagonal, and satisfies the triangle-inequality checks. The matrix defines a metric over the four domain labels; when lifted to distinct node instances through their labels, it need not satisfy identity of indiscernibles over the node-instance space. The proposition nevertheless remains conditional because the implemented T3 content distance is cosine-derived rather than a separately verified metric.

**Proposition (conditional triangle-inequality preservation).** *If Dist<sub>content</sub> and Dist<sub>schema</sub> each satisfy the triangle inequality on the comparison space, then the unscaled combination β<sub>ℓ</sub> · Dist<sub>content</sub> + (1 − β<sub>ℓ</sub>) · Dist<sub>schema</sub> also satisfies the triangle inequality for any β<sub>ℓ</sub> ∈ [0, 1].*

*Proof sketch:* For any three comparable nodes x, y, and z, apply the triangle inequality to each component and multiply the resulting inequalities by β<sub>ℓ</sub> ≥ 0 and (1 − β<sub>ℓ</sub>) ≥ 0. Adding the two inequalities yields the triangle inequality for the combined distance. ∎

Accordingly, the proposition is a sufficient mathematical condition for triangle-inequality preservation; we do not claim that every canonical node-level distance used in the experiments instantiates a metric. If both components are metrics on the same underlying comparison space, additional metric properties may follow under the corresponding identity assumptions; this is not claimed for every canonical node representation used here. The proposition is deliberately limited to the unscaled combination. The implemented replacement cost additionally multiplies this combination by layer-specific insertion/deletion weights; therefore, we do not claim here that the complete tree-edit-distance cost is itself a metric. APTED is used as an exact optimizer for the implemented edit-cost model, subject to its algorithmic assumptions (Pawlik & Augsten, 2015).

### 3.5 Hybrid scoring

The final structural similarity score is normalized from the APTED cost as follows. Let C be the edit cost, C<sub>max</sub> the sum of all deletion and insertion costs for the two trees, and ρ = 0.80 the configured edit-budget ratio. We compute q = C/C<sub>max</sub> and sim<sub>struct</sub> = max(0, 1 − q/ρ); pairs with q > ρ are assigned zero structural similarity. The hybrid score is sim<sub>hybrid</sub> = α · sim<sub>struct</sub> + (1 − α) · sim<sub>global</sub>, where sim<sub>global</sub> is a single-vector document-embedding cosine similarity computed under the encoder's stated input-length protocol and α = 0.60 in the configured hybrid experiment. Structural-only evaluation sets α = 1.0.

### 3.6 Complexity

APTED computes the exact tree edit distance in O(n³) worst-case time for trees of size n (Pawlik & Augsten, 2015). On the canonical 138-pair benchmark, the SW-BTED alignment and scoring component took 2.1312 s in total (15.44 ms mean, 17.94 ms median, 23.52 ms at the 95th percentile) on a Windows 11 machine with Python 3.13.5 and 16 logical CPUs. These measurements exclude parsing, model loading, and embedding inference; pair-level timings and protocol are provided in the supplementary reproducibility artifacts.

![SW-BTED end-to-end pipeline.](../docs/submission_figures/figure_2_end_to_end_pipeline.png)

**Figure 2.** End-to-end pipeline from document parsing through tree alignment and classification.

---

## 4. Experimental Setup

### 4.1 Datasets

**FPT capstone proposals.** We evaluate on 138 real (non-synthetically-augmented) document pairs. We excluded an earlier GPT-4o-mini-augmented extension of this dataset after an internal audit found the augmentation method risked introducing label leakage — a well-documented risk in predictive modeling generally, in which information about the prediction target is unintentionally introduced through the data-preparation process itself (Kaufman et al., 2012) — since the augmented pairs' positive label was defined by the generation process rather than independently verified. We therefore exclude the augmented extension from all canonical results.

| Dataset statistic | Value | Provenance |
|---|---:|---|
| Unique documents | 178 | Canonical `full_texts.json` / `pairs.csv` |
| Evaluated pairs | 138 | Canonical `pairs.csv`, SHA-256 in the manifest |
| Positive pairs | 38 | `Type_A` rows; `label=1` in `pairs.csv` |
| Negative pairs | 100 | `Type_B` (50) and `Type_C` (50); `label=0` in `pairs.csv` |
| Independent human annotation record | Not present in current package | Original label-construction metadata are not available in the current frozen artifact |

The current package preserves the pair labels and type names but not a complete independent annotation or pair-construction log. Accordingly, the 138-pair results are reported as benchmark evidence under the frozen dataset manifest, not as a newly human-annotated corpus.

**Structural-perturbation benchmark.** We construct 20 synthetic document pairs by swapping the Functional Requirements (D2) and Technical Realization (D3) domain content within a single source document, preserving 100% of the original sentence-level vocabulary. We label these pairs as negative examples on the argument that section misplacement across functional domains represents a meaningful structural divergence relevant to automated compliance review under a structured requirements standard such as ISO/IEC/IEEE 29148 — an argument we present as our own reasoned position rather than a claim independently verified against a specific clause of that standard.

**GitBugs duplicate bug reports.** We report an exploratory cross-domain transfer artifact involving 300 GitBugs pairs (Patil, 2025), using the Bettenburg-taxonomy-grounded domain schema described in Section 3.2. The available project artifact does not preserve a complete split manifest or machine-readable adaptation record, so this result is not treated as confirmed held-out generalization.

### 4.2 Baselines

| Baseline | Description |
|---|---|
| TF-IDF | Bag-of-words cosine similarity |
| Standard TED | Unweighted tree edit distance (Zhang & Shasha, 1989) |
| Section Cosine | Per-section TF-IDF averaging, unweighted |
| Genuine Flat Domain SBERT | Per-domain SBERT embedding averaging, no tree alignment; F1 = 0.4314 ± 0.0160 |
| pq-Gram | Structural profile-based tree comparison (Augsten et al., 2005) |
| Single-vector MiniLM (256-token truncation) | Reimers & Gurevych (2019) |
| Single-vector BGE-small (512-token truncation) | Xiao et al. (2023) |
| Single-vector MPNet (384-token truncation) | Song et al. (2020) |

Qwen3-Embedding-4B (Zhang et al., 2025) was added as a current-generation single-vector external baseline under a 2048-token truncation protocol. Its
pooled out-of-fold F1 is 0.9870 under the same 138-pair protocol; the complete
pair-level audit and provenance are reported in the supplementary project
artifacts.

For the schema-matched embedding audit, the same D1–D4 text blocks are embedded independently and their four cosine scores are averaged without tree alignment. MiniLM obtains F1 = 0.4314 ± 0.0160 (TP=38, FP=100, TN=0, FN=0); the independent BGE-small rerun produces the same degenerate all-positive prediction pattern. These are diagnostic baselines for isolating schema decomposition, not substitutes for the stronger single-vector embedding baselines.

### 4.3 Evaluation protocol

The primary 138-pair natural-document results in Table 1 use 5-fold stratified cross-validation, a standard technique for obtaining reliable estimates of classifier accuracy while controlling variance from a single train/test split (Kohavi, 1995), with classification thresholds selected independently per fold using only the training portion of that fold. For the clean TF-IDF and Section Cosine baselines, vectorizers are fitted separately within each fold using only documents in the training portion, and the learned transformations are applied to the test-fold documents. Thresholds are selected over the grid 0.00, 0.005, ..., 1.00 using training-fold scores only; the test fold is never used for threshold tuning. Statistical significance between methods is assessed via the exact binomial McNemar test (McNemar, 1947) on paired predictions, following general best-practice guidance for statistical significance testing in NLP and ML research (Dror et al., 2018). Holm-Bonferroni correction (Holm, 1979) is applied over the nine planned baseline comparisons against SW-BTED Structural-Only; the Hybrid operating-point audit is excluded from this inferential family. McNemar p-values test paired disagreement outcomes, not equality of F1 values; accordingly, “no statistically significant difference detected” is not treated as an equivalence claim. Mean-fold F1, pooled out-of-fold F1, and MCC are reported in Table 1; precision and recall are provided in the supplementary machine-readable results. Supplemental evaluations use the protocols described in their respective subsections.

### 4.4 Reproducibility

Reproducibility package: [SW-BTED repository](https://github.com/Tuan-Kiet-Le/SW_BTED), release tag `v1.0.0-canonical-138-2026-08-24`; the canonical dataset, pair hash, configuration, and audit outputs are documented in `reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md` and `repro_candidate_138/data/dataset/`.

---

## 5. Results

### 5.1 Comparison against the full baseline suite (natural documents)

**Table 1.** Natural-document performance on the frozen 138-pair benchmark.

| Method | Mean-fold F1 ± SD | Pooled OOF F1 | MCC | Result vs. SW-BTED Structural-Only |
|---|---:|---:|---:|---|
| **SW-BTED Structural-Only** | **0.9498 ± 0.0253** | **0.9500** | **0.9320** | — |
| SW-BTED Hybrid (α = 0.6) | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | Separate hybrid audit; not the structural-only primary result |
| TF-IDF | 0.9867 ± 0.0267 | 0.9870 | 0.9821 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |
| Standard TED (unweighted) | 0.4364 ± 0.0162 | 0.4368 | 0.0748 | SW-BTED higher; Holm-adjusted p = 2.45 × 10⁻²⁵ |
| Section Cosine | 0.6837 ± 0.0894 | 0.6667 | 0.5568 | SW-BTED higher; Holm-adjusted p = 1.36 × 10⁻⁷ |
| Genuine Flat Domain SBERT | 0.4314 ± 0.0160 | 0.4318 | 0.0000 | SW-BTED higher; Holm-adjusted p = 2.27 × 10⁻²⁸ |
| pq-Gram | 0.9479 ± 0.0478 | 0.9474 | 0.9274 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |
| Single-vector MiniLM | 0.9867 ± 0.0267 | 0.9870 | 0.9821 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |
| Single-vector BGE-small | 0.9882 ± 0.0235 | 0.9870 | 0.9821 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |
| Single-vector MPNet | 0.9882 ± 0.0235 | 0.9870 | 0.9821 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |
| Single-vector Qwen3 | 0.9867 ± 0.0267 | 0.9870 | 0.9821 | No statistically significant paired difference detected; Holm-adjusted p = 1.000 |

**Protocol note.** The TF-IDF and Section Cosine values in this table come
from the fold-local canonical 138-pair suite, which fits TF-IDF and Section
Cosine vectorizers separately on training-fold documents and selects
thresholds on training folds using a 0.005 grid. Earlier archived harnesses
reported different values (`0.4364` for TF-IDF and `0.4081` for Section
Cosine) under a different input/scope configuration; those historical values
are not silently mixed into this clean-suite table. Qwen3 was independently
re-evaluated with the same 0.005 train-fold grid and produced the reported
`0.9867 ± 0.0267` mean-fold F1 (`0.9870` pooled OOF).

All p-values shown in Table 1 are Holm-adjusted values. Raw and adjusted McNemar p-values are preserved in the machine-readable statistical audit.

SW-BTED has higher F1 than Standard TED, Section Cosine, and the Genuine Flat Domain SBERT diagnostic, with significant paired prediction differences under the stated McNemar tests. It has lower raw F1 than TF-IDF and the single-vector MiniLM, BGE-small, MPNet, and Qwen3 baselines. Its raw F1 is slightly higher than pq-Gram, but no statistically significant paired prediction difference is detected under the current test. This indicates that the method's primary value on natural documents is structural attribution and tested robustness, rather than universal accuracy superiority.

![Canonical 138-pair pooled F1 results.](../docs/submission_figures/figure_3_canonical_results.png)

**Figure 3.** Pooled F1 scores on the canonical 138-pair real-only evaluation. The figure is descriptive; inferential comparisons are reported in Table 1 and use paired predictions.

#### 5.1.1 Document-disjoint robustness

We additionally audited document dependence by grouping connected document components so that no document appears in both training and test groups. The primary pair-level result is retained as the canonical estimate because it is the frozen canonical 138-pair protocol; the grouped result is a supplemental robustness estimate.

| Protocol | Mean-fold F1 ± SD | Pooled F1 | MCC | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pair-level StratifiedKFold | 0.9498 ± 0.0253 | 0.9500 | 0.9320 | 38 | 4 | 96 | 0 |
| Document-disjoint GroupKFold | 0.9160 ± 0.0928 | 0.9157 | 0.8862 | 38 | 7 | 93 | 0 |

The decrease is material but not a collapse, and the larger fold variance motivates caution when interpreting the pair-level estimate as document-independent generalization.

### 5.2 Controlled schema-reassignment test

**Table 2.** Controlled schema-reassignment diagnostic.

**Input-identity audit.** The 20 original/perturbed pairs use byte-identical text strings on both sides; only the D2/D3 tree schema labels are swapped. Consequently, truncation cannot explain the equal text-embedding scores within this paired benchmark. This test isolates whether the method consumes explicit functional schema information; it is not presented as a fair competition between methods receiving the same information.

| Method | Accuracy | False Positive Rate |
|---|---|---|
| Single-vector SBERT | 0.0% | 100.0% (20/20) |
| **SW-BTED Structural-Only** | **100.0%** | **0.0% (0/20)** |
| SW-BTED Hybrid (α = 0.6) | 0.0% | 100.0% (20/20) |

McNemar exact test (McNemar, 1947): n10 = 20, n01 = 0, p = 1.9073 × 10⁻⁶. The original and perturbed sides receive byte-identical text strings; only the D2/D3 tree schema labels are swapped. Thus this result establishes sensitivity of SW-BTED to the tested structural change, while the evaluated text-only baseline does not observe it under the current input protocol. It does not establish a universal structural-blindness result for all embedding models or chunking strategies.

### 5.3 Observable structural perturbation audit

To complement the schema-reassignment test, we constructed 20 additional controlled negatives in which D2/D3 child content is reassigned in the tree and the serialized embedding input order is also changed. This gives text-only baselines an observable input change. We report SW-BTED threshold sensitivity separately from the MiniLM raw score distribution because 0.45 was not prespecified for this constructed set and the two score spaces are not calibrated to one another. Across cutoffs from 0.40 to 0.70, SW-BTED structural rejection ranges from 7/20 to 19/20. At the illustrative cutoff of 0.45 used in Table 3, SW-BTED rejects 17/20 pairs (85%). MiniLM cosine similarities remain high, with a range of 0.7759–1.0000; its raw distribution is reported as a diagnostic rather than converted into directly comparable rejection counts. These labels are constructed by design and are reported as a secondary diagnostic, not as natural-document classification evidence.

### 5.4 The Structural-Only / Hybrid Mode tradeoff

**Table 3.** Structural-Only and Hybrid evaluation across benchmark protocols.

| Configuration | Pair-level F1 | Document-disjoint pooled F1 | Schema reassignment | Observable perturbation |
|---|---:|---:|---:|---:|
| Structural-Only (α = 1.0) | 0.9500 | 0.9157 | **100.0%** | 85.0% at illustrative cutoff 0.45 |
| Hybrid (α = 0.6) | 1.0000 | 0.9048 | **0.0%** | Not evaluated |

The document-disjoint column uses connected-component GroupKFold and thresholds selected on training groups only. The schema-reassignment and observable-perturbation columns are different controlled benchmarks; neither should be read as natural-document accuracy. The Hybrid score is retained as a historical fixed-configuration audit, and its lower document-disjoint result shows that the pair-level perfect score does not establish document-independent superiority. Notably, the Hybrid mode's pair-level advantage reverses under document-disjoint evaluation, where Structural-Only slightly exceeds Hybrid (0.9157 vs. 0.9048). This reinforces the interpretation of α = 0.6 as a benchmark-specific operating point rather than a generally superior configuration.

### 5.5 Cross-domain transfer with adaptation: duplicate bug report detection

**Table 4.** Exploratory cross-domain transfer on GitBugs.

| Configuration | Out-of-the-box F1 | After hyperparameter adaptation |
|---|---|---|
| Structural-Only | 0.50 (degenerate) | 0.6725 |
| Hybrid (α = 0.6) | 0.9026 | **0.9141** |
| SBERT (reference) | — | 0.9074 — descriptive comparison only |

This result is consistent with, and extends, prior findings that duplicate-detection performance is sensitive to how bug-report content is structurally organized and compared (Ghadhab & Amor, 2024). It is exploratory evidence of transfer after changing the domain schema and adapting the operating configuration; because the target-domain protocol uses labeled GitBugs data and its split provenance is incomplete, we do not present it as zero-shot or confirmed held-out generalization.

### 5.6 Inspectable structural attribution case studies

![Per-domain structural similarity for the three canonical structural-attribution cases.](../docs/submission_figures/figure_4_interpretability_traces.png)

**Figure 4.** Per-domain structural similarity derived from the canonical APTED mappings. The accompanying machine-readable trace reports representative node replacements, deletions, and insertions.

We illustrate SW-BTED's inspectable structural attribution with three representative document pairs, addressing the general limitation — documented broadly in the interpretability literature (Lipton, 2016; Doshi-Velez & Kim, 2017) — that high classification accuracy alone does not provide a human-inspectable account of a model's decision:

- **Case A (`SU26SE102`–`SU26SE102_plag`, true plagiarism pair):** strong sub-tree correspondence concentrated in Technical Realization and Execution Planning. This pair is retained in the 138-pair evaluation because neither document occurs in the explicit regenerated-document exclusion set.
- **Case B (`SP26SE068`–`SU26SE063`, domain near-miss, ground-truth negative):** single-vector SBERT is a false positive due to shared domain vocabulary; SW-BTED's sub-tree decomposition correctly localizes the divergence. We present this case as an illustration, not evidence of a systemic embedding weakness — our aggregate results (Section 5.1) show no statistically significant overall disadvantage for the single-vector SBERT baseline.
- **Case C (`SP26SE122`–`SP26SE055`, disjoint negative Type_C pair):** both methods classify the pair as negative, while the trace shows broad divergence across the four domain sub-trees.

---

## 6. Discussion

### 6.1 Why structural alignment helps where it helps

The results in Sections 5.1–5.3 illustrate bounded effects: the performance gap relative to uniform TED is consistent with reduced topic conflation through explicit layer weighting; the schema-reassignment benchmark responds to a functional reorganization that leaves the tested text input unchanged; and the observable perturbation audit suggests that this sensitivity persists for many constructed cases even when serialized text input changes. These are properties of the tested protocols, not universal statements about all embedding models.

### 6.2 The Structural-Only / Hybrid tradeoff as practical guidance

Within the tested settings, Structural-Only (α = 1.0) is the more appropriate operating mode when structural/schema sensitivity is prioritized. Hybrid Mode (α = 0.6) is a historical operating point carried forward from preliminary development and fixed for the canonical audit; it was not selected by a fully nested, document-disjoint optimization. It should therefore be treated as an operating point requiring validation on deployment data rather than as a universally optimal setting.

### 6.3 Why natural-document accuracy is not the primary contribution

SW-BTED's contribution is not raw accuracy improvement over the strongest available natural-document baselines. The near-identical scores of TF-IDF and several neural baselines also suggest that this natural benchmark is comparatively easy at the aggregate classification level. Given the absence of a statistically significant paired difference under the current test, the method's demonstrated value rests on three properties not directly exposed by a single-vector baseline: a formal sufficient condition for preserving the triangle inequality, a bounded normalized structural similarity score, and inspectable structural attribution.

### 6.4 Transfer: configuration, not redesign

The bug-report result suggests that the core alignment algorithm was retained after the domain taxonomy was changed. Accordingly, the GitBugs result is treated as exploratory rather than as held-out generalization evidence because the available artifact does not preserve a complete disjoint tuning/test protocol.

---

## 7. Limitations and Threats to Validity

**No natural-document accuracy superiority is claimed.** The main claim is structural attribution and controlled robustness, not universal accuracy dominance.

**Pair-level versus document-disjoint evaluation.** The supplemental robustness result is reported in Section 5.1.1. Its lower pooled F1 and larger fold variance indicate that the pair-level estimate should not be interpreted as fully document-independent generalization.

**Label provenance.** The frozen package preserves pair labels and pair types but not the original independent annotation or pair-construction record. Consequently, conclusions are conditional on the benchmark labels as provided, and the dataset should not be interpreted as a newly independently re-annotated corpus.

**Pair-level dependence in statistical tests.** Because multiple pair observations may share documents, pair-level McNemar tests should be interpreted as conditional on the canonical pair benchmark rather than as document-independent inferential evidence.

**The domain-schema requirement is a real constraint.** Genres lacking an independent, citable functional taxonomy — of the kind formalized for requirements engineering (ISO/IEC/IEEE 29148:2018) or empirically derived for bug reports (Bettenburg et al., 2008) — are not well served by the method as designed.

**Cross-domain transfer required manual diagnosis, not automatic adaptation.** The GitBugs result is treated as exploratory rather than as held-out generalization evidence because the available artifact does not preserve a complete disjoint tuning/test protocol.

**The perturbation-benchmark ground-truth labeling reflects the authors' reasoned architectural argument**, not a claim independently verified against a specific standards clause. The observable audit additionally depends on a sensitivity analysis because its 0.45 cutoff was not prespecified.

**Evaluation spans two document genres.** The cross-domain result is useful evidence of transfer, but two genres do not establish broad domain universality; a third independently curated genre remains future work.


**Embedding input-length limits.** All single-vector embedding baselines are evaluated under model-specific truncation protocols. In the canonical audit, 159/178 documents exceed MiniLM's 256-token limit, 103/178 exceed BGE-small's 512-token limit, 124/178 exceed MPNet's 384-token limit, and 18/178 exceed Qwen3's configured 2048-token limit. These comparisons therefore characterize the stated truncation protocols rather than unrestricted long-document representations.

**Sample sizes are modest.**


---

## 8. Conclusion and Future Work

We presented SW-BTED, a schema-weighted tree-edit-distance framework for semi-structured document similarity, with inspectable structural attribution, a formal sufficient condition for preserving the triangle inequality, and a bounded normalized structural similarity score. Future work includes: evaluating a third domain (e.g., legal contracts, using CUAD's 41 expert-annotated clause types; Hendrycks et al., 2021); developing automatic cross-domain hyperparameter adaptation; comparing against current-generation embedding models under an explicit long-document protocol; and extending the structural-perturbation benchmark to multi-document constructions.

---

## References

Amirzhanov, A., Turan, C., & Makhmutova, A. (2025). Plagiarism types and detection methods: A systematic survey of algorithms in text analysis. *Frontiers in Computer Science*, 7, Article 1504725.

Augsten, N., Böhlen, M., & Gamper, J. (2005). Approximate matching of hierarchical data using pq-grams. In *Proceedings of the 31st International Conference on Very Large Data Bases (VLDB)*.

Bettenburg, N., Just, S., Schröter, A., Weiss, C., Premraj, R., & Zimmermann, T. (2008). What makes a good bug report? In *Proceedings of the 16th ACM SIGSOFT International Symposium on Foundations of Software Engineering (FSE '08)* (pp. 308–318). ACM.

Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017). SemEval-2017 Task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)* (pp. 1–14). Association for Computational Linguistics.

Das, D., Gilbert, J., Hajiaghayi, M. T., Kociumaka, T., & Saha, B. (2023). Weighted edit distance computation: Strings, trees, and Dyck. In *Proceedings of the 55th Annual ACM Symposium on Theory of Computing (STOC 2023)* (pp. 377–390). ACM.

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

Patil, A. (2025). GitBugs: Bug reports for duplicate detection, retrieval augmented generation, triage, and more. *arXiv preprint arXiv:2504.09651*.

Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit distance. *ACM Transactions on Database Systems*, 40(1), Article 3.

Project Management Institute (PMI). (2021). *A guide to the project management body of knowledge (PMBOK guide)* (7th ed.). Project Management Institute.

Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Liu, A. Y., et al. (2025). Qwen3-Embedding: Advancing text embedding and reranking through foundation models. *arXiv preprint arXiv:2506.05176*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics.

Robertson, S., & Robertson, J. (2006). *Mastering the requirements process* (2nd ed.). Addison-Wesley.

Salatino, A. A., Thanapalasingam, T., Mannocci, A., Birukou, A., Osborne, F., & Motta, E. (2020). The Computer Science Ontology: A comprehensive automatically-generated taxonomy of research areas. *Data Intelligence*, 2(3), 379–416.

Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.

Torsello, A., & Hancock, E. R. (2001). Efficiently computing weighted tree edit distance using relaxation labeling. In *Energy Minimization Methods in Computer Vision and Pattern Recognition (EMMCVPR 2001)*. Springer.

van Rijsbergen, C. J. (1979). *Information retrieval* (2nd ed.). Butterworths.

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv preprint arXiv:2309.07597*.

Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing distance between trees and related problems. *SIAM Journal on Computing*, 18(6), 1245–1262.

---
