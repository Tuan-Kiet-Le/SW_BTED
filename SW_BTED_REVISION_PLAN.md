# SW-BTED Manuscript Revision Plan

**Manuscript:** *When Does Tree Structure Matter? A Schema-Weighted Edit Distance Framework for Cross-Domain Document Similarity*
**Goal:** Nâng bản thảo lên mức đủ mạnh để nhắm tới **journal/conference Q2 trở lên**, đồng thời giữ khả năng thử venue Q1 nếu kết quả sau revision đủ tốt.

---

## 0. Nguyên tắc sửa paper

Không nên mở rộng dataset hoặc thêm quá nhiều experiment trước khi xử lý các vấn đề nền tảng.

**Thứ tự ưu tiên:**

1. Sửa theoretical claim.
2. Sửa evaluation protocol.
3. Sửa baseline implementation.
4. Làm method reproducible.
5. Chạy lại canonical experiments.
6. Thêm ablation + hard structural benchmark.
7. Củng cố cross-domain transfer.
8. Nâng interpretability từ case study thành evidence định lượng.
9. Cuối cùng mới polish writing, figures và venue positioning.

---

# 1. P0 — Bắt buộc sửa trước khi submit

## P0.1. Sửa claim về “metric preservation”

### Vấn đề hiện tại

Section 3.4 chứng minh rằng:

\[
d_\beta(u,v)=\beta_\ell Dist_{content}(u,v)+(1-\beta_\ell)Dist_{schema}(u,v)
\]

là metric nếu hai component distance đều là metric.

Tuy nhiên actual replacement cost lại được định nghĩa:

\[
w_{rep}^{(\ell)}(u,v)=\left(w_{del}^{(\ell)}(u)+w_{ins}^{(\ell)}(v)\right)d_\beta(u,v)
\]

Proof hiện tại chưa chứng minh rằng **toàn bộ replacement cost** vẫn thỏa symmetry và triangle inequality. Ngoài ra manuscript chưa định nghĩa chính xác `Dist_content`, `Dist_schema`, `w_del`, `w_ins`.

### Cần làm

- [ ] Viết lại theorem để theorem chỉ claim đúng phần đã chứng minh.
- [ ] Tách rõ:
  - node-level combined distance,
  - edit-operation cost,
  - tree-edit objective.
- [ ] Không viết rằng metric property là điều kiện bắt buộc để APTED tính exact TED nếu chưa có chứng minh/citation trực tiếp.
- [ ] Kiểm tra `Dist_content` thực tế có phải metric hay không.
- [ ] Nếu dùng cosine-derived distance, xác minh mathematical property trước khi gọi nó là metric.
- [ ] Đồng bộ wording trong Abstract, Introduction, Contributions, Discussion và Conclusion.

### Hướng sửa an toàn

**Option A — Thu hẹp theorem:** chứng minh convex combination của hai metric là metric, rồi nói rõ proposition chỉ áp dụng cho **node-level content/schema distance**, không tự động áp dụng cho toàn bộ edit-cost function.

**Option B — Thiết kế lại edit cost:** chỉ nên chọn nếu thật sự cần giữ claim mạnh “metric-preserving framework”.

### Definition of Done

- [ ] Không còn logical gap giữa theorem và cost function thực tế.
- [ ] Reviewer biết chính xác cái gì được chứng minh và cái gì không.
- [ ] Không có claim mạnh hơn proof.

---

## P0.2. Làm Section 3 đủ để reproduce

### CapTree construction

- [ ] Quy tắc parse document.
- [ ] Cách nhận diện T2 Domain.
- [ ] Cách sinh T3 Intent.
- [ ] Một Intent node đại diện cho sentence, paragraph, requirement hay semantic unit nào.
- [ ] Cách xử lý missing sections.
- [ ] Cách xử lý duplicated/nested sections.

### T4 Terminology

- [ ] Keyword extraction algorithm.
- [ ] Lemmatization procedure.
- [ ] Computer Science Ontology lookup.
- [ ] Technology Equivalence Map:
  - được tạo bằng tay hay tự động,
  - số lượng mapping,
  - rule thêm mapping,
  - có dùng label/test knowledge hay không.

### Distances

- [ ] Công thức chính xác `Dist_content`.
- [ ] Công thức chính xác `Dist_schema`.
- [ ] Range và normalization.
- [ ] Embedding model dùng tại từng layer.

### TED costs

- [ ] `w_del`.
- [ ] `w_ins`.
- [ ] `w_rep`.
- [ ] Cost theo từng layer.
- [ ] Root handling.
- [ ] Normalization từ raw TED thành similarity.

### Hyperparameter table

| Parameter | Capstone | GitBugs | Search range | Tuned on |
|---|---:|---:|---|---|
| β_T2 | TBD | TBD | TBD | TBD |
| β_T3 | TBD | TBD | TBD | TBD |
| β_T4 | TBD | TBD | TBD | TBD |
| α | TBD | TBD | TBD | TBD |
| Decision threshold | fold-specific | fold-specific | 0–1 | train folds |

### Nên thêm pseudocode

- [ ] Algorithm 1: `BuildCapTree(document, schema)`
- [ ] Algorithm 2: `SWBTED(tree_A, tree_B, beta)`
- [ ] Algorithm 3: threshold tuning/evaluation protocol nếu cần.

### Definition of Done

Một người khác chỉ đọc manuscript + supplement có thể implement lại SW-BTED mà không cần hỏi tác giả.

---

## P0.3. Audit cross-validation để tránh document leakage

### Vấn đề cần kiểm tra

Dataset là **document pairs**, nhưng manuscript mới nói “5-fold stratified cross-validation”. Chưa rõ split theo pair hay theo source document.

Nếu có:

- train pair `(A, B)`
- test pair `(A, C)`

thì train/test không hoàn toàn độc lập ở document level.

### Cần làm

- [ ] Tạo graph: node = document, edge = evaluated pair.
- [ ] Đếm số document xuất hiện trong nhiều pair.
- [ ] Kiểm tra overlap train/test trong protocol hiện tại.
- [ ] Lưu overlap audit.

### Protocol ưu tiên

**Preferred:** connected-component grouped cross-validation.

Không để hai pairs thuộc cùng connected document component rơi vào train và test khác nhau.

Nếu dataset quá nhỏ, cân nhắc `GroupKFold`/`StratifiedGroupKFold` theo document family.

### Chạy lại toàn bộ

- [ ] SW-BTED.
- [ ] TF-IDF.
- [ ] Standard TED.
- [ ] Section Cosine.
- [ ] SBERT.
- [ ] BGE.
- [ ] MPNet.
- [ ] Qwen3.
- [ ] pq-Gram.

### Nếu muốn giữ kết quả cũ

| Protocol | Purpose |
|---|---|
| Historical pair-level CV | comparability với result cũ |
| Grouped CV | leakage-resistant main result |

**Main result nên dùng grouped protocol.**

### Definition of Done

- [ ] Không có uncontrolled document overlap giữa train/test.
- [ ] Tất cả baselines dùng cùng split IDs.
- [ ] Lưu pair-level OOF predictions.

---

## P0.4. Sửa “full-document embedding” baseline

### Cần audit từng model

- [ ] tokenizer max length.
- [ ] số token mỗi document.
- [ ] % documents bị truncate.
- [ ] actual input sau truncation.
- [ ] pooling method.

### Baseline mới nên có

#### Baseline A — Chunked embedding

1. Chia document thành chunks.
2. Encode từng chunk.
3. Mean/weighted pooling.
4. Cosine giữa pooled representations.

#### Baseline B — Schema-matched embedding

\[
sim(A,B)=\sum_d \lambda_d\cos(E(A_d),E(B_d))
\]

Trong đó `d` tương ứng D1–D4.

Baseline này trả lời câu hỏi:

> lợi ích đến từ TED hay chỉ từ việc chia document đúng schema?

#### Baseline C — Long-context modern embedding

Giữ Qwen3 hoặc một current-generation embedding model có context phù hợp và preprocessing minh bạch.

### Cần sửa wording

Tránh absolute claim kiểu:

> flat embeddings are structurally blind by construction

Nên chuyển thành empirical claim:

> the evaluated single-vector baselines showed limited sensitivity to the tested cross-domain structural perturbations.

### Definition of Done

- [ ] Không baseline nào bị silent truncation.
- [ ] Preprocessing được document rõ.
- [ ] Có ít nhất một competitive schema-aware nhưng non-TED baseline.

---

# 2. P1 — Experiments có tác động lớn đến khả năng accept

## P1.1. Mở rộng structural perturbation benchmark

### Vấn đề hiện tại

Benchmark mới có 20 negative pairs và chủ yếu dùng D2↔D3 swap, quá sát với inductive bias của SW-BTED.

### Taxonomy perturbation đề xuất

#### S1 — Cross-domain swap

- D1 ↔ D2
- D2 ↔ D3
- D3 ↔ D4

#### S2 — Partial migration

Chuyển một phần content sang sai functional domain.

#### S3 — Section deletion

Xóa một domain quan trọng.

#### S4 — Duplicate insertion

Lặp section/requirement ở sai vị trí.

#### S5 — Within-domain reorder

Đảo thứ tự content nhưng vẫn nằm đúng domain.

**Expected:** SW-BTED không nên phạt quá mạnh nếu functional placement vẫn đúng.

#### S6 — Structure-preserving paraphrase

Giữ schema, paraphrase mạnh.

#### S7 — Structure-preserving semantic corruption

Giữ heading/schema nhưng đổi nội dung substantive.

#### S8 — Terminology substitution

Đổi equivalent và non-equivalent technology terms.

### Kích thước mục tiêu

- [ ] 100–300 perturbation pairs trở lên nếu nguồn lực cho phép.
- [ ] Không cần mọi case được tạo thủ công.

### Phải có positive controls

Ví dụ:

- paraphrase same structure → positive,
- synonym replacement → positive,
- benign within-domain reorder → positive.

### Metrics

- Accuracy.
- Precision.
- Recall.
- F1.
- MCC.
- Per-perturbation accuracy.

### Definition of Done

Có thể trả lời rõ:

> SW-BTED nhạy với loại structural corruption nào và invariant với thay đổi nào?

---

## P1.2. Thêm schema-matched embedding baseline

### Baseline

\[
sim_{schema}=\sum_{d=1}^{4}\lambda_d\cdot cosine(E(D_d^A),E(D_d^B))
\]

Thử:

- uniform `λ_d`,
- learned `λ_d`,
- same weights as SW-BTED nếu hợp lý.

### Models

- [ ] BGE.
- [ ] Qwen3 embedding.

### Research question

> Does tree edit alignment add value beyond schema-based section decomposition?

### Definition of Done

Nếu SW-BTED không thắng về natural F1 vẫn có thể giữ contribution nếu:

- mạnh hơn trên structural perturbations,
- có edit trace rõ,
- robustness tốt hơn.

---

## P1.3. Ablation study

### Ablations nên có

- [ ] Full SW-BTED.
- [ ] Uniform TED.
- [ ] No schema weighting.
- [ ] No T4 terminology.
- [ ] No ontology normalization.
- [ ] T2 only.
- [ ] T2 + T3.
- [ ] T2 + T3 + T4.
- [ ] Remove schema distance.
- [ ] Remove content distance.

### Hyperparameter sensitivity

Plot:

- [ ] F1 vs `β_T2`.
- [ ] F1 vs `β_T3`.
- [ ] F1 vs `β_T4`.
- [ ] Natural F1 vs `α`.
- [ ] Perturbation accuracy vs `α`.

### Output mong muốn

Một figure thể hiện rõ:

> semantic classification performance ↔ structural sensitivity

---

## P1.4. Củng cố GitBugs cross-domain evaluation

### Vấn đề hiện tại

“After hyperparameter adaptation” chưa rõ:

- parameter nào được tune,
- dùng bao nhiêu target-domain labels,
- tune trên data nào,
- test ở đâu.

### Nếu dùng target labels

Không gọi là pure cross-domain generalization.

Dùng một trong các term:

- cross-domain adaptation,
- few-shot adaptation,
- target-domain calibration.

### Protocol mạnh hơn

#### Option A — Project holdout

Tune trên projects A/B/C, test trên project D.

#### Option B — Few-shot adaptation

So sánh:

- 0 labels,
- 20 labels,
- 50 labels,
- 100 labels.

#### Option C — Zero-shot schema transfer

Dùng taxonomy mới nhưng không dùng target labels để tune.

### Main table gợi ý

| Method | Zero-shot | Adapted | Held-out Project |
|---|---:|---:|---:|
| SW-BTED Struct | | | |
| SW-BTED Hybrid | | | |
| Schema embedding | | | |
| Qwen3 | | | |

### Definition of Done

Paper phân biệt rõ:

- algorithm transfer,
- schema transfer,
- hyperparameter adaptation,
- label-free generalization.

---

## P1.5. Nâng interpretability evaluation

### Hiện tại

3 case studies mới chứng minh hệ thống có thể **xuất trace**, chưa chứng minh trace đúng/hữu ích theo human judgment.

### Low-cost evaluation

Cho 2–3 annotators đánh dấu:

> domain nào là nguyên nhân chính khiến pair khác nhau?

So với:

- human top-1 domain,
- SW-BTED lowest-similarity / highest-edit-cost domain.

### Metrics

- Top-1 agreement.
- Top-2 recall.
- Cohen's kappa / Fleiss' kappa.
- Spearman correlation nếu annotators rank domain severity.

### Nếu chưa làm human study

Hạ claim từ:

> demonstrated genuine structural interpretability

xuống:

> provides inspectable structural attribution traces

---

# 3. P2 — Bổ sung giúp paper mạnh hơn

## P2.1. Domain thứ ba

### Candidate

**Legal contracts / CUAD** là lựa chọn hợp lý vì có taxonomy và genre khác đáng kể so với proposal/bug report.

### Mục tiêu

- [ ] 100–300 pairs hoặc proof-of-transfer có kiểm soát.
- [ ] taxonomy mapping rõ.
- [ ] baseline Qwen/BGE.
- [ ] zero-shot + adapted SW-BTED.

### Title rule

Chỉ giữ chữ **Cross-Domain** nếu evidence transfer đủ mạnh. Nếu không, đổi title về semi-structured document similarity.

---

## P2.2. End-to-end runtime

Hiện runtime chủ yếu phản ánh alignment/scoring.

Nên đo:

| Component | Mean ms | P95 ms |
|---|---:|---:|
| Parsing | | |
| Embedding | | |
| Tree construction | | |
| APTED | | |
| Total | | |

- [ ] Thêm scalability curve: number of tree nodes vs runtime.

---

## P2.3. Error analysis

Lấy ít nhất:

- 10 false positives,
- 10 false negatives

cho SW-BTED và strongest baseline.

Phân nhóm:

- shared domain vocabulary,
- missing section,
- parser error,
- taxonomy mismatch,
- terminology normalization error,
- semantic paraphrase,
- structural mismatch,
- threshold issue.

---

# 4. Sửa từng section trong manuscript

## Abstract

- [ ] Giữ practical problem + parity honesty.
- [ ] Không claim “metric-preserving” cho toàn framework nếu theorem chỉ áp dụng node-level distance.
- [ ] Không foreground `0.9498 vs 0.4314` như strongest evidence.
- [ ] Nhấn mạnh: competitive natural performance + structural sensitivity + inspectable attribution.
- [ ] Cập nhật toàn bộ số sau khi rerun grouped CV.

## Section 1 — Introduction

- [ ] Hạ absolute claim về embedding structural blindness.
- [ ] Không lấy một baseline yếu làm headline contribution.
- [ ] Viết contributions theo:
  1. schema-grounded representation,
  2. schema/content weighted alignment,
  3. document-disjoint evaluation,
  4. perturbation benchmark,
  5. transfer/adaptation study,
  6. inspectable structural attribution.

## Section 2 — Related Work

Bổ sung:

- [ ] Long-document embeddings.
- [ ] Hierarchical document embeddings.
- [ ] Section-aware similarity/retrieval.
- [ ] Structure-aware document models.
- [ ] Explainable similarity/alignment.

Phải trả lời rõ SW-BTED khác gì với:

1. hierarchical embeddings,
2. section-wise cosine,
3. tree kernels,
4. neural structure-aware retrieval,
5. conventional weighted TED.

## Section 3 — Method

- [ ] Complete mathematical definitions.
- [ ] Construction rules.
- [ ] Pseudocode.
- [ ] Hyperparameter table.
- [ ] Normalization.
- [ ] Complexity.
- [ ] Worked example.

## Section 4 — Experimental Setup

Bổ sung:

- [ ] number of unique documents,
- [ ] positive/negative ratio,
- [ ] average/median length,
- [ ] average CapTree nodes,
- [ ] pair-generation procedure,
- [ ] exact split rule,
- [ ] random seed,
- [ ] fold IDs,
- [ ] overlap audit,
- [ ] hyperparameter search space,
- [ ] target-domain adaptation protocol,
- [ ] repository link.

## Section 5 — Results

### Reporting format thống nhất

Nên dùng:

- Mean-fold F1 ± SD.
- Pooled OOF F1.
- Precision.
- Recall.
- MCC.
- PR-AUC nếu imbalance đáng kể.

Không trộn range như `0.9744–0.9867` với `mean ± SD` trong cùng table.

### Tables/Figures mới

- [ ] Natural benchmark.
- [ ] Expanded perturbation benchmark.
- [ ] Ablation table.
- [ ] α tradeoff curve.
- [ ] β sensitivity.
- [ ] Cross-domain transfer table.
- [ ] Attribution agreement nếu có.

## Section 6 — Discussion

Phải trả lời trực tiếp:

### When does tree structure matter?

- structural/schema compliance,
- functional section placement,
- localization requirements.

### When does it not matter?

- easy lexical/semantic pairs,
- cases where flat embeddings already separate data well,
- purely semantic duplication where hybrid/embedding may be preferable.

**Central message nên là:**

> tree structure matters under specific structural failure modes and interpretability requirements, not universally.

## Section 7 — Limitations

Thêm:

- [ ] schema construction cost.
- [ ] taxonomy dependence.
- [ ] manual equivalence-map dependence.
- [ ] parser error propagation.
- [ ] synthetic perturbations may not represent all real misuse.
- [ ] edit traces are inspectable attributions, not automatically causal explanations.
- [ ] target-domain adaptation cost.

## Section 8 — Conclusion

Claim nên tập trung vào:

- competitive rather than universally superior accuracy,
- stronger sensitivity to functional structural changes,
- inspectable alignment,
- applicability when meaningful domain taxonomy exists.

---

# 5. Sửa schema-grounding principle

### Vấn đề

Paper nói schema **must** derive từ independent expert-authored taxonomy, nhưng D3 `Technical Realization` lại là author extension.

### Wording an toàn hơn

> Domain schemas should be grounded primarily in independently established functional taxonomies. Any extensions or deviations must be explicitly documented and justified.

### Hoặc

- [ ] Tìm additional external grounding riêng cho D3.

### Definition of Done

Principle không tự mâu thuẫn với implementation của capstone schema.

---

# 6. Dataset audit checklist

- [ ] Count unique documents.
- [ ] Count pairs.
- [ ] Positive/negative ratio.
- [ ] Pair type distribution.
- [ ] Documents appearing in multiple pairs.
- [ ] Connected components in pair graph.
- [ ] Duplicate files.
- [ ] Near-duplicate filenames.
- [ ] Synthetic/regenerated document exclusion.
- [ ] Label provenance.
- [ ] Pair-generation rule.
- [ ] Token-length distribution.
- [ ] Missing functional domains.
- [ ] Tree-node distribution.
- [ ] Technology-map coverage.
- [ ] Parser failure rate.

Nên lưu audit thành machine-readable artifact, ví dụ `dataset_audit.json`.

---

# 7. Reproducibility package

```text
sw-bted/
├── README.md
├── requirements.txt
├── configs/
│   ├── capstone.yaml
│   └── gitbugs.yaml
├── data/
│   ├── README.md
│   ├── pair_ids.csv
│   └── fold_assignments.csv
├── src/
│   ├── parsing/
│   ├── captree/
│   ├── distances/
│   ├── ted/
│   └── evaluation/
├── scripts/
│   ├── build_trees.py
│   ├── run_baselines.py
│   ├── run_sw_bted.py
│   ├── run_ablation.py
│   └── run_statistics.py
├── outputs/
│   ├── pair_predictions/
│   ├── metrics/
│   └── traces/
└── supplementary/
    ├── technology_equivalence_map.*
    ├── perturbation_protocol.md
    └── annotation_guideline.md
```

---

# 8. Statistical analysis plan

- [ ] Dùng cùng folds cho tất cả methods.
- [ ] Store OOF prediction cho từng pair.
- [ ] Paired test chỉ trên identical test observations.
- [ ] Multiple-comparison correction.
- [ ] Kiểm tra document dependence trước khi diễn giải p-value.
- [ ] Report Precision, Recall, F1, MCC, PR-AUC khi phù hợp.
- [ ] Nếu bootstrap, ưu tiên resampling ở document/group level thay vì pair level nếu dependency tồn tại.

---

# 9. Revised Research Questions

## RQ1 — Natural-document discrimination

> How accurately does SW-BTED distinguish similar and dissimilar natural semi-structured documents compared with semantic, lexical, and structural baselines?

## RQ2 — Structural sensitivity

> How sensitive is SW-BTED to functional structural perturbations that preserve substantial lexical and semantic content?

## RQ3 — Component contribution

> Which components of schema weighting, hierarchy, and terminology normalization contribute to performance?

## RQ4 — Cross-domain transfer

> How effectively can SW-BTED transfer to a structurally distinct document genre under zero-shot and limited adaptation settings?

## RQ5 — Structural attribution

> To what extent do SW-BTED edit traces localize the document regions responsible for a similarity judgment?

---

# 10. Proposed main results structure

## Table 1 — Natural benchmark

Competitive methods + grouped CV.

## Table 2 — Structural perturbations

Per perturbation type.

## Table 3 — Ablation

T2/T3/T4/schema/content/ontology.

## Figure 1 — CapTree

Giữ.

## Figure 2 — Pipeline

Giữ.

## Figure 3 — Natural benchmark

Cập nhật theo protocol mới.

## Figure 4 — α tradeoff curve

Natural F1 vs perturbation sensitivity.

## Figure 5 — β sensitivity

Per-layer hyperparameter.

## Figure 6 — Interpretability trace

Case study.

## Table 4 — Cross-domain transfer

Zero-shot vs adapted vs held-out.

## Table 5 — Attribution agreement

Nếu có human annotation.

---

# 11. Execution order

## Phase A — Correctness first

- [ ] Audit theorem.
- [ ] Define distances.
- [ ] Define edit costs.
- [ ] Fix claims toàn manuscript.
- [ ] Audit dataset graph.
- [ ] Build document-disjoint folds.
- [ ] Audit embedding truncation.

**Không chuyển sang Phase B trước khi hoàn thành Phase A.**

## Phase B — Canonical rerun

- [ ] Freeze fold IDs.
- [ ] Re-run TF-IDF.
- [ ] Re-run Standard TED.
- [ ] Re-run Section Cosine.
- [ ] Re-run SBERT/chunked SBERT.
- [ ] Re-run BGE.
- [ ] Re-run MPNet.
- [ ] Re-run Qwen3.
- [ ] Re-run pq-Gram.
- [ ] Re-run SW-BTED.
- [ ] Save pair-level OOF predictions.

## Phase C — Critical new experiments

- [ ] Schema-matched embedding baseline.
- [ ] Ablation.
- [ ] β sensitivity.
- [ ] α sensitivity.
- [ ] Expanded perturbation benchmark.
- [ ] Error analysis.

## Phase D — Generalization

- [ ] Rebuild GitBugs protocol.
- [ ] Zero-shot evaluation.
- [ ] Adapted evaluation.
- [ ] Held-out project evaluation.
- [ ] Optional third domain.

## Phase E — Interpretability

- [ ] Generate structured edit traces.
- [ ] Prepare annotation guideline.
- [ ] Human domain-localization study.
- [ ] Agreement metrics.

## Phase F — Final manuscript rewrite

- [ ] Rewrite Abstract.
- [ ] Rewrite Contributions.
- [ ] Rewrite Method.
- [ ] Rewrite Experimental Setup.
- [ ] Replace all old numbers.
- [ ] Rebuild tables/figures.
- [ ] Update Discussion.
- [ ] Update Limitations.
- [ ] Complete repository link.
- [ ] Final citation audit.
- [ ] Final consistency audit.

---

# 12. Stop conditions trước submission

Không submit nếu còn một trong các điểm sau:

- [ ] Theorem claim mạnh hơn proof.
- [ ] Train/test còn uncontrolled document overlap.
- [ ] Baseline bị silent truncation.
- [ ] Main algorithm chưa reproduce được từ paper.
- [ ] Hyperparameter adaptation dùng test labels.
- [ ] Table dùng incompatible reporting conventions.
- [ ] Repository còn TODO.
- [ ] “Cross-domain” claim mạnh hơn experiment.
- [ ] Interpretability claim mạnh hơn evidence.
- [ ] Structural benchmark chỉ có một loại hand-crafted swap.

---

# 13. Minimum viable Q2 revision

## Must-have

- [ ] Fix theorem.
- [ ] Complete method definitions.
- [ ] Grouped CV.
- [ ] Fix embedding truncation.
- [ ] Add schema-matched embedding baseline.
- [ ] Run core ablations.
- [ ] Expand perturbation benchmark thành nhiều transformation types.
- [ ] Clarify GitBugs adaptation.
- [ ] Complete reproducibility link.
- [ ] Rewrite claims.

## Có thể để future work

- [ ] Third domain.
- [ ] Large human interpretability study.
- [ ] Extensive scalability experiment.
- [ ] Automatic schema induction.

---

# 14. Stronger Q1-attempt revision

Nếu muốn thử Q1 trước:

- [ ] Tất cả Minimum Q2 items.
- [ ] 100–300+ structural perturbation pairs.
- [ ] Held-out GitBugs project evaluation.
- [ ] Third domain hoặc stronger multi-project transfer.
- [ ] Quantitative interpretability study.
- [ ] End-to-end runtime/scalability.
- [ ] Detailed error analysis.
- [ ] Full public reproducibility package.

---

# 15. Suggested final positioning

Paper không nên được bán như:

> “SW-BTED is more accurate than embeddings.”

Vì natural-document results hiện không support claim đó.

Positioning mạnh hơn:

> **SW-BTED is a schema-aware structural similarity framework designed for settings where functional organization and inspectable alignment matter. It remains competitive with strong semantic baselines on natural documents while exposing structural divergence that single-vector comparisons do not explicitly localize.**

Central message nên trả lời đúng title:

> **Tree structure matters most when similarity depends not only on what content is present, but also on where functionally distinct content belongs.**

---

# 16. Final revision checklist

## Theory

- [ ] Theorem valid.
- [ ] Assumptions explicit.
- [ ] Claims aligned with theorem.

## Method

- [ ] Fully defined.
- [ ] Pseudocode included.
- [ ] Hyperparameters documented.

## Data

- [ ] Dataset provenance clear.
- [ ] No uncontrolled train/test document overlap.
- [ ] Pair labels auditable.

## Baselines

- [ ] No silent truncation.
- [ ] Same splits.
- [ ] Same tuning rules.
- [ ] Schema-matched baseline included.
- [ ] Current-generation embedding included.

## Experiments

- [ ] Natural benchmark.
- [ ] Perturbation benchmark.
- [ ] Ablation.
- [ ] Cross-domain transfer.
- [ ] Error analysis.

## Interpretability

- [ ] Trace examples.
- [ ] Claim calibrated to evidence.
- [ ] Quantitative evaluation if feasible.

## Statistics

- [ ] Consistent metric reporting.
- [ ] Paired tests.
- [ ] Multiple-comparison correction.
- [ ] Confidence intervals where appropriate.

## Writing

- [ ] Abstract updated.
- [ ] Contributions updated.
- [ ] No universal superiority claim.
- [ ] “By construction” claims checked.
- [ ] Limitations complete.

## Reproducibility

- [ ] Repository available.
- [ ] Configs available.
- [ ] Fold IDs available.
- [ ] Pair predictions available.
- [ ] Perturbation generation documented.

---

## Recommended immediate next task

Làm theo đúng thứ tự này:

1. Audit theorem trong Section 3.4.
2. Audit pair graph và cross-validation leakage.
3. Audit tokenizer truncation cho mọi embedding baseline.
4. Freeze corrected evaluation protocol.
5. Sau đó mới rerun experiments.

Nếu ba audit đầu tiên làm thay đổi kết quả đáng kể, hãy coi các số hiện tại trong manuscript là **historical results**, không tiếp tục xây conclusion mới dựa trên chúng.
