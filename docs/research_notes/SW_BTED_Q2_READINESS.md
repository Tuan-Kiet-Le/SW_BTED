# SW-BTED: Q2 Readiness Implementation Spec
# Adaptive T5 + Interpretability Case Study + New Baselines
> **Mục đích:** 3 tasks cần implement để paper đạt chuẩn Q2.
> Thực hiện theo đúng thứ tự TASK 1 → TASK 2 → TASK 3.
> Dừng và báo cáo sau mỗi task. KHÔNG gộp vào một lần chạy.

---

## TASK 1 — ADAPTIVE T5 ACTIVATION

### Bối cảnh

Ablation Group A cho thấy T5 (Semantic Role) không giúp ích trên FPT
và làm giảm F1 = −0.1044 trên PURE (p = 5.42×10⁻⁷).
Nguyên nhân: T5 gây parsing noise trên câu ngắn/đơn giản.
Fix: Chỉ kích hoạt T5 khi câu đủ phức tạp theo heuristic.

### 1.1 Heuristic Logic

```python
def should_activate_t5(sentence: str) -> bool:
    """
    Quyết định có chạy Semantic Role Extraction (T5) không.
    Trả về True nếu câu đủ phức tạp để T5 có ích.
    """
    tokens = sentence.split()
    token_count = len(tokens)

    # Đếm dấu hiệu cú pháp phức tạp
    clause_indicators = (
        sentence.count(',') +
        sentence.count(';') +
        sentence.lower().count(' and ') +
        sentence.lower().count(' which ') +
        sentence.lower().count(' that ') +
        sentence.lower().count(' when ') +
        sentence.lower().count(' where ') +
        sentence.lower().count(' to ') +
        sentence.lower().count(' by ') +
        sentence.lower().count(' using ')
    )

    # Điều kiện kích hoạt T5:
    # (1) Câu đủ dài (> 15 tokens) VÀ
    # (2) Có ít nhất 1 dấu hiệu cú pháp phức tạp
    return token_count > 15 and clause_indicators >= 1


def build_t4_node(sentence: str, feature_label: str):
    """
    Build T4 node với quyết định adaptive về T5.
    """
    node = AtomicRequirementNode(
        text=sentence,
        feature_label=feature_label
    )

    if should_activate_t5(sentence):
        # Chạy spaCy dependency parsing → tạo T5 nodes → T6 leaves
        roles = extract_semantic_roles(sentence)
        node.children = build_t5_nodes(roles)
        node.t5_activated = True
    else:
        # Skip T5 → T4 connect trực tiếp xuống T6 leaf keywords
        keywords = extract_keywords_direct(sentence)
        node.children = build_t6_leaves(keywords)
        node.t5_activated = False

    return node
```

### 1.2 Logging bắt buộc

Với mỗi document được parse, ghi lại:
```python
activation_log = {
    "document_id": str,
    "total_atomic_reqs": int,
    "t5_activated_count": int,
    "t5_skipped_count": int,
    "t5_activation_rate": float,  # = activated / total
}
# Lưu ra: diagnostics/t5_activation_rates.json
```

**Kỳ vọng:**
- FPT dataset: activation rate ≈ 80–90% (câu dài, phức tạp)
- PURE dataset: activation rate ≈ 10–30% (câu ngắn, đơn giản)

Nếu activation rate của FPT < 60% hoặc PURE > 50%
→ **DỪNG LẠI, báo cáo**, điều chỉnh ngưỡng threshold
   (thử `token_count > 12` hoặc `clause_indicators >= 2`).

### 1.3 Re-evaluation sau Adaptive T5

Chạy lại **toàn bộ** evaluation protocol (5-fold CV, proper split)
với cấu hình adaptive T5 trên cả 2 dataset:

```python
EVAL_CONFIG = {
    "datasets": ["FPT", "PURE_adapted"],
    "alpha": 0.6,          # giữ nguyên từ trước
    "beta_config": "per_layer",
    "t5_mode": "adaptive", # ← thay đổi duy nhất
    "folds": 5,
    "metrics": ["precision", "recall", "f1", "roc_auc",
                "type_a_tpr", "type_b_tnr", "type_c_tnr"],
}
```

So sánh với A1 (T5 always on) và A2 (T5 always off):

```
Target table để báo cáo:

Variant              | FPT F1  | PURE F1 | Ghi chú
---------------------|---------|---------|--------
A1: T5 always ON     | 0.9707  | 0.7612  | Baseline trước
A2: T5 always OFF    | 0.9707  | 0.8656  | Best PURE
A_new: T5 ADAPTIVE   | ≥0.9707 | ≥0.8300 | Target: tốt hơn A1 trên PURE
                                           mà không làm tệ FPT
```

**Điều kiện thành công của Task 1:**
```
✅ FPT F1 ≥ 0.9707 (không tệ hơn baseline)
✅ PURE F1 ≥ 0.82  (cải thiện đáng kể so với 0.7612)
✅ McNemar Adaptive vs A1 trên PURE: p < 0.01
```

Nếu không đạt → thử điều chỉnh ngưỡng heuristic, báo cáo kết quả
của ít nhất 3 ngưỡng khác nhau để người dùng chọn.

### 1.4 Output Task 1

```
results/adaptive_t5/
├── t5_activation_rates.json        # activation rate theo dataset
├── adaptive_t5_FPT_results.json    # metrics trên FPT
├── adaptive_t5_PURE_results.json   # metrics trên PURE
├── adaptive_vs_variants_table.csv  # so sánh A1, A2, A_new
└── mcnemar_adaptive_vs_A1.csv      # significance test
```

**DỪNG LẠI sau Task 1 — báo cáo kết quả trước khi làm Task 2.**

---

## TASK 2 — INTERPRETABILITY CASE STUDY

### Bối cảnh

SW-BTED không beat B1 (TF-IDF) và B2 (SBERT) về F1.
Để trả lời câu hỏi "So what?" của reviewer Q2, cần chứng minh
SW-BTED cung cấp **structural explanation** mà embedding methods không có.

Đây là experiment định tính — không cần user study lớn.
Chỉ cần 10–15 cặp được phân tích kỹ, trình bày rõ ràng.

### 2.1 Chọn cặp để phân tích

Chọn **15 cặp** từ FPT test set theo 3 nhóm:

```python
CASE_STUDY_GROUPS = {
    # Nhóm 1: SW-BTED đúng, SBERT cũng đúng
    # → Dùng để show SW-BTED explanation rõ hơn SBERT score
    "both_correct": 5,

    # Nhóm 2: SW-BTED đúng, SBERT SAI (False Positive/Negative)
    # → Dùng để show SW-BTED phát hiện được gì mà SBERT bỏ qua
    # → ĐÂY LÀ NHÓM QUAN TRỌNG NHẤT
    "sw_correct_sbert_wrong": 5,

    # Nhóm 3: SBERT đúng, SW-BTED SAI
    # → Dùng để trình bày limitation trung thực
    "sbert_correct_sw_wrong": 5,
}

# Cách chọn nhóm 2 (quan trọng nhất):
# Lấy prediction của B2 (SBERT) và SW-BTED trên toàn test set
# Tìm các cặp mà:
#   sbert_pred != ground_truth AND swbted_pred == ground_truth
```

> ⚠️ Nếu không tìm được đủ 5 cặp nhóm 2 →
> **DỪNG LẠI, báo cáo số lượng thực tế.**
> Đây là thông tin quan trọng: nếu SW-BTED và SBERT sai giống nhau
> trên mọi case → explanation advantage không thực sự tồn tại.

### 2.2 Với mỗi cặp, tạo "Explanation Report"

```python
def generate_explanation_report(doc_A, doc_B, ground_truth):
    report = {}

    # ── SBERT Explanation (đơn giản, không có structure) ──
    sbert_score = cosine_similarity(embed(doc_A), embed(doc_B))
    report["sbert"] = {
        "score": round(sbert_score, 4),
        "prediction": "SIMILAR" if sbert_score >= 0.51 else "DIFFERENT",
        "explanation": f"Global cosine similarity = {sbert_score:.4f}",
        # ← Đây là tất cả những gì SBERT có thể nói
    }

    # ── SW-BTED Explanation (có structure) ──
    tree_A = build_tree(doc_A)
    tree_B = build_tree(doc_B)
    sim_score, edit_ops = compute_sw_bted_with_trace(tree_A, tree_B)

    # Tổng hợp matching theo domain
    domain_matches = {}
    for op in edit_ops:
        if op.type == "match":
            domain = op.node_A.domain
            if domain not in domain_matches:
                domain_matches[domain] = []
            domain_matches[domain].append({
                "layer": op.layer,
                "node_A": op.node_A.label,
                "node_B": op.node_B.label,
                "cost": op.cost,
            })

    # Top matched semantic roles (T5 level)
    role_matches = [
        op for op in edit_ops
        if op.layer == "T5" and op.type == "match"
    ]

    report["sw_bted"] = {
        "score": round(sim_score, 4),
        "prediction": "SIMILAR" if sim_score >= 0.35 else "DIFFERENT",
        "domain_matching_summary": domain_matches,
        "top_matched_roles": [
            {
                "domain": r.node_A.domain,
                "role_type": r.node_A.role,
                "A_value": r.node_A.label,
                "B_value": r.node_B.label,
                "normalized_A": r.node_A.canonical,
                "normalized_B": r.node_B.canonical,
            }
            for r in role_matches[:10]  # top 10
        ],
        "explanation_narrative": generate_narrative(domain_matches, role_matches),
    }

    report["ground_truth"] = ground_truth
    return report


def generate_narrative(domain_matches, role_matches) -> str:
    """
    Tạo câu giải thích dạng human-readable.
    Ví dụ output:
    'Both proposals share identical functional structure:
     Actor=Student performs Action=Build on Object=Web Application
     using Technology=ReactJS (D2_FUNCTIONAL match).
     Technical constraints also overlap:
     both require Performance=<2s response time (D3_TECHNICAL match).'
    """
    # TODO: implement narrative generation
    pass
```

### 2.3 Format output Case Study (dùng trực tiếp trong paper)

Với mỗi cặp trong nhóm 2 (SW-BTED đúng, SBERT sai), tạo bảng:

```
Case Study Example — Pair ID: FPT_XXX vs FPT_YYY
Ground Truth: SIMILAR (True Positive)

┌─────────────────────────────────────────────────────────┐
│ Method      │ Score  │ Prediction │ Explanation          │
├─────────────────────────────────────────────────────────┤
│ SBERT       │ 0.4823 │ DIFFERENT ❌│ "cosine sim = 0.48" │
│ SW-BTED     │ 0.7214 │ SIMILAR   ✅│ See below           │
└─────────────────────────────────────────────────────────┘

SW-BTED Structural Explanation:
D2_FUNCTIONAL: 4/5 Actors matched
  - "Student" ↔ "Applicant" [Actor role, T5]
    → both normalized to: "end_user" via TEM
  - "Build registration form" ↔ "Implement submission module"
    → Action: "build/implement", Object: "form/module"
    → CSO match: both map to "web_form_development"
D3_TECHNICAL: Security constraints matched
  - "JWT authentication" ↔ "OAuth2 login"
    → both normalized to: "authentication_mechanism" via TEM

SBERT Failure Reason:
Surface vocabulary is different ("Student/Build/form" vs
"Applicant/Implement/module") → cosine similarity below threshold.
SW-BTED overcomes this via semantic role normalization.
```

Lưu tất cả 15 case reports ra:
```
results/case_study/
├── group1_both_correct/
│   ├── case_01.json
│   ├── case_01_table.md    # format sẵn để copy vào paper
│   └── ...
├── group2_sw_wins/         # ← QUAN TRỌNG NHẤT
│   ├── case_06.json
│   ├── case_06_table.md
│   └── ...
├── group3_sbert_wins/      # honest limitation
│   └── ...
└── case_study_summary.md   # tổng hợp 15 cases, sẵn để paste vào paper
```

### 2.4 Summary Statistics Case Study

```python
summary = {
    "total_cases_analyzed": 15,
    "group2_sw_wins": {
        "count": int,           # thực tế tìm được
        "common_failure_mode_of_sbert": str,  # pattern chung tại sao SBERT sai
        "common_sw_bted_fix": str,            # T5/TEM/CSO nào giúp SW-BTED đúng
    },
    "group3_sbert_wins": {
        "count": int,
        "common_failure_mode_of_swbted": str, # để viết limitation
    },
}
# Lưu ra: results/case_study/case_study_summary.json
```

**Điều kiện thành công Task 2:**
```
✅ Tìm được ít nhất 3 cặp nhóm 2 (SW-BTED đúng, SBERT sai)
✅ Mỗi cặp có narrative explanation rõ ràng ≥ 3 câu
✅ Có thể identify pattern chung tại sao SBERT fails
✅ case_study_summary.md sẵn sàng paste vào paper
```

**DỪNG LẠI sau Task 2 — báo cáo số lượng cặp nhóm 2 tìm được.**

---

## TASK 3 — THÊM BASELINES MỚI

### Bối cảnh

Thêm BM25 và SimCSE để strengthen baseline comparison.
**KHÔNG thêm LLM-as-Judge (GPT-4o)** — gây vấn đề reproducibility.

### 3.1 Baseline B6 — BM25

```python
# Implementation: rank_bm25 library
# pip install rank-bm25

from rank_bm25 import BM25Okapi
import numpy as np

def compute_bm25_similarity(doc_A: str, doc_B: str) -> float:
    """
    BM25 similarity giữa hai documents.
    Treat doc_A như query, doc_B như corpus (và ngược lại),
    lấy trung bình 2 chiều.
    """
    tokens_A = doc_A.lower().split()
    tokens_B = doc_B.lower().split()

    # Direction 1: A queries B
    bm25_B = BM25Okapi([tokens_B])
    score_AtoB = bm25_B.get_scores(tokens_A)[0]

    # Direction 2: B queries A
    bm25_A = BM25Okapi([tokens_A])
    score_BtoA = bm25_A.get_scores(tokens_B)[0]

    # Symmetric score (normalized)
    raw_score = (score_AtoB + score_BtoA) / 2

    # Normalize to [0, 1] using sigmoid
    sim = 1 / (1 + np.exp(-raw_score * 0.1))
    return float(sim)

# Threshold selection: trên validation set (same protocol)
# Tên trong bảng: "B6: BM25"
```

### 3.2 Baseline B7 — SimCSE

```python
# Implementation: princeton-nlp/sup-simcse-roberta-large
# pip install transformers torch

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SimCSEEncoder:
    def __init__(self):
        model_name = "princeton-nlp/sup-simcse-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        # Truncate nếu quá dài (max 512 tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Dùng [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embedding, dim=-1)

    def similarity(self, doc_A: str, doc_B: str) -> float:
        emb_A = self.encode(doc_A)
        emb_B = self.encode(doc_B)
        return float(torch.cosine_similarity(emb_A, emb_B).item())

# Lưu ý: sup-simcse-roberta-large là ~1.4GB model
# Nếu memory constraint → dùng "princeton-nlp/sup-simcse-bert-base-uncased"
# Tên trong bảng: "B7: SimCSE"
```

> ⚠️ **Nếu không thể download model** do network restriction →
> dùng `sentence-transformers/all-mpnet-base-v2` làm thay thế,
> ghi rõ trong paper. DỪNG và báo cáo trước khi skip.

### 3.3 Evaluation Protocol cho B6, B7

**Cùng protocol với các baseline cũ:**
```python
NEW_BASELINES = {
    "B6_BM25": compute_bm25_similarity,
    "B7_SimCSE": simcse_encoder.similarity,
}

for baseline_name, similarity_fn in NEW_BASELINES.items():
    run_5fold_cv(
        similarity_fn=similarity_fn,
        datasets=["FPT", "PURE_adapted"],
        threshold_search="on_validation_set",
        metrics=["precision", "recall", "f1", "roc_auc",
                 "type_a_tpr", "type_b_tnr", "type_c_tnr"],
        mcnemar_vs="SW-BTED",
        bonferroni_alpha=0.01,
    )
```

### 3.4 Updated Master Results Table

Sau khi có B6, B7, tạo bảng hoàn chỉnh:

```
Table X (Updated): SW-BTED vs All Baselines — FPT Dataset

Method            | F1              | ROC-AUC         | McNemar p
------------------|-----------------|-----------------|----------
SW-BTED (Prop.)  | 0.9707 ± 0.0350 | 1.0000 ± 0.0000 | —
B1: Cosine TF-IDF | 0.9939 ± 0.0136 | 1.0000 ± 0.0000 | 0.2188
B2: Cosine SBERT  | 0.9593 ± 0.0387 | 1.0000 ± 0.0000 | 0.7539
B3: Standard TED  | 0.7548 ± 0.0641 | 0.7500 ± 0.0805 | 5.68×10⁻¹⁴ ✓
B4: pq-Gram       | 0.7512 ± 0.0372 | 0.8712 ± 0.0422 | 1.14×10⁻¹³ ✓
B5: Section Cosine| 0.8246 ± 0.0489 | 0.9231 ± 0.0277 | 1.12×10⁻⁷  ✓
B6: BM25          | TBD             | TBD             | TBD
B7: SimCSE        | TBD             | TBD             | TBD
```

Lưu ra: `results/updated_baselines/full_comparison_table.csv`

---

## PHẦN 4 — CLAIM MATRIX CẬP NHẬT SAU 3 TASKS

Agent điền vào sau khi có kết quả:

| Claim | Evidence cần | Source |
|-------|-------------|--------|
| "6-layer > 3-layer" | A1 vs A5, p<0.01 cả 2 dataset | Ablation A |
| "Adaptive T5 generalizes across document types" | A_new FPT≥0.9707 AND PURE≥0.82 | Task 1 |
| "Structural explainability beyond embedding similarity" | ≥3 case study nhóm 2 | Task 2 |
| "SW-BTED > all tree-based methods" | vs B3,B4,B5 p<0.01 | Main eval |
| "SW-BTED competitive with embedding methods" | vs B1,B2,B6,B7 không significant | Main eval |
| "Hybrid > pure structural (FPT)" | D4 vs D6 p<0.01 | Ablation D |

---

## PHẦN 5 — CHECKLIST TỔNG

```
TASK 1: Adaptive T5
  [ ] Implement should_activate_t5() heuristic
  [ ] Log activation rates cho cả 2 dataset
  [ ] Kiểm tra: FPT rate ≥ 60%, PURE rate ≤ 50%
  [ ] Chạy 5-fold CV với T5 adaptive
  [ ] So sánh với A1 và A2
  [ ] Output: results/adaptive_t5/
  [ ] DỪNG và báo cáo

TASK 2: Case Study
  [ ] Tìm cặp nhóm 2 (SW-BTED đúng, SBERT sai)
  [ ] Báo cáo số lượng thực tế TRƯỚC KHI làm tiếp
  [ ] Generate explanation report cho 15 cặp
  [ ] Tạo narrative dạng human-readable
  [ ] Output: results/case_study/
  [ ] DỪNG và báo cáo

TASK 3: New Baselines
  [ ] Implement B6 (BM25) với rank-bm25
  [ ] Implement B7 (SimCSE) với princeton-nlp model
  [ ] Chạy 5-fold CV cùng protocol
  [ ] McNemar vs SW-BTED với Bonferroni
  [ ] Update master results table
  [ ] Output: results/updated_baselines/
```

---

## PHẦN 6 — QUYẾT ĐỊNH AGENT PHẢI HỎI

> **DỪNG và hỏi người dùng** trong các trường hợp:

| Tình huống | Câu hỏi |
|-----------|---------|
| FPT activation rate < 60% | "Ngưỡng T5 quá cao. Thử token_count > 12 không?" |
| PURE activation rate > 50% | "Ngưỡng T5 quá thấp. Thử token_count > 20 không?" |
| Adaptive T5 FPT F1 < 0.9707 | "T5 adaptive làm tệ FPT. Dừng — cần điều chỉnh trước." |
| Nhóm 2 case study < 3 cặp | "Không đủ case SW-BTED beats SBERT. Báo cáo pattern thất bại." |
| SimCSE B7 > SW-BTED significant | "SW-BTED thua SimCSE có ý nghĩa. Báo cáo trước khi tiếp tục." |
| BM25 B6 > SW-BTED significant | "SW-BTED thua BM25 có ý nghĩa. Cần thảo luận với người dùng." |
