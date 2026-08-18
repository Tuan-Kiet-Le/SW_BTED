# SW-BTED: Ablation Study & Dataset-2 (PURE) Implementation Spec
> **Mục đích:** Agent đọc file này và implement đầy đủ ablation study (19 variants)
> trên 2 dataset: Dataset-1 (FPT capstone) và Dataset-2 (PURE adapted).
> Thực hiện tuần tự theo thứ tự PHASE bên dưới. Dừng và báo cáo sau mỗi PHASE.

---

## PHẦN 1 — DATASET-2: PURE ADAPTATION

### 1.1 Tại sao dùng PURE và tại sao phải adapt

PassionNet datasets (UAV, WorldVista, PURE, OPENCOSS, StackOverflow, Bugzilla)
là **sentence-pair datasets** — mỗi sample là một cặp câu requirement đơn lẻ.
SW-BTED hoạt động ở **document level** → granularity mismatch.

PURE được xây từ **79 SRS documents** (THEMAS và Mashbot làm nguồn chính).
Đây là dataset duy nhất trong PassionNet có thể group theo document gốc
để tạo document-level pairs phù hợp với SW-BTED.

> ⚠️ **KHÔNG dùng StackOverflow, Bugzilla** — đây là bug reports, không phải
> requirement documents. Domain quá xa so với capstone registration forms.
> **KHÔNG dùng UAV, WorldVista, OPENCOSS** — không có SRS document gốc để group.

### 1.2 Pipeline adapt PURE thành document-level dataset

```
Input:  PURE sentence-pair dataset (Malik et al. 2023a)
        → Mỗi row: (req_i, req_j, label) với label ∈ {conflict, neutral}

Output: Document-pair dataset (SW-BTED compatible)
        → Mỗi row: (doc_A, doc_B, label) với label ∈ {1=similar, 0=different}
```

**Bước 1.2.1 — Group requirements theo SRS document gốc:**

```python
# PURE dataset chứa requirements từ THEMAS và Mashbot
# Dựa vào prefix của requirement ID để phân biệt nguồn gốc
# Nếu không có ID prefix → dùng content clustering để group

# Target: tạo ~20-30 "pseudo-documents"
# Mỗi pseudo-document = tập hợp requirements từ cùng một system/module

GROUPING_STRATEGY = {
    "primary": "source_document_id",      # nếu có trong metadata
    "fallback": "kmeans_clustering",       # k=15-20 clusters
    "min_reqs_per_doc": 5,                 # loại document quá nhỏ
    "max_reqs_per_doc": 50,                # loại document quá lớn
}
```

**Bước 1.2.2 — Xây dựng cây SW-BTED cho mỗi pseudo-document:**

```python
# Với mỗi pseudo-document D:
# T1: ROOT = document_id
# T2: DOMAIN = D2_FUNCTIONAL (tất cả requirements đều là functional)
#              D3_TECHNICAL_REALIZATION (nếu có NFR)
# T3: GROUP = bỏ qua (không có actor grouping ở sentence level)
# T4: ATOMIC_REQ = mỗi requirement sentence = 1 node
# T5: SEMANTIC_ROLE = parse bằng spaCy en_core_web_trf
# T6: LEAF = normalize qua CSO + TEM

# Lưu ý: T3 (GROUP) không xuất hiện → cây có dạng
# ROOT → DOMAIN → ATOMIC_REQ → SEMANTIC_ROLE → LEAF (5 tầng thực tế)
```

**Bước 1.2.3 — Tạo document-pair labels:**

```python
# Với mỗi cặp pseudo-document (D_A, D_B):
# Đếm số requirement pairs (r_i, r_j) với r_i ∈ D_A, r_j ∈ D_B
# có label = "duplicate" trong PURE dataset gốc

def label_document_pair(doc_A, doc_B, pure_pairs):
    matching_pairs = [
        p for p in pure_pairs
        if p.req_i in doc_A.requirements
        and p.req_j in doc_B.requirements
        and p.label == "duplicate"
    ]
    overlap_ratio = len(matching_pairs) / min(len(doc_A.requirements),
                                               len(doc_B.requirements))

    # Ngưỡng: nếu > 30% requirements trùng lặp → label = 1 (similar)
    OVERLAP_THRESHOLD = 0.30
    return 1 if overlap_ratio >= OVERLAP_THRESHOLD else 0
```

**Bước 1.2.4 — Kiểm tra balance và đủ mẫu:**

```python
TARGET = {
    "min_total_pairs": 150,
    "min_positive_pairs": 40,
    "min_negative_pairs": 80,
    "positive_ratio_range": (0.20, 0.50),  # không quá imbalanced
}
# Nếu không đạt → báo cáo và hỏi người dùng trước khi tiếp tục
```

**Bước 1.2.5 — Ghi nhận và báo cáo:**

Lưu ra `datasets/pure_adapted/`:
```
pure_adapted/
├── pseudo_documents.json       # danh sách pseudo-docs + requirements
├── document_pairs.csv          # (doc_A_id, doc_B_id, label, overlap_ratio)
├── dataset_stats.json          # tổng số pairs, tỷ lệ pos/neg, distribution
└── adaptation_log.txt          # ghi lại quyết định grouping
```

> **DỪNG LẠI** sau Bước 1.2.5 và báo cáo `dataset_stats.json`
> cho người dùng xác nhận trước khi chạy ablation.

---

## PHẦN 2 — ABLATION STUDY: 19 VARIANTS

### 2.0 Framework chung

Tất cả 19 variants dùng **cùng evaluation protocol**:
- 5-fold stratified CV với proper train/val/test split (60/20/20)
- Threshold tối ưu chọn trên validation set
- McNemar test giữa mỗi variant và SW-BTED-full (proposed)
- Chạy trên **cả 2 dataset** (FPT + PURE adapted)

**Base config (SW-BTED-full = proposed method):**
```yaml
# Đây là điểm xuất phát, các variant thay đổi từng phần
alpha: 0.60
beta:
  T2: 0.0
  T3: 0.6
  T4: 0.9
  T5: 0.0
  T6: 0.8
w_del: {T2: 2.0, T3: 1.5, T4: 1.0, T5: 0.8, T6: 0.5}
w_ins: {T2: 2.0, T3: 1.5, T4: 1.0, T5: 0.8, T6: 0.5}
normalization: {cso: true, tech_equiv_map: true}
layers: [T1, T2, T3, T4, T5, T6]
prefilter_threshold: 0.25
```

---

### 2.1 Nhóm A — Layer Structure Ablation (5 variants)
*Câu hỏi: Mỗi tầng trong cây 6 tầng có thực sự đóng góp không?*

#### A1: SW-BTED-6L (= SW-BTED-full, baseline của ablation)
```yaml
# Không thay đổi gì — đây là điểm tham chiếu cho nhóm A
layers: [T1, T2, T3, T4, T5, T6]
```

#### A2: SW-BTED-5L-noRole
```yaml
# Bỏ T5 (Semantic Role) — T4 connect trực tiếp xuống T6
layers: [T1, T2, T3, T4, T6]
# T4 ATOMIC_REQ kết nối trực tiếp với T6 LEAF
# T6 = tất cả keywords trong câu, không phân nhóm theo role
change: "Remove T5 layer. T4 children = all leaf keywords directly."
```

#### A3: SW-BTED-5L-noGroup
```yaml
# Bỏ T3 (Group) — T2 connect trực tiếp xuống T4
layers: [T1, T2, T4, T5, T6]
change: "Remove T3 layer. T2 DOMAIN children = T4 ATOMIC_REQ nodes directly."
```

#### A4: SW-BTED-4L
```yaml
# Bỏ cả T3 và T5
layers: [T1, T2, T4, T6]
change: "Remove T3 and T5. Tree: ROOT → DOMAIN → ATOMIC_REQ → LEAF"
```

#### A5: SW-BTED-3L (cây gốc ban đầu — lower bound)
```yaml
# Cây 3 tầng ban đầu trước khi có kiến trúc mới
layers: [T1, T2_legacy, T3_keyword]
change: |
  Rebuild original 3-layer tree:
  T1: ROOT
  T2: 7 fixed sections (Context, Solution, Theory, FR, NFR, Products, Tasks)
  T3: flat keyword list per section (no semantic role, no normalization)
  Cost: uniform weights, no schema distance
```

---

### 2.2 Nhóm B — Cost Function Ablation (4 variants)
*Câu hỏi: β_ℓ per-layer có tốt hơn β đồng nhất không?*

#### B1: SW-BTED-β_specific (= SW-BTED-full)
```yaml
# Không thay đổi
beta: {T2: 0.0, T3: 0.6, T4: 0.9, T5: 0.0, T6: 0.8}
```

#### B2: SW-BTED-β_uniform
```yaml
# Cùng một giá trị β cho mọi tầng
# Dùng giá trị trung bình của proposed: mean(0.0,0.6,0.9,0.0,0.8) ≈ 0.46 → round = 0.5
beta: {T2: 0.5, T3: 0.5, T4: 0.5, T5: 0.5, T6: 0.5}
change: "Same beta=0.5 for all layers."
```

#### B3: SW-BTED-β_content_only
```yaml
# β=1 mọi tầng → chỉ dùng content distance, bỏ qua schema distance
beta: {T2: 1.0, T3: 1.0, T4: 1.0, T5: 1.0, T6: 1.0}
change: "w_rep = (w_del+w_ins) * Dist_content only. Schema distance = 0."
# Lưu ý: T2 và T5 với β=1 → dist_schema bị bỏ qua hoàn toàn
# → Hai nodes khác domain hoặc khác role type vẫn có thể match nếu content gần nhau
```

#### B4: SW-BTED-β_schema_only
```yaml
# β=0 mọi tầng → chỉ dùng schema distance, bỏ qua content distance
beta: {T2: 0.0, T3: 0.0, T4: 0.0, T5: 0.0, T6: 0.0}
change: "w_rep = (w_del+w_ins) * Dist_schema only. Content distance = 0."
# → Similarity chỉ dựa trên cấu trúc type, không xem xét nội dung từ khóa
```

---

### 2.3 Nhóm C — Normalization Ablation (4 variants)
*Câu hỏi: CSO và Tech Equivalence Map có đóng góp thực sự không?*

#### C1: SW-BTED-full-norm (= SW-BTED-full)
```yaml
normalization: {cso: true, tech_equiv_map: true}
```

#### C2: SW-BTED-no-TEM
```yaml
normalization: {cso: true, tech_equiv_map: false}
change: |
  Bỏ Tech Equivalence Map lookup tại T6.
  Các từ không có trong CSO giữ nguyên raw lemmatized form.
  Ví dụ: "JWT" không được map về "json_web_token" → giữ nguyên "jwt"
```

#### C3: SW-BTED-no-CSO
```yaml
normalization: {cso: false, tech_equiv_map: true}
change: |
  Bỏ CSO v3.5 lookup tại T6.
  Chỉ dùng Tech Equivalence Map cho các thuật ngữ đã biết.
  Các từ khác giữ nguyên lemmatized form.
```

#### C4: SW-BTED-no-norm
```yaml
normalization: {cso: false, tech_equiv_map: false}
change: |
  Bỏ toàn bộ normalization.
  T6 LEAF = raw lowercased lemmatized keywords.
  Không tra CSO, không tra TEM.
  → Đây là lower bound của normalization contribution.
```

---

### 2.4 Nhóm D — Alpha Weight Ablation (6 variants)
*Câu hỏi: α=0.6 có phải điểm cân bằng tốt nhất không? Structural TED có thực sự giúp ích so với chỉ dùng embedding?*

#### D1: SW-BTED-α0.0 (= Baseline B2 về mặt toán học)
```yaml
alpha: 0.0
change: |
  sim = 0.0 * sim_struct + 1.0 * sim_global
  → Chỉ dùng SBERT global embedding, bỏ hoàn toàn TED
  → Kết quả phải ≈ B2 (Cosine SBERT). Nếu không → có bug.
  Dùng như sanity check.
```

#### D2: SW-BTED-α0.2
```yaml
alpha: 0.2
change: "sim = 0.2*sim_struct + 0.8*sim_global (embedding dominant)"
```

#### D3: SW-BTED-α0.4
```yaml
alpha: 0.4
change: "sim = 0.4*sim_struct + 0.6*sim_global"
```

#### D4: SW-BTED-α0.6 (= SW-BTED-full)
```yaml
alpha: 0.6  # proposed, không thay đổi
```

#### D5: SW-BTED-α0.8
```yaml
alpha: 0.8
change: "sim = 0.8*sim_struct + 0.2*sim_global (structural dominant)"
```

#### D6: SW-BTED-α1.0
```yaml
alpha: 1.0
change: |
  sim = 1.0*sim_struct + 0.0*sim_global
  → Chỉ dùng TED, bỏ hoàn toàn embedding
  → Pure structural similarity
```

---

## PHẦN 3 — EXECUTION PLAN

### 3.1 Thứ tự chạy

```
PHASE 0: Dataset-2 preparation (PURE adaptation)
  → Dừng, báo cáo dataset_stats.json, chờ xác nhận

PHASE 1: Sanity checks (bắt buộc trước khi chạy full ablation)
  → Chạy D1 (α=0.0) và kiểm tra kết quả ≈ B2 baseline
  → Chạy A5 (3L) và kiểm tra kết quả ≈ B3/B5 range
  → Nếu sanity checks fail → dừng, báo cáo

PHASE 2: Nhóm A (Layer ablation) — 5 variants × 2 datasets
PHASE 3: Nhóm B (Cost function ablation) — 4 variants × 2 datasets
PHASE 4: Nhóm C (Normalization ablation) — 4 variants × 2 datasets
PHASE 5: Nhóm D (Alpha ablation) — 6 variants × 2 datasets
PHASE 6: Statistical tests + Summary tables
```

### 3.2 Output cho mỗi variant

```python
# Với mỗi variant X chạy trên dataset D:
output = {
    "variant_id": "A2",
    "variant_name": "SW-BTED-5L-noRole",
    "dataset": "FPT" | "PURE_adapted",
    "metrics": {
        "precision": "mean ± std",
        "recall": "mean ± std",
        "f1": "mean ± std",
        "roc_auc": "mean ± std",
    },
    "mcnemar_vs_full": {
        "chi2": float,
        "p_value": float,
        "significant": bool,  # p < 0.01 after Bonferroni
    },
    "delta_f1_vs_full": float,  # F1(variant) - F1(full), âm = variant tệ hơn
}
```

### 3.3 Lưu trữ kết quả

```
results/ablation/
├── group_A_layer/
│   ├── A1_6L_FPT.json
│   ├── A1_6L_PURE.json
│   ├── A2_5L_noRole_FPT.json
│   ├── ...
│   └── group_A_summary.csv
├── group_B_cost/
│   └── group_B_summary.csv
├── group_C_norm/
│   └── group_C_summary.csv
├── group_D_alpha/
│   └── group_D_summary.csv
└── ablation_master_table.csv   # tất cả 19 variants × 2 datasets
```

---

## PHẦN 4 — BẢNG KẾT QUẢ MỤC TIÊU (template cho paper)

Agent tạo các bảng này từ kết quả thực nghiệm:

### Bảng 4.1 — Layer Ablation Results

| Variant | Layers | FPT F1 | PURE F1 | McNemar p (FPT) | ΔF1 vs Full |
|---------|--------|--------|---------|-----------------|-------------|
| SW-BTED-6L (Full) | T1-T6 | — | — | — | 0.000 |
| SW-BTED-5L-noRole | T1-T2,T4,T6 | — | — | — | — |
| SW-BTED-5L-noGroup | T1-T2,T4-T6 | — | — | — | — |
| SW-BTED-4L | T1,T2,T4,T6 | — | — | — | — |
| SW-BTED-3L (Legacy) | 3 layers | — | — | — | — |

### Bảng 4.2 — Cost Function Ablation Results

| Variant | β config | FPT F1 | PURE F1 | McNemar p | ΔF1 |
|---------|----------|--------|---------|-----------|-----|
| β_specific (Full) | per-layer | — | — | — | 0.000 |
| β_uniform | 0.5 all | — | — | — | — |
| β_content_only | 1.0 all | — | — | — | — |
| β_schema_only | 0.0 all | — | — | — | — |

### Bảng 4.3 — Normalization Ablation Results

| Variant | CSO | TEM | FPT F1 | PURE F1 | McNemar p | ΔF1 |
|---------|-----|-----|--------|---------|-----------|-----|
| Full norm | ✅ | ✅ | — | — | — | 0.000 |
| No TEM | ✅ | ❌ | — | — | — | — |
| No CSO | ❌ | ✅ | — | — | — | — |
| No norm | ❌ | ❌ | — | — | — | — |

### Bảng 4.4 — Alpha Sensitivity Analysis

| α | Role | FPT F1 | PURE F1 | ΔF1 vs α=0.6 |
|---|------|--------|---------|-------------|
| 0.0 | Embedding only (≈B2) | — | — | — |
| 0.2 | Embedding dominant | — | — | — |
| 0.4 | Balanced (embed) | — | — | — |
| **0.6** | **Proposed** | **—** | **—** | **0.000** |
| 0.8 | Balanced (struct) | — | — | — |
| 1.0 | Structural only | — | — | — |

---

## PHẦN 5 — CLAIM MATRIX (dùng để viết paper)

Agent điền vào bảng này sau khi có kết quả:

| Claim trong paper | Bằng chứng cần thiết | Nhóm ablation |
|------------------|---------------------|---------------|
| "Kiến trúc 6 tầng vượt trội 3 tầng gốc" | F1(A1) > F1(A5), p<0.01 | Nhóm A |
| "Semantic Role (T5) đóng góp đáng kể" | F1(A1) > F1(A2), p<0.01 | Nhóm A |
| "β per-layer tốt hơn β đồng nhất" | F1(B1) > F1(B2), p<0.01 | Nhóm B |
| "TEM+CSO cải thiện performance" | F1(C1) > F1(C4), p<0.01 | Nhóm C |
| "Structural TED đóng góp ngoài embedding" | F1(D4) > F1(D1), p<0.01 | Nhóm D |
| "Kết hợp struct+embed > từng thành phần" | F1(D4) > F1(D1) AND F1(D4) > F1(D6) | Nhóm D |

> **Lưu ý quan trọng:** Nếu một claim không được support bởi kết quả
> → KHÔNG được viết claim đó vào paper.
> → Thay bằng: "We observe that [X] achieves comparable performance..."
> → Báo cáo cho người dùng ngay khi phát hiện.

---

## PHẦN 6 — CÁC QUYẾT ĐỊNH AGENT PHẢI HỎI

> **DỪNG LẠI** và hỏi xác nhận trước khi tiếp tục:

| Tình huống | Câu hỏi |
|-----------|---------|
| PURE adaptation tạo được < 150 pairs | "Dataset quá nhỏ. Có muốn điều chỉnh OVERLAP_THRESHOLD (hiện = 0.30) để tạo thêm positive pairs không?" |
| PURE adapted positive ratio < 20% hoặc > 50% | "Dataset imbalanced. Có muốn undersample negative để cân bằng không?" |
| Sanity check D1 (α=0.0) cho F1 khác B2 > 0.03 | "Có thể có bug trong combine_sw_similarity. Cần kiểm tra lại trước khi chạy tiếp." |
| Sanity check A5 (3L) cho F1 > SW-BTED-full | "Cây 3 tầng tốt hơn 6 tầng — đây là kết quả bất thường. Báo cáo ngay." |
| Bất kỳ claim nào trong Phần 5 không được support | "Claim [X] không có bằng chứng. Không được đưa vào paper. Báo cáo để người dùng quyết định." |
| Chạy xong PHASE 2 (Nhóm A) | "Báo cáo group_A_summary.csv trước khi chạy tiếp Nhóm B." |
