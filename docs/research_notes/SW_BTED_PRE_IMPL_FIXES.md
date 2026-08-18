# SW-BTED: Tài liệu Kiểm tra & Sửa lỗi Trước khi Chạy Full Evaluation
> **Quan trọng:** Agent phải thực hiện PHASE 0 trước, báo cáo kết quả,
> rồi mới chạy PHASE 1 và PHASE 2. KHÔNG được gộp tất cả vào một lần chạy.

---

## BỐI CẢNH — Vấn đề phát hiện từ bảng kết quả mới nhất

### Bảng kết quả hiện tại (cần kiểm tra tính hợp lệ)

| Phương pháp | Ngưỡng | Precision | Recall | F1 | ROC-AUC |
|-------------|--------|-----------|--------|----|---------|
| **SW-BTED** | 0.35±0.0327 | 0.9449±0.0645 | **1.0000±0.0000** | **0.9707±0.0350** | **1.0000±0.0000** |
| B1: Cosine TF-IDF | 0.24±0.0396 | 0.9882±0.0263 | 1.0000±0.0000 | 0.9939±0.0136 | 1.0000±0.0000 |
| B2: Cosine SBERT | 0.51±0.0515 | 0.9240±0.0720 | 1.0000±0.0000 | 0.9593±0.0387 | 1.0000±0.0000 |
| B3: Standard TED | 0.37±0.0200 | 0.6283±0.0773 | 0.9500±0.0523 | 0.7548±0.0641 | 0.7500±0.0805 |
| B4: pq-Gram | 0.05±0.0045 | 0.6327±0.0318 | 0.9250±0.0523 | 0.7512±0.0372 | 0.8712±0.0422 |
| B5: Section Cosine | 0.58±0.0964 | 0.7742±0.1471 | 0.9250±0.1355 | 0.8246±0.0489 | 0.9231±0.0277 |

### 3 tín hiệu bất thường cần điều tra

**Tín hiệu 1 — ROC-AUC = 1.0000 ± 0.0000 (SW-BTED, B1, B2):**
ROC-AUC = 1 nghĩa là model phân tách hoàn hảo positive và negative
trên mọi fold không có ngoại lệ. Điều này chỉ xảy ra khi:
- (A) Dataset quá dễ — tất cả positive cases rất giống nhau (verbatim copy)
- (B) Pre-filter đang hoạt động như một **hard oracle** thay vì computational shortcut

**Tín hiệu 2 — Recall = 1.0000 ± 0.0000 (SW-BTED, B1, B2):**
Không bỏ sót bất kỳ positive case nào trên mọi fold → positive cases
không có diversity, hoặc threshold được chọn quá thấp (bias về recall).

**Tín hiệu 3 — B1 (TF-IDF) vẫn F1 = 0.9939:**
TF-IDF gần bằng SW-BTED (0.9707) → dataset không có đủ hard positive cases
để SW-BTED thể hiện lợi thế của cây 6 tầng và semantic role matching.

### Nguyên nhân nghi ngờ chính — Pre-filter hard override

Trong implementation hiện tại có đoạn:
```python
if sim_global < 0.25 or sim_struct_beta == 0.0:
    sim = 0.0   # ← ĐÂY LÀ VẤN ĐỀ
```
Nếu tất cả negative cases có `sim_global < 0.25` → chúng bị gán `sim=0` cứng
→ classifier phân biệt hoàn hảo → ROC-AUC = 1.0
→ Kết quả là của embedding threshold, KHÔNG phải của thuật toán TED.

---

## PHASE 0 — Chẩn đoán (BẮT BUỘC CHẠY TRƯỚC, báo cáo kết quả)

### Task 0.1 — Thống kê phân phối sim_global

```python
# Chạy script này TRƯỚC KHI làm bất cứ điều gì khác
# Mục tiêu: biết bao nhiêu % negative cases bị pre-filter chặn

import numpy as np

results = {
    "positive_pairs": {
        "sim_global_mean": None,   # điền vào sau khi chạy
        "sim_global_std": None,
        "pct_below_0.25": None,    # % positive bị pre-filter loại nhầm
    },
    "negative_pairs": {
        "sim_global_mean": None,
        "sim_global_std": None,
        "pct_below_0.25": None,    # % negative bị pre-filter chặn đúng
    }
}

# Xuất histogram sim_global cho positive và negative riêng biệt
# Lưu ra: diagnostics/sim_global_distribution.png
# Lưu ra: diagnostics/sim_global_stats.json
```

**Điều kiện đánh giá kết quả Task 0.1:**

| Kết quả | Kết luận | Hành động |
|---------|----------|-----------|
| negative pct_below_0.25 > 80% | Pre-filter là oracle, ROC-AUC=1 do pre-filter | → Bắt buộc Fix A |
| negative pct_below_0.25 = 40–80% | Pre-filter có đóng góp nhưng không phải toàn bộ | → Fix A + tính ROC-AUC trên subset |
| negative pct_below_0.25 < 40% | Pre-filter không phải nguyên nhân chính | → Chuyển sang Task 0.2 |

### Task 0.2 — Thống kê difficulty distribution của dataset

```python
# Dùng sim_global (SBERT cosine) làm proxy để phân loại
# KHÔNG dùng ground truth labels ở bước này

tier_stats = {}
for pair in all_pairs:
    sbert_sim = compute_sbert_cosine(pair.doc_A, pair.doc_B)
    if pair.label == 1:  # positive
        if sbert_sim > 0.85:
            tier = "easy_positive"
        else:
            tier = "hard_positive"
    else:  # negative
        if sbert_sim >= 0.30:
            tier = "hard_negative"
        else:
            tier = "easy_negative"

# Lưu ra: diagnostics/tier_distribution.json
# Xuất: {"easy_positive": N, "hard_positive": N,
#         "hard_negative": N, "easy_negative": N}
```

**Điều kiện đánh giá Task 0.2:**

| Kết quả | Kết luận |
|---------|----------|
| hard_positive < 30% tổng positive | Dataset thiếu hard cases → giải thích tại sao TF-IDF gần bằng SW-BTED |
| hard_negative < 20% tổng negative | Negative quá dễ → giải thích ROC-AUC = 1 |

### Task 0.3 — Kiểm tra alpha trong implementation

Xác định chính xác `alpha` trong code đang làm gì:

```python
# Tìm trong code đoạn tính similarity cuối cùng
# và xác nhận công thức thuộc CASE nào:

# CASE A (khả năng cao nhất):
sim = alpha * sim_struct + (1 - alpha) * sim_global

# CASE B:
# alpha chỉ là tên khác của một beta_ell trong config

# CASE C:
# alpha có vai trò khác — mô tả cụ thể
```

Lưu xác nhận ra: `diagnostics/alpha_role_confirmation.txt`

### Output bắt buộc của PHASE 0

```
diagnostics/
├── sim_global_distribution.png   # histogram phân phối
├── sim_global_stats.json         # thống kê số liệu
├── tier_distribution.json        # phân phối theo difficulty
└── alpha_role_confirmation.txt   # CASE A/B/C
```

**DỪNG LẠI sau PHASE 0 và báo cáo kết quả cho người dùng.**
PHASE 1 chỉ được chạy sau khi người dùng xác nhận tiếp tục.

---

## PHASE 1 — Sửa lỗi Pre-filter (chạy sau PHASE 0)

### Fix A — Pre-filter không được override similarity score

**Vấn đề:**
```python
# ❌ Cách hiện tại — pre-filter là hard decision:
if sim_global < 0.25 or sim_struct_beta == 0.0:
    sim = 0.0
```

**Sửa thành:**
```python
# ✅ Cách đúng — pre-filter chỉ skip APTED computation:
PREFILTER_THRESHOLD = 0.25

if sim_global < PREFILTER_THRESHOLD:
    # Skip tính APTED (tiết kiệm computation)
    sim_struct = 0.0   # assume cây quá khác nhau
else:
    sim_struct = compute_apted(tree_A, tree_B, beta)  # tính thật

# Similarity cuối LUÔN kết hợp cả hai thành phần:
sim = alpha * sim_struct + (1 - alpha) * sim_global
# → Không bao giờ set cứng sim = 0.0 dựa trên điều kiện threshold
```

**Lý do:** Pre-filter là computational shortcut (tránh chạy APTED tốn kém
khi hai tài liệu đã rất khác nhau về embedding). Nhưng `sim_global` vẫn
phải đóng góp vào score cuối — nếu set `sim = 0.0` cứng, thành phần
`(1-alpha) * sim_global` bị xóa sổ không có lý do lý thuyết.

### Fix B — Tính ROC-AUC trên hai subset riêng biệt

Vì pre-filter có thể vẫn ảnh hưởng đến ROC-AUC tổng, cần báo cáo thêm:

```python
# Tính ROC-AUC trên 2 subset:
# (1) Toàn bộ dataset (như hiện tại)
roc_auc_full = roc_auc_score(y_true, y_scores)

# (2) Chỉ các cặp VƯỢT QUA pre-filter (sim_global >= 0.25)
mask_above_prefilter = [sim_global[i] >= 0.25 for i in range(len(pairs))]
roc_auc_filtered = roc_auc_score(
    y_true[mask_above_prefilter],
    y_scores[mask_above_prefilter]
)

# Lưu cả hai vào results_leak_free.csv
```

Nếu `roc_auc_filtered < roc_auc_full` đáng kể → xác nhận pre-filter đang
inflate ROC-AUC, và `roc_auc_filtered` mới là số thật cần báo cáo trong paper.

### Fix C — Alpha/Beta báo cáo dưới dạng phân phối, không phải hằng số

Vì alpha và beta được chọn per-fold trên validation:

```python
# Thay vì báo cáo "optimal α=0.8, β=0.7" (sai):
# Báo cáo:
alpha_per_fold = [fold1_alpha, fold2_alpha, ...]
beta_per_fold  = [fold1_beta,  fold2_beta,  ...]

report = {
    "alpha": f"{np.mean(alpha_per_fold):.2f} ± {np.std(alpha_per_fold):.4f}",
    "beta":  f"{np.mean(beta_per_fold):.2f} ± {np.std(beta_per_fold):.4f}",
}
# Lưu ra: results/hyperparameter_distribution.json
```

---

## PHASE 2 — Full Evaluation (chạy sau PHASE 1)

> Đây là nội dung từ `SW_BTED_EVALUATION_FIXES.md` — chạy lại với pre-filter đã sửa.

### Task 2.1 — Leak-free evaluation với proper split

```
Dataset split:
├── 5-fold stratified CV (outer loop)
│   └── Mỗi fold: train(60%) | val(20%) | test(20%)
│       ├── Threshold tối ưu → tìm trên val
│       └── Đánh giá cuối → chỉ trên test
```

**Quan trọng — TF-IDF fit chỉ trên train:**
```python
# ❌ Sai: fit TF-IDF trên toàn bộ dataset
tfidf.fit(all_documents)

# ✅ Đúng: fit chỉ trên train documents của fold hiện tại
train_docs = [doc for pair in train_pairs for doc in [pair.doc_A, pair.doc_B]]
tfidf.fit(train_docs)
tfidf.transform(val_docs)   # transform val
tfidf.transform(test_docs)  # transform test
```

### Task 2.2 — McNemar's test đầy đủ 5 cặp + Bonferroni

```python
pairs_to_test = [
    ("SW-BTED", "B1_Cosine_TFIDF"),
    ("SW-BTED", "B2_Cosine_SBERT"),    # ← BẮT BUỘC có
    ("SW-BTED", "B3_Standard_TED"),
    ("SW-BTED", "B4_pqGram"),
    ("SW-BTED", "B5_Section_Cosine"),
]

alpha_bonferroni = 0.05 / len(pairs_to_test)  # = 0.01

# Lưu ra: results/mcnemar_results.csv
# Columns: pair, b, c, chi2, p_value, significant_at_0.01, note
```

**Ghi chú Wilcoxon bắt buộc thêm vào report:**
```
"Wilcoxon Signed-Rank Test: p = 0.0625 (tất cả các cặp).
Với N=5 fold, p-value tối thiểu có thể đạt được của
Wilcoxon two-sided là 0.0625, đại diện cho trường hợp
SW-BTED vượt trội baseline trên tất cả 5 fold.
Significance tại α=0.05 không thể được claim qua test này
với sample size này. McNemar's test được dùng làm primary
significance measure."
```

### Task 2.3 — Difficulty Tier Analysis

```python
tier_definitions = {
    "easy_positive":  lambda label, sbert: label == 1 and sbert > 0.85,
    "hard_positive":  lambda label, sbert: label == 1 and sbert <= 0.85,
    "hard_negative":  lambda label, sbert: label == 0 and sbert >= 0.30,
    "easy_negative":  lambda label, sbert: label == 0 and sbert < 0.30,
}

# Tính F1 riêng theo tier cho mỗi phương pháp
# Lưu ra: results/results_by_tier.csv
```

**Điều kiện SW-BTED claim đóng góp thực sự:**
```
f1_hard_positive(SW-BTED) > f1_hard_positive(best_baseline) + 0.03
```

### Task 2.4 — Output files

```
results/
├── results_leak_free.csv          # bảng metric đầy đủ
├── mcnemar_results.csv            # significance tests
├── results_by_tier.csv            # phân tích theo difficulty
├── hyperparameter_distribution.json   # alpha/beta per fold
└── roc_auc_comparison.csv         # full vs filtered ROC-AUC
```

---

## CHECKLIST TỔNG — Thứ tự bắt buộc

```
PHASE 0 — Chẩn đoán (CHẠY TRƯỚC, DỪNG LẠI, BÁO CÁO)
  [ ] Task 0.1: Thống kê phân phối sim_global → diagnostics/sim_global_stats.json
  [ ] Task 0.2: Thống kê difficulty tier → diagnostics/tier_distribution.json
  [ ] Task 0.3: Xác nhận role của alpha → diagnostics/alpha_role_confirmation.txt
  [ ] BÁO CÁO KẾT QUẢ CHO NGƯỜI DÙNG — CHỜ XÁC NHẬN

PHASE 1 — Sửa lỗi pre-filter (sau khi PHASE 0 được xác nhận)
  [ ] Fix A: Pre-filter không set cứng sim=0
  [ ] Fix B: Tính ROC-AUC trên full và filtered subset
  [ ] Fix C: Alpha/beta báo cáo dưới dạng mean ± std

PHASE 2 — Full evaluation (sau khi PHASE 1 được xác nhận)
  [ ] Task 2.1: Leak-free 5-fold với TF-IDF fit đúng cách
  [ ] Task 2.2: McNemar đủ 5 cặp + Bonferroni + ghi chú Wilcoxon
  [ ] Task 2.3: Difficulty tier analysis
  [ ] Task 2.4: Xuất đủ output files
```

---

## CÁC QUYẾT ĐỊNH AGENT PHẢI HỎI NGƯỜI DÙNG

> **DỪNG LẠI** và hỏi xác nhận trong các trường hợp sau.
> KHÔNG tự quyết định và chạy tiếp.

| Tình huống | Câu hỏi cần hỏi |
|-----------|-----------------|
| negative pct_below_0.25 > 80% | "Pre-filter là nguyên nhân chính của ROC-AUC=1. Có muốn tăng PREFILTER_THRESHOLD để giữ lại nhiều hard negative hơn không?" |
| hard_positive < 30% tổng positive | "Dataset thiếu hard cases. Có muốn bổ sung thêm hard positive pairs trước khi chạy full evaluation không?" |
| alpha là CASE B hoặc C | "Alpha không phải trọng số kết hợp TED+embedding. Cần xác nhận công thức đúng trước khi viết lại lý thuyết." |
| McNemar vs B2 cho p ≥ 0.01 | "SW-BTED không vượt B2 có ý nghĩa thống kê. Báo cáo kết quả trung thực — không bỏ qua hoặc giấu cặp này." |
| Sau Fix A: F1 của SW-BTED giảm > 0.05 | "F1 giảm đáng kể sau khi sửa pre-filter. Confirm trước khi chạy full PHASE 2." |
