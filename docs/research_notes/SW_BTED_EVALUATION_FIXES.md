# SW-BTED: Tài liệu Sửa lỗi Evaluation
> **Mục đích:** Agent đọc file này và thực hiện tuần tự 4 nhóm fix bên dưới.
> Sau mỗi nhóm, xuất kết quả ra file riêng để kiểm tra trước khi chạy tiếp.
> **KHÔNG** gộp tất cả vào một lần chạy duy nhất.

---

## BỐI CẢNH — Tóm tắt tình trạng hiện tại

### Kết quả cũ (LEAK — không dùng được)

| Phương pháp | Ngưỡng | Precision | Recall | F1 | ROC-AUC |
|-------------|--------|-----------|--------|----|---------|
| **SW-BTED** | 0.10 | 0.7432 | 0.9250 | **0.8228** | 0.8906 |
| B1: Cosine TF-IDF | 0.24 | 0.9882 | 1.0000 | **0.9939** | 1.0000 |
| B2: Cosine SBERT | 0.60 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| B3: Standard TED | 0.35 | 0.6132 | 0.9750 | **0.7504** | 0.7500 |
| B4: pq-Gram | 0.10 | 0.9214 | 0.2500 | **0.3728** | 0.8712 |
| B5: Section Cosine | 0.25 | 0.6898 | 1.0000 | **0.8147** | 0.8988 |

**Vấn đề đã xác định:**
- B2 đạt F1 = 1.0000, STD = 0.0000 trên mọi metric → không thể xảy ra trong
  dataset hợp lệ → dấu hiệu chắc chắn của threshold overfitting trên test set
- Tất cả threshold được chọn trên test set thay vì validation set riêng biệt

### Kết quả mới (leak-free dataset — đã có)
- SW-BTED F1 = 0.8228 trên 6-layer disjoint dataset
- McNemar vs B1: p = 9.3132×10⁻¹⁰ ✓ significant
- McNemar vs B3: p = 1.9220×10⁻⁴ ✓ significant
- Wilcoxon (5-fold): p = 0.0625 trên mọi so sánh → **không significant** (giới hạn toán học)
- Tham số α = 0.8, β = 0.7 xuất hiện trong kết quả nhưng **chưa có trong công thức**

---

## FIX 1 — Chạy lại toàn bộ baselines trên leak-free dataset

### Vấn đề
Hiện tại chỉ có F1 của SW-BTED trên dataset mới. Chưa biết B1–B5 đạt bao nhiêu
trên cùng dataset này. Không có bảng so sánh đầy đủ → không thể claim gì.

### Yêu cầu thực hiện

**Bước 1.1 — Tách dataset đúng cách (bắt buộc, làm trước):**
```
Toàn bộ dataset
├── train:      60%  (dùng để fit TF-IDF vocabulary nếu cần)
├── validation: 20%  (CHỈ dùng để chọn threshold tối ưu)
└── test:       20%  (CHỈ dùng để đánh giá cuối — không được nhìn trước)
```

> ⚠️ Threshold tối ưu của TỪNG phương pháp phải được tìm trên `validation` set.
> Sau khi chọn xong threshold → cố định lại → chạy trên `test` một lần duy nhất.
> Không được quét threshold trên test set.

**Bước 1.2 — Chạy 5 baselines với threshold từ validation:**

Với mỗi baseline B1–B5:
1. Quét threshold trên validation set (range phù hợp, step = 0.01)
2. Chọn threshold cho F1 tốt nhất trên validation
3. Áp threshold đó lên test set → ghi nhận Precision, Recall, F1, ROC-AUC
4. Lặp lại cho 5 fold → tính mean ± std

**Bước 1.3 — Chạy SW-BTED với cùng protocol:**

Tương tự — α và β được chọn trên validation, cố định trước khi test.

**Bước 1.4 — Xuất bảng kết quả:**

Lưu ra file `results_leak_free.csv` với cấu trúc:
```
method, threshold_val, precision_mean, precision_std, recall_mean, recall_std,
f1_mean, f1_std, roc_auc_mean, roc_auc_std,
typeA_tpr_mean, typeA_tpr_std, typeB_tnr_mean, typeB_tnr_std, typeC_tnr_mean, typeC_tnr_std
```

### Điều kiện dừng của Fix 1
- B2 (Cosine SBERT) không còn F1 = 1.0000 trên test set
- Tất cả STD > 0.0000 (trừ trường hợp metric thực sự bằng nhau trên mọi fold)
- Threshold của mọi phương pháp được chọn từ validation, không phải test

---

## FIX 2 — Thay thế Wilcoxon bằng McNemar làm primary significance test

### Vấn đề
Với N = 5 fold, p-value tối thiểu của Wilcoxon two-sided là **0.0625**.
Đây là **giới hạn toán học tuyệt đối** — không thể đạt p < 0.05 dù SW-BTED
thắng trên cả 5 fold. Không thể dùng Wilcoxon để claim significance với N=5.

### Lý do chọn McNemar thay thế
McNemar phù hợp hơn vì:
- Test trực tiếp trên từng sample (không phụ thuộc số fold)
- Đã có kết quả mạnh: vs B1 p = 9.3×10⁻¹⁰, vs B3 p = 1.9×10⁻⁴
- Chuẩn mực phổ biến trong NLP/IR comparison (được dùng trong PassionNet
  và nhiều bài ESWA khác)

### Yêu cầu thực hiện

**Bước 2.1 — Chạy McNemar đầy đủ cho tất cả cặp:**

```python
# Với mỗi cặp (SW-BTED, Bx), tính:
# - contingency table: b = SW-BTED đúng / Bx sai
#                      c = SW-BTED sai / Bx đúng
# - McNemar statistic: chi2 = (|b - c| - 1)^2 / (b + c)
# - p-value từ chi-squared distribution df=1

pairs = [
    ("SW-BTED", "B1_Cosine_TFIDF"),
    ("SW-BTED", "B2_Cosine_SBERT"),   # ← QUAN TRỌNG: phải có cái này
    ("SW-BTED", "B3_Standard_TED"),
    ("SW-BTED", "B4_pqGram"),
    ("SW-BTED", "B5_Section_Cosine"),
]
```

> ⚠️ **Bắt buộc phải có McNemar vs B2.** Nếu thiếu, reviewer sẽ nghi ngờ
> kết quả được cherry-pick.

**Bước 2.2 — Áp dụng Bonferroni correction:**

So sánh 5 cặp → ngưỡng significance điều chỉnh: α_corrected = 0.05 / 5 = **0.01**

```
p < 0.01 → Significant sau Bonferroni correction
0.01 ≤ p < 0.05 → Marginally significant (ghi chú rõ)
p ≥ 0.05 → Not significant (phải báo cáo trung thực)
```

**Bước 2.3 — Xuất kết quả:**

Lưu ra file `mcnemar_results.csv`:
```
pair, b, c, chi2_statistic, p_value, significant_bonferroni, note
```

**Bước 2.4 — Giữ Wilcoxon nhưng chuyển xuống Appendix:**

Vẫn báo cáo Wilcoxon p = 0.0625 với ghi chú rõ ràng:
```
"Note: With N=5 folds, the minimum achievable two-sided Wilcoxon
p-value is 0.0625, which represents SW-BTED outperforming the
baseline on all 5 folds. Statistical significance at α=0.05 cannot
be claimed via this test at this sample size; McNemar's test on
individual predictions is used as the primary significance measure."
```

> Không được xóa Wilcoxon khỏi paper — reviewer sẽ hỏi tại sao không có.
> Trình bày trung thực giới hạn của nó tốt hơn là giấu đi.

---

## FIX 3 — Định nghĩa tường minh tham số α trong công thức

### Vấn đề
Walkthrough báo cáo "α = 0.8, β = 0.7" nhưng công thức chính thức
chỉ có $\beta_\ell$ per layer. α **chưa được định nghĩa ở bất kỳ đâu**
trong lý thuyết → reviewer sẽ từ chối paper ngay tại vòng 1.

### Yêu cầu thực hiện

**Bước 3.1 — Xác định α đang làm gì trong code:**

Agent kiểm tra implementation và xác nhận α thuộc một trong các trường hợp:

```
CASE A: α là trọng số kết hợp TED + embedding trong similarity score cuối
CASE B: α là một trong các β_ℓ bị đặt tên khác trong code
CASE C: α là hệ số scale khác (cần mô tả cụ thể)
```

**Bước 3.2 — Nếu là CASE A (khả năng cao nhất):**

Bổ sung công thức similarity score cuối vào paper:

$$\text{Sim}(A, B) = \alpha \cdot \bigl(1 - \widehat{\text{TED}}(A, B)\bigr) + (1 - \alpha) \cdot \text{cosine}(\mathbf{e}_A, \mathbf{e}_B)$$

Trong đó:
- $\widehat{\text{TED}}(A, B) = \text{TED}(A,B) \;/\; \max\_possible\_cost$ (normalized về [0,1])
- $\mathbf{e}_A, \mathbf{e}_B$ là global embedding tại Root layer (T1)
- $\alpha \in [0,1]$ là trọng số cân bằng giữa structural similarity và semantic similarity

**Bước 3.3 — Bổ sung vào bảng tham số (config.yaml):**

```yaml
# Thêm vào config.yaml hiện tại
alpha: 0.8          # weight for structural TED component vs global embedding
# NOTE: beta trong config là beta_T4 (tầng Atomic Requirement)
# các beta_ell còn lại vẫn theo bảng trong SW_BTED_CHANGES.md
```

**Bước 3.4 — Cập nhật bảng ablation study:**

Thêm α vào bảng hyperparameter tuning với các giá trị đã thử:

```
α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}  × β ∈ {0.3, 0.5, 0.7, 0.9}
→ Báo cáo F1 trên validation set cho từng combination
→ Optimal: α=0.8, β=0.7 (đã xác nhận)
```

---

## FIX 4 — Bổ sung phân tích theo Difficulty Tier

### Vấn đề
F1 tổng thể không đủ để phân biệt khả năng thực sự của SW-BTED vs baselines.
SW-BTED được thiết kế để bắt **idea-level plagiarism** — nếu không có phân
tích riêng cho loại này, đóng góp thực sự của cây 6 tầng không được chứng minh.

### Yêu cầu thực hiện

**Bước 4.1 — Gán nhãn difficulty tier cho từng cặp trong test set:**

```python
TIER_LABELS = {
    "easy_positive":  # cùng ý tưởng, diễn đạt gần giống (cosine SBERT > 0.85)
    "hard_positive":  # cùng ý tưởng, diễn đạt khác nhau (cosine SBERT 0.5–0.85)
    "hard_negative":  # cùng domain nhưng khác bài toán (cosine SBERT 0.3–0.6)
}
```

> Dùng cosine SBERT score như một proxy để gán tier — không dùng nhãn ground truth
> (tránh circular reasoning). Điều chỉnh ngưỡng tier nếu phân phối lệch quá.

**Bước 4.2 — Tính F1 riêng theo tier cho mỗi phương pháp:**

Lưu ra `results_by_tier.csv`:
```
method, f1_easy_pos, f1_hard_pos, f1_hard_neg, n_easy_pos, n_hard_pos, n_hard_neg
```

**Bước 4.3 — Điều kiện claim đóng góp của SW-BTED:**

SW-BTED có thể claim đóng góp có ý nghĩa nếu:
```
f1_hard_pos(SW-BTED) > f1_hard_pos(best_baseline) + 0.03
```
Nếu điều kiện này không đạt → cần xem lại dataset hoặc tham số cây 6 tầng.

---

## CHECKLIST TỔNG — Thứ tự thực hiện

```
[ ] FIX 1: Chạy lại baselines trên leak-free dataset với proper train/val/test split
      └── Output: results_leak_free.csv
      └── Kiểm tra: B2 không còn F1=1.0, mọi STD > 0

[ ] FIX 2: McNemar đầy đủ cho 5 cặp + Bonferroni correction
      └── Output: mcnemar_results.csv
      └── Kiểm tra: Có kết quả vs B2, có ghi chú Wilcoxon N=5 limitation

[ ] FIX 3: Định nghĩa α trong công thức + cập nhật config.yaml
      └── Output: Xác nhận CASE A/B/C + công thức bổ sung
      └── Kiểm tra: α xuất hiện trong cả lý thuyết lẫn implementation

[ ] FIX 4: Phân tích theo difficulty tier
      └── Output: results_by_tier.csv
      └── Kiểm tra: SW-BTED thắng ở hard_positive tier

[ ] TỔNG HỢP: Sau khi có đủ 4 output trên, tạo bảng kết quả cuối
      └── Output: final_results_table.md (sẵn sàng đưa vào paper)
```

---

## CÁC QUYẾT ĐỊNH AGENT PHẢI HỎI XÁC NHẬN TRƯỚC KHI LÀM

> **DỪNG LẠI** và hỏi người dùng trong các trường hợp sau:

1. **α thuộc CASE B hoặc C** (không phải CASE A như giả thuyết):
   Cần người dùng xác nhận công thức đúng trước khi viết lại lý thuyết.

2. **Dataset không đủ mẫu để tách train/val/test 60/20/20:**
   Nếu tổng số cặp < 50 → hỏi trước khi dùng stratified k-fold thay thế.

3. **McNemar vs B2 cho p ≥ 0.05:**
   Đây là kết quả bất lợi nhưng phải báo cáo trung thực.
   Hỏi người dùng về hướng xử lý (không được tự ý bỏ qua hoặc không báo cáo).

4. **Sau FIX 1, F1 của SW-BTED giảm đáng kể so với 0.8228:**
   Hỏi xác nhận trước khi tiếp tục các fix còn lại.
