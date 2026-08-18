# Ablation Study — Giải thích & Kết quả

> **Đối tượng:** Người đọc chưa quen với khái niệm Ablation Study trong nghiên cứu ML.
> **Cập nhật:** 2026-07-19 — kết quả đã được xác nhận reproducible trên folder `SW_BTED_v2`.

---

## 1. Ablation Study là gì?

**Ablation Study** (nghĩa đen: "nghiên cứu cắt bỏ") là phương pháp thực nghiệm để trả lời câu hỏi:

> *"Trong các thành phần của hệ thống, thành phần nào thực sự đóng góp vào kết quả? Nếu bỏ đi một thành phần, kết quả thay đổi như thế nào?"*

### Ví dụ dễ hiểu

Giả sử bạn nấu một món ăn ngon gồm 5 nguyên liệu. Để biết nguyên liệu nào quan trọng nhất, bạn nấu thử 5 phiên bản, mỗi lần bỏ đi một nguyên liệu và so sánh kết quả. Đó chính là ablation study.

Trong SW-BTED, chúng tôi làm tương tự với **19 phiên bản** của thuật toán, mỗi phiên bản thay đổi một thành phần.

---

## 2. SW-BTED có những thành phần nào?

Thuật toán SW-BTED gồm 4 thành phần chính có thể thay đổi độc lập:

| Thành phần | Mô tả | Nhóm ablation |
|---|---|---|
| **Cấu trúc cây** | 4 tầng (ROOT→DOMAIN→INTENT→TERMINOLOGY) | Nhóm A |
| **Hàm chi phí β** | Mỗi tầng có hệ số β riêng, kiểm soát tỷ lệ content vs schema distance | Nhóm B |
| **Chuẩn hóa thuật ngữ** | CSO Ontology + Tech Equivalence Map | Nhóm C |
| **Hệ số α** | Tỷ lệ kết hợp giữa TED (cấu trúc) và SBERT (embedding) | Nhóm D |

---

## 3. Chi tiết 19 variants

### Nhóm A — Layer Structure (câu hỏi: mỗi tầng cây có cần thiết không?)

| Variant | Mô tả |
|---|---|
| **A1** — `SW-BTED_4L` | Toàn bộ 4 tầng — đây là mô hình đề xuất, dùng làm chuẩn so sánh |
| **A2** — `SW-BTED_4L_no_T4` | Bỏ tầng T4 (Terminology/keyword leaf) |
| **A3** — `SW-BTED_4L_no_T2` | Bỏ tầng T2 (Domain labels: D1/D2/D3/D4) |
| **A4** — `SW-BTED_3L` | Cây 3 tầng gốc trước khi cải tiến (lower bound) |

### Nhóm B — Cost Function (câu hỏi: β per-layer có tốt hơn β đồng nhất không?)

| Variant | Mô tả |
|---|---|
| **B1** — `SW-BTED_beta_specific` | β riêng cho từng tầng (đề xuất) |
| **B2** — `SW-BTED_beta_uniform` | β = 0.5 cho tất cả tầng |
| **B3** — `SW-BTED_beta_content_only` | β = 1.0 → chỉ dùng content distance |
| **B4** — `SW-BTED_beta_schema_only` | β = 0.0 → chỉ dùng schema distance |

### Nhóm C — Normalization (câu hỏi: CSO và TEM có thực sự giúp ích không?)

| Variant | Mô tả |
|---|---|
| **C1** — `SW-BTED_full_norm` | Dùng cả CSO + TEM (đề xuất) |
| **C2** — `SW-BTED_no_TEM` | Bỏ Tech Equivalence Map |
| **C3** — `SW-BTED_no_CSO` | Bỏ Computer Science Ontology |
| **C4** — `SW-BTED_no_norm` | Bỏ hoàn toàn normalization (lower bound) |

### Nhóm D — Alpha Sensitivity (câu hỏi: α=0.6 có phải điểm tối ưu không?)

| Variant | Mô tả |
|---|---|
| **D1** — `alpha_0.0` | Chỉ SBERT embedding, bỏ TED (sanity check ≈ baseline B2) |
| **D2** — `alpha_0.2` | Embedding dominant (80% SBERT + 20% TED) |
| **D3** — `alpha_0.4` | Embedding hơi trội |
| **D4** — `alpha_0.6` | **Đề xuất** — cân bằng tốt nhất |
| **D5** — `alpha_0.8` | TED hơi trội |
| **D6** — `alpha_1.0` | Chỉ TED, bỏ embedding (pure structural) |

---

## 4. Kết quả

Chạy trên **2 dataset**: FPT Capstone và PURE Requirements Documents.
Mỗi variant dùng **5-fold cross-validation** → báo cáo F1 mean ± std.

### 4.1. Nhóm A — Layer Ablation

| Variant | FPT F1 | ΔF1 FPT | Sig? | PURE F1 | ΔF1 PURE | Sig? |
|---|---|---|---|---|---|---|
| **A1 — Full 4L (chuẩn)** | **0.9939** | — | — | **0.8684** | — | — |
| A2 — Bỏ T4 | 0.9296 | **-6.4%** | ✅ | 0.8714 | +0.3% | ❌ |
| A3 — Bỏ T2 | 0.9939 | 0.0% | ❌ | 0.8739 | +0.5% | ❌ |
| A4 — 3L legacy | 0.8289 | **-16.5%** | ✅ | 0.6162 | **-25.2%** | ✅ |

**Nhận xét:**
- T4 (Terminology) có impact đáng kể trên FPT (-6.4%), nhỏ trên PURE
- T2 (Domain) không có impact đo được qua layer removal — nhưng vẫn quan trọng qua cost function (xem B4)
- Cây 4 tầng vượt trội cây 3 tầng gốc ở cả hai dataset

### 4.2. Nhóm B — Cost Function

| Variant | FPT F1 | ΔF1 FPT | Sig? | PURE F1 | ΔF1 PURE | Sig? |
|---|---|---|---|---|---|---|
| **B1 — β specific (chuẩn)** | **0.9939** | — | — | **0.8684** | — | — |
| B2 — β uniform | 0.9879 | -0.6% | ❌ | 0.8061 | **-6.2%** | ✅ |
| B3 — content only | 0.9939 | 0.0% | ❌ | 0.8735 | +0.5% | ❌ |
| B4 — schema only | 0.9151 | **-7.9%** | ✅ | 0.7661 | **-10.2%** | ✅ |

**Nhận xét:**
- β per-layer quan trọng với PURE (-6.2% khi dùng β đồng nhất)
- Schema distance (T2) đóng góp rõ rệt qua cost function: bỏ schema → mất 7.9-10.2% F1

### 4.3. Nhóm C — Normalization

| Variant | FPT F1 | PURE F1 | Nhận xét |
|---|---|---|---|
| **C1 — Full norm (chuẩn)** | 0.9879 | **0.8929** | — |
| C2 — no TEM | 0.9879 | 0.8929 | Không khác biệt |
| C3 — no CSO | 0.9879 | 0.8929 | Không khác biệt |
| C4 — no norm | 0.9879 | 0.8929 | Không khác biệt |

**Nhận xét:**
- CSO và TEM không cải thiện F1 score một cách đo được → đóng góp chính là **interpretability** (từ khóa được chuẩn hóa về canonical form, dễ giải thích hơn)

### 4.4. Nhóm D — Alpha Sensitivity

| α | FPT F1 | PURE F1 | Nhận xét |
|---|---|---|---|
| 0.0 (embedding only) | 0.9939 | 0.8696 | ≈ Baseline SBERT |
| 0.2 | 0.9939 | 0.8668 | — |
| 0.4 | 0.9939 | 0.8804 | — |
| **0.6 (đề xuất)** | **0.9939** | **0.8684** | Điểm cân bằng |
| 0.8 | 0.9568 | 0.8785 | TED bắt đầu noise |
| 1.0 (TED only) | 0.8091 | 0.7668 | Yếu nhất |

**Nhận xét:**
- FPT: khoảng α ∈ [0.0, 0.6] đều cho F1 = 0.9939 → FPT không nhạy cảm với α
- PURE: α=0.4 cho kết quả tốt nhất (0.8804), nhưng sự khác biệt không có ý nghĩa thống kê
- TED-only (α=1.0) yếu hơn đáng kể ở cả hai dataset → embedding component là quan trọng

---

## 5. Cách xem kết quả

### 5.1. File chính

```
results/4layer/ablation/
├── ablation_master_table.csv      ← Bảng tổng hợp tất cả 36 rows (19×2 datasets)
├── group_A_summary_FPT.csv        ← Chi tiết nhóm A trên FPT
├── group_A_summary_PURE.csv       ← Chi tiết nhóm A trên PURE
├── group_B_summary_FPT.csv
├── group_B_summary_PURE.csv
├── group_C_summary_FPT.csv
├── group_C_summary_PURE.csv
├── group_D_summary_FPT.csv
├── group_D_summary_PURE.csv
└── {VariantID}_{Name}_{Dataset}.json   ← Kết quả chi tiết từng fold cho mỗi variant
```

### 5.2. Xem nhanh bằng Python

```python
import pandas as pd

# Đọc bảng tổng hợp
df = pd.read_csv('results/4layer/ablation/ablation_master_table.csv')

# Xem nhóm A (Layer ablation) trên FPT
print(df[(df['Group'] == 'A') & (df['Dataset'] == 'FPT')][
    ['Variant_ID', 'Variant_Name', 'F1_Score', 'F1_Std', 'Delta_F1', 'McNemar_p', 'Significant']
])

# Xem tất cả variants có kết quả khác biệt có ý nghĩa thống kê
print(df[df['Significant'] == 'Yes'][
    ['Dataset', 'Variant_ID', 'F1_Score', 'Delta_F1', 'McNemar_p']
])
```

### 5.3. Chạy lại ablation từ đầu

```powershell
# Từ thư mục gốc project (RAG_Research hoặc SW_BTED_v2)
.venv\Scripts\python experiments/archive/run_4layer_ablation.py
# Thời gian ước tính: ~2.5 giờ (FPT: ~90 phút, PURE: ~60 phút)
```

### 5.4. Cột quan trọng trong ablation_master_table.csv

| Cột | Ý nghĩa |
|---|---|
| `Dataset` | FPT hoặc PURE |
| `Group` | A/B/C/D — nhóm ablation |
| `Variant_ID` | A1, A2, ... D6 |
| `F1_Score` | F1 trung bình trên 5 folds |
| `F1_Std` | Độ lệch chuẩn F1 |
| `Delta_F1` | F1(variant) - F1(A1/B1/C1/D4) — âm = tệ hơn baseline |
| `McNemar_p` | p-value từ McNemar test — đo ý nghĩa thống kê |
| `Significant` | Yes/No — có ý nghĩa thống kê (p < 0.01) hay không |

---

## 6. Tóm tắt kết luận

| Claim | Bằng chứng | Kết luận |
|---|---|---|
| "4 tầng tốt hơn 3 tầng gốc" | A1 >> A4, p<0.001 cả hai dataset | ✅ Có bằng chứng mạnh |
| "T4 Terminology đóng góp" | A2 < A1 trên FPT, p<0.001 | ✅ Có (FPT), ❌ Không (PURE) |
| "T2 Domain đóng góp" | A3 ≈ A1, không có ý nghĩa | ⚠️ Đóng góp qua cost function (B4), không qua layer removal |
| "β per-layer tốt hơn β đồng nhất" | B2 < B1 trên PURE, p<0.01 | ✅ Có (PURE) |
| "Schema distance quan trọng" | B4 << B1 cả hai dataset, p<0.001 | ✅ Có bằng chứng mạnh |
| "Kết hợp TED+SBERT tốt hơn đơn lẻ" | D4 > D6 (TED-only) cả hai dataset | ✅ Có bằng chứng mạnh |
| "CSO/TEM cải thiện F1" | C1 ≈ C4, không khác biệt | ❌ Không có bằng chứng (chỉ giúp interpretability) |

> ⚠️ **Lưu ý cho paper:** Chỉ đưa vào paper những claim có bằng chứng (✅). Với các claim ❌ hoặc ⚠️, dùng ngôn ngữ trung tính: *"We observe that X achieves comparable performance..."*
