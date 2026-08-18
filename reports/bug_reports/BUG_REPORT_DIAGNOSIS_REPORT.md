# Báo Cáo Chẩn Đoán Cấu Trúc Cây SW-BTED Trên Miền Bug Reports (Bug Report Diagnosis Report)

> **Tài liệu chẩn đoán hoàn chỉnh:** Thực hiện theo đặc tả kiểm toán trong [AGENT_TASKS_bug_report_diagnosis.md](../../docs/project_management/AGENT_TASKS_bug_report_diagnosis.md).  
> **Mục đích:** Xử lý triệt để Task 1 (Phép thử Ablation cô lập nguyên nhân cơ học), Task 5 (Kiểm định thống kê McNemar giữa Hybrid Mode Adapted vs. SBERT Full-Text), và Task 6 (Kiểm định thống kê trực tiếp giữa Structural-Only vs. SBERT Full-Text & Phương pháp chọn ngưỡng).

---

## 🔬 1. Kết Quả Phép Thử Ablation Cô Lập & Giải Trình Mâu Thuẫn (Task 1 Resolution)

### A. Phép Thử Ablation Trên 4 Cấu Hình Đối Chứng ($n=300$)

| Cấu Hình Đối Chứng | Ngưỡng Budget Ratio ($k$) | Ngưỡng Cổng Lọc Gate | Duplicate Positives Mean ± Std | Hard Negatives Mean ± Std | Easy Negatives Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A (Gốc - Mặc Định FPT)** | `0.4` | `0.25` | **0.2727 ± 0.0761** | **0.0902 ± 0.0707** | **0.0628 ± 0.0448** |
| **Config B (Tắt Cổng Lọc SBERT)** | `0.4` | **0.00** | **0.2727 ± 0.0761** | **0.0902 ± 0.0707** | **0.0628 ± 0.0448** |
| **Config C (Thích Ứng Budget)** | **1.0** | `0.25` | **0.3902 ± 0.1069** | **0.1308 ± 0.1148** | **0.0807 ± 0.0744** |
| **Config D (Thích Ứng Toàn Bộ)** | **1.0** | **0.00** | **0.3933 ± 0.0984** | **0.1794 ± 0.0866** | **0.1440 ± 0.0587** |

---

### B. Mẫu Thuẫn Giá Trị $0.2727$ Và Giải Trình Cơ Học Hoàn Chỉnh

1. **Vì sao ở Cấu hình Gốc (Config A: Ratio=0.4), giá trị Positives không phải là $0.0000$ mà là $0.2727$?**  
   - Trong hàm `normalize_similarity()` của mô hình SW-BTED (`src/05_sw_bted.py`), công thức tính tổng hợp điểm tương đồng là:  
     $$\text{sim} = \alpha \cdot sim_{\text{struct}} + (1 - \alpha) \cdot sim_{\text{sbert}}$$
   - Với $\alpha = 0.6$, thành phần nhúng toàn cục SBERT đóng góp $(1 - 0.6) \cdot sim_{\text{sbert}} = 0.4 \cdot sim_{\text{sbert}}$.
   - Ngay cả khi $sim_{\text{struct}}$ bị **hard-prune về $0.0000$** do chi phí biến đổi vượt quá $0.4$, hàm vẫn trả về điểm hợp nhất:  
     $$\text{sim} = 0.6 \cdot 0.0000 + 0.4 \cdot sim_{\text{sbert}} = 0.4 \cdot 0.6818 = \mathbf{0.2727}$$
2. **Chứng Minh Bằng Thực Nghiệm Cô Lập:**  
   - So sánh Config A và Config B cho thấy việc tắt Cổng lọc SBERT (`Gate = 0.00`) **không làm thay đổi điểm số** ($0.2727 \implies 0.2727$). Điều này khẳng định Cổng lọc Gate không đóng vai trò triệt tiêu tín hiệu.
   - Ngược lại, khi chuyển từ Ratio=$0.4$ sang Ratio=$1.0$ (Config D), thành phần cây $sim_{\text{struct}}$ thực sự được giải phóng khỏi bị prune, đưa điểm tương đồng trung bình của cặp Positive tăng mạnh từ **$0.2727$ lên $0.3933$** ($+0.1206$).

---

## 📊 2. Bảng Đánh Giá Hiệu Năng 5-Fold Stratified CV Chuẩn Xác (Task 5 Verification)

Độ lệch chuẩn thực tế của Hybrid Mode Adapted là **$\pm 0.0348$** (đã đính chính lỗi gõ phím $\pm 0.348$). Toàn bộ các ngưỡng tối ưu $t^*$ được lựa chọn nghiêm ngặt trên tập train của từng fold (Strict Held-Out Thresholding):

| Phương Pháp / Baseline | Ngưỡng Budget Ratio ($k$) | 5-Fold CV F1-Score | Precision | Recall | Confusion Matrix (TP, FP, TN, FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SBERT Full-Text Baseline** | N/A | **0.9074 ± 0.0304** | `0.9362` | `0.8800` | $(88, 6, 194, 12)$ |
| **SW-BTED Structural-Only (Unbounded)** | $k = 1.0 \times \text{Max\_Cost}$ | **0.6725 ± 0.0194** | `0.5852` | `0.7900` | $(79, 56, 144, 21)$ |
| **SW-BTED Hybrid Mode (Adapted, $\alpha=0.6$)** | $k = 1.0 \times \text{Max\_Cost}$ | **0.9141 ± 0.0348** | `0.9192` | **`0.9100`** | **$(91, 8, 192, 9)$** |

- **F1 theo từng Fold của SBERT Full-Text:** `[0.9231, 0.8718, 0.8718, 0.9231, 0.9474]` $\implies \mathbf{0.9074 \pm 0.0304}$
- **F1 theo từng Fold of Hybrid Adapted:** `[0.9756, 0.9000, 0.8718, 0.9000, 0.9231]` $\implies \mathbf{0.9141 \pm 0.0348}$

---

## 📐 3. Phép Kiểm Thống Kê Trực Tiếp: Joint Contingency Tables & McNemar Tests (Task 5 & 6)

### A. Ma Trận Tần Suất Liên Hợp 2x2: Hybrid Adapted vs. SBERT Full-Text (Task 5)

| | Hybrid Adapted Đúng | Hybrid Adapted Sai | Tổng SBERT |
| :--- | :---: | :---: | :---: |
| **SBERT Full-Text Đúng** | **280** ($n_{11}$) | **2** ($n_{10}$) | **282** |
| **SBERT Full-Text Sai** | **3** ($n_{01}$) | **15** ($n_{00}$) | **18** |
| **Tổng Hybrid** | **283** | **17** | **300** |

* **Số ca bất đồng:** $n_{10} = 2$, $n_{01} = 3$.
* **Exact Binomial McNemar Test:** $p\text{-value} = \text{binomtest}(\min(2, 3), 2+3, p=0.5) = \mathbf{1.0000} > 0.05$.
* **Kết luận:** SW-BTED Hybrid Mode Adapted và SBERT Full-Text **ngang hàng về mặt thống kê (Statistically Tied)**.

---

### B. Ma Trận Tần Suất Liên Hợp 2x2: Structural-Only (Unbounded) vs. SBERT Full-Text (Task 6)

| | Structural-Only Đúng | Structural-Only Sai | Tổng SBERT |
| :--- | :---: | :---: | :---: |
| **SBERT Full-Text Đúng** | **215** ($n_{11}$) | **67** ($n_{10}$) | **282** |
| **SBERT Full-Text Sai** | **8** ($n_{01}$) | **10** ($n_{00}$) | **18** |
| **Tổng Structural-Only** | **223** | **77** | **300** |

* **Số ca bất đồng:** $n_{10} = 67$ (SBERT đúng, Struct-Only sai), $n_{01} = 8$ (Struct-Only đúng, SBERT sai). Total discordant pairs = $75$.
* **Exact Binomial McNemar Test:**  
  $$p\text{-value} = \text{binomtest}(8, 75, p=0.5) = \mathbf{1.0099 \times 10^{-12}} \quad (p \ll 0.001)$$
* **Kết luận:** Mô hình Structural-Only đứng một mình ($F1=0.6725$) **thua kém có ý nghĩa thống kê cực kỳ rõ rệt ($p = 1.0099 \times 10^{-12}$)** so với SBERT Full-Text ($F1=0.9074$). Tín hiệu cây chỉ phát huy hiệu quả khi được hợp nhất với SBERT trong Hybrid Mode ($\alpha=0.6$).

---

### C. Phương Pháp Chọn NgưỡngPhân Loại (Methodology Note For Paper)
- Ngưỡng phân loại $t^*$ được dò tìm tự động bằng **Grid Search** trong khoảng $[0.0, 1.0]$ với bước nhảy $0.005$.
- Tiêu chí tối ưu là **tối đa hóa F1-Score trên tập huấn luyện (train split $tr$)** của từng fold, sau đó khóa cố định ngưỡng $t^*$ này để đánh giá trực tiếp trên tập kiểm thử chưa từng xuất hiện (unseen test split $te$), đảm bảo $100\%$ không bị rò rỉ dữ liệu (Data Leakage).
