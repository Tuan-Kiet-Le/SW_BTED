# Báo Cáo Thực Nghiệm Bug Reports Cross-Domain Validation (CapTree 4 Tầng Đồng Nhất)

## 1. Phân Bổ Điểm Tương Đồng Thô (Raw Similarity Distributions Across 4 Ablation Configurations)

| Cấu Hình Đối Chứng | Ngưỡng Budget Ratio ($k$) | Ngưỡng Cổng Lọc Gate | Duplicate Positives Mean ± Std | Hard Negatives Mean ± Std | Easy Negatives Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A (Gốc - Mặc Định FPT)** | `0.4` | `0.25` | **0.2727 ± 0.0761** | **0.0902 ± 0.0707** | **0.0628 ± 0.0448** |
| **Config B (Tắt Cổng Lọc SBERT)** | `0.4` | **0.00** | **0.2727 ± 0.0761** | **0.0902 ± 0.0707** | **0.0628 ± 0.0448** |
| **Config C (Thích Ứng Budget)** | **1.0** | `0.25` | **0.3902 ± 0.1069** | **0.1308 ± 0.1148** | **0.0807 ± 0.0744** |
| **Config D (Thích Ứng Toàn Bộ)** | **1.0** | **0.00** | **0.3933 ± 0.0984** | **0.1794 ± 0.0866** | **0.1440 ± 0.0587** |

> **Giải trình giá trị $0.2727$ ở Cấu hình Gốc:** Thành phần điểm $0.2727$ của cặp Positive ở Config A đến từ $0.4 \times sim_{\text{sbert}}$ (SBERT global embedding) trong công thức tổng hợp $\alpha \cdot sim_{\text{struct}} + (1-\alpha) \cdot sim_{\text{sbert}}$, khi điểm $sim_{\text{struct}}$ thuần túy bị hard-prune về $0.0000$ do `ratio = 0.4`.

---

## 2. Kết Quả Đánh Giá Phân Loại 5-Fold Stratified Cross-Validation ($n=300$)

### Benchmark Baseline & SW-BTED Comparison Sample ($n=300$)

| Phương Pháp / Baseline | Cấu Trúc Cây $O(n^3)$ | 5-Fold CV F1-Score | Precision | Recall | Confusion Matrix (TP, FP, TN, FN) | McNemar $p$-value vs SBERT Full-Text |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Full-Text** | KHÔNG | **0.9084 ± 0.0535** | `0.9184` | `0.9000` | $(90, 8, 192, 10)$ | $p = 1.0000$ |
| **SBERT Full-Text** | KHÔNG | **0.9074 ± 0.0304** | `0.9362` | `0.8800` | $(88, 6, 194, 12)$ | Reference Baseline |
| **Genuine Flat Domain SBERT** | KHÔNG ($0.0$) | **0.7673 ± 0.0201** | `0.8353` | `0.7100` | $(71, 14, 186, 29)$ | $p = 7.0255 \times 10^{-5}$ |
| **SW-BTED Structural-Only (Unbounded)** | **CÓ ($O(n^3)$)** | **0.6725 ± 0.0194** | `0.5852` | `0.7900` | $(79, 56, 144, 21)$ | **$p = 1.0099 \times 10^{-12}$ (Significantly Worse)** |
| **SW-BTED Hybrid Mode (Adapted, $\alpha=0.6$)** | **CÓ ($O(n^3) + \text{SBERT}$)** | **0.9141 ± 0.0348** | `0.9192` | `0.9100` | $(91, 8, 192, 9)$ | **$p = 1.0000$ (Statistically Tied)** |

---

## 3. Ma Trận Tần Suất Liên Hợp 2x2 & Direct McNemar Tests ($n=300$)

### A. Hybrid Adapted vs. SBERT Full-Text
- Ma trận 2x2: $n_{11}=280, n_{00}=15, n_{10}=2, n_{01}=3$.
- Exact Binomial McNemar Test: $p = \mathbf{1.0000 > 0.05}$ $\implies$ **Statistically Tied**.

### B. Structural-Only (Unbounded) vs. SBERT Full-Text
- Ma trận 2x2: $n_{11}=215, n_{00}=10, n_{10}=67, n_{01}=8$. Total discordant pairs = $75$.
- Exact Binomial McNemar Test: $p = \mathbf{1.0099 \times 10^{-12} \ll 0.001}$ $\implies$ **Significantly Inferior**.
