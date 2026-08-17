# SW-BTED v2 — Schema-Weighted Bounded Tree Edit Distance

> **Primary benchmark:** 138 real-only pairs (38 positive, 100 negative)  
> **Representation:** four layers — Root → Domain → Intent → Terminology  
> **Status:** frozen scientific protocol; research-complete, pre-submission.

The public-safe repository contains source code, experiment runners, manuscript
artifacts, figures, and reproducibility instructions. Large/private datasets,
model caches, Kaggle working files, and historical experiment branches are
intentionally excluded from version control. See
`reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md` before interpreting results.

---

## 📁 Thư Mục & Cấu Trúc File (Directory Structure)

```text
SW_BTED_v2/
├── README.md                          <-- (File này) Hướng dẫn tổng quan & chỉ mục
├── AGENT_TASKS_audit_resolution.md    <-- Danh sách kiểm tra nhiệm vụ Audit & Novelty (Task 1 - 7 Done)
│
├── reports/                           <-- THƯ MỤC CHỨA TẤT CẢ BÁO CÁO MARKDOWN
│   ├── AUDIT_AND_NOVELTY_SUMMARY.md   <-- Báo cáo tóm tắt tổng hợp toàn bộ kết quả Audit & Novelty
│   ├── audit/
│   │   ├── count_reconciliation.md    <-- [Task 7] Báo cáo đối soát số lượng 138 Real / 42 Paraphrase Probe
│   │   ├── gpt_augmentation_prompt.md <-- [Task 1] Audit prompt sinh GPT & xác nhận 0% rò rỉ nhãn
│   │   ├── audit_report.md            <-- [Task 1] Phân tích 3 lát cắt dữ liệu (Real-only, Probe, Combined)
│   │   ├── beta_provenance_resolution.md <-- [Task 2] Thử nghiệm & xác nhận bộ trọng số β chuẩn trên Real data
│   │   └── significance_report_v2.md  <-- [Task 3] Kiểm định thống kê 15 phép so sánh có hiệu chỉnh Holm-Bonferroni
│   └── novelty_test/
│       └── NOVELTY_TEST_REPORT.md     <-- [Task 5] Thử nghiệm tính mới (Flat Baseline O(1) vs SW-BTED O(n³))
│
└── data_results/                      <-- THƯ MỤC CHỨA DỮ LIỆU BẢNG BIỂU CSV THÔ
    ├── real_vs_augmented_breakdown.csv
    ├── beta_provenance_results.csv
    ├── significance_report_v2.csv
    └── flat_baseline_comparison.csv
```

---

## 📊 Kết Quả Thực Nghiệm Chuẩn (Canonical Results)

### 1. Benchmark chính (138 real-only pairs):
* **SW-BTED:** $F1 = \mathbf{0.9498 \pm 0.0253}$, $Precision = \mathbf{0.9056}$, $Recall = \mathbf{1.0000}$.
* **Clean-suite result:** SW-BTED significantly outperforms Standard TED ($F1=0.4364$), Section Cosine ($F1=0.6837$), and Genuine Flat Domain SBERT ($F1=0.4314$); it is statistically at parity with clean full-document TF-IDF ($F1=0.9867$) and strong embedding baselines.
* **pq-Gram Baseline:** Đạt $F1 = 0.9479 \pm 0.0478$ trên clean Real-only data và không khác biệt có ý nghĩa thống kê với SW-BTED.

### 2. Thử Nghiệm Tính Mới (Novelty Test - Task 5):
* **Flat Schema-Weighted Baseline ($O(1)$ SBERT per domain):** $F1 = 0.4314 \pm 0.0160$, $Precision = 0.2751$.
* **SW-BTED ($O(n^3)$ Tree Edit Distance):** $F1 = \mathbf{0.9498 \pm 0.0253}$, $Precision = \mathbf{0.9056}$.
* **Kết luận:** Thuật toán Tree Edit Distance gióng hàng sub-tree ($T_3, T_4$) vượt trội hoàn toàn so với mô hình phẳng $O(1)$ với mức tăng **+51.84% F1** và **+63.05% Precision** ($p = 2.52 \times 10^{-29}$).

### 3. Đánh Giá Độ Bền Vững Với Paraphrase (Paraphrase Probe - 42 Pairs):
* **SW-BTED:** Phát hiện **100% văn bản bị GPT paraphrase ($Recall = 1.0000$)**.
* **Thuần từ vựng (TF-IDF & pq-Gram):** Bị thất bại hoàn toàn (**$Recall = 0.0000$**) do không xử lý được biến đổi từ vựng.
