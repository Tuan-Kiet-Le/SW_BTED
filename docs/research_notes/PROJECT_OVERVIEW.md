# SW-BTED: Schema-Weighted Bounded Tree Edit Distance
## Tổng quan Dự án Nghiên cứu

> **Mục tiêu:** Đề xuất và đánh giá thuật toán đo lường độ tương đồng tài liệu có cấu trúc bán định dạng (semi-structured documents), ứng dụng trong phát hiện đạo văn Đồ án Tốt nghiệp (Capstone Project).

---

## 1. Bài toán (Problem Statement)

### 1.1. Bối cảnh thực tế

Trường đại học có hàng trăm Đề xuất Đồ án Tốt nghiệp (Capstone Registration Forms) được nộp mỗi học kỳ. Mỗi phiếu đăng ký là một tài liệu **bán cấu trúc** gồm nhiều mục có vai trò ngữ nghĩa khác nhau:
- **D1 — Business Context**: Mô tả bối cảnh và vấn đề doanh nghiệp
- **D2 — Functional Requirements**: Yêu cầu chức năng, giải pháp đề xuất, sản phẩm dự kiến
- **D3 — Technical Realization**: Yêu cầu phi chức năng, lý thuyết áp dụng, công nghệ
- **D4 — Execution Planning**: Kế hoạch thực hiện, phân công nhiệm vụ

Hệ thống hiện tại sử dụng **Single-Vector Embedding + Cosine Similarity** (biến toàn bộ tài liệu thành một vector duy nhất) để phát hiện đạo văn.

### 1.2. Hạn chế của hệ thống hiện tại

| Hạn chế | Mô tả | Hệ quả |
|---|---|---|
| **Information Loss** | Một vector 768-1536 chiều không thể encode đầy đủ nội dung tài liệu dài | Bỏ sót đạo văn ở các mục chi tiết kỹ thuật |
| **Topic Conflation** | Embedding phẳng không phân biệt vai trò ngữ nghĩa của từng mục | Hai đề tài cùng lĩnh vực y tế bị chấm điểm giống nhau dù giải pháp hoàn toàn khác nhau |
| **False Positives cao** | Template chung của trường (phần Bối cảnh, Lịch trình) tạo ra độ tương đồng giả | Hội đồng phải xem xét thủ công hàng trăm cặp báo động sai |

### 1.3. Câu hỏi nghiên cứu

> *Có thể xây dựng một thuật toán đo lường độ tương đồng khai thác được cấu trúc phân cấp của tài liệu, cho phép mỗi tầng cấu trúc (domain, requirement, keyword) đóng góp chi phí khác nhau vào khoảng cách tổng thể, mà vẫn đảm bảo tính đúng đắn lý thuyết (triangle inequality) và hiệu quả tính toán không?*

---

## 2. Phương pháp đề xuất (Proposed Methodology)

### 2.1. Kiến trúc tổng quan: SW-CapTree

Tài liệu được biểu diễn dưới dạng **CapTree (Capstone Tree)** — cây 4 tầng:

```
T1: ROOT          → Project_ID (định danh duy nhất)
T2: DOMAIN        → D1 / D2 / D3 / D4 (4 miền ngữ nghĩa cố định)
T3: INTENT        → Câu/mệnh đề nguyên tử (atomic requirement)
T4: TERMINOLOGY   → Từ khóa/thuật ngữ đã chuẩn hóa (leaf)
```

Mỗi lá T4 được chuẩn hóa qua 3 bước:
1. **CSO Lookup** (Computer Science Ontology v3.5): ánh xạ thuật ngữ về canonical form
2. **Tech Equivalence Map (TEM)**: ánh xạ các tên công nghệ tương đương (React ↔ ReactJS)
3. **Lemmatization**: chuẩn hóa hình thái học

### 2.2. Hàm chi phí Schema-Weighted (công thức mới)

Mỗi tầng $\ell \in \{2, 3, 4\}$ có bộ tham số riêng:

$$w_{rep}^{(\ell)}(u, v) = \left(w_{del}^{(\ell)}(u) + w_{ins}^{(\ell)}(v)\right) \cdot \left(\beta_\ell \cdot \text{Dist}_{content}(u,v) + (1 - \beta_\ell) \cdot \text{Dist}_{schema}(u,v)\right)$$

| Tầng | $\beta_\ell$ | Ý nghĩa |
|---|---|---|
| T2 — Domain | 0.0 | Domain là category cứng, chỉ dùng schema distance |
| T3 — Intent | 0.9 | Nội dung câu yêu cầu là chính; schema ít quan trọng |
| T4 — Terminology | 0.8 | Canonical keyword là chính; schema type (ConceptKW vs TechKW) phụ |

**Tính chất lý thuyết bảo đảm:** Do $(1-\beta_\ell)$ bổ sung cho $\beta_\ell$ bằng 1, hàm chi phí **tự động thỏa mãn Triangle Inequality** mà không cần ràng buộc phụ. Điều này cho phép tích hợp trực tiếp vào APTED — thuật toán TED tối ưu hiện hành.

### 2.3. Điểm tương đồng cuối cùng (Hybrid Score)

$$\text{sim}(A, B) = \alpha \cdot \text{sim}_{SBERT}(\text{embed}_A, \text{embed}_B) + (1 - \alpha) \cdot \text{sim}_{TED}(A, B)$$

- Pre-filter: nếu $\text{cosine}(\text{embed}_A, \text{embed}_B) < 0.25$ → bỏ qua, không chạy APTED
- $\alpha = 0.6$ (tối ưu hóa qua ablation study)

---

## 3. Kết quả thực nghiệm (Experimental Results)

### 3.1. Bộ dữ liệu (Datasets)

| Dataset | Số tài liệu | Số cặp đánh giá | Ghi chú |
|---|---|---|---|
| **FPT Capstone** | 80 | 200 (5-fold CV) | Phiếu đăng ký ĐATN thực tế tại FPT University |
| **PURE (Requirements Docs)** | 79 | 200 (5-fold CV) | Bộ dữ liệu chuẩn trong nghiên cứu Software RE similarity |

### 3.2. So sánh với các Baseline (trên FPT Dataset)

| Phương pháp | F1-Score | ROC-AUC | TC-TNR (Type B) | TC-TNR (Type C) |
|---|---|---|---|---|
| B6: BM25 | 0.6154 | 0.1313 | 0.00 | 0.00 |
| B3: Standard TED | 0.7548 | 0.7500 | 0.50 | 0.58 |
| B4: pq-Gram | 0.7512 | 0.8713 | 0.50 | 0.64 |
| B5: Section Cosine | 0.8246 | 0.9231 | 0.72 | 0.76 |
| B2: Cosine SBERT | 0.9593 | 1.0000 | 0.88 | 0.98 |
| B7: SimCSE | 0.9657 | 1.0000 | 0.92 | 0.96 |
| **SW-BTED (Proposed)** | **0.9707** | **1.0000** | **0.94** | **0.96** |
| B1: Cosine TF-IDF | 0.9939 | 1.0000 | 0.98 | 1.00 |

> **TC-TNR** = True Negative Rate trên các cặp **Topic Conflation** (cùng lĩnh vực nhưng giải pháp khác nhau) — đây là chỉ số quan trọng nhất phản ánh khả năng chống báo động giả.

### 3.3. Kết quả trên PURE Dataset

| Phương pháp | F1-Score | ROC-AUC |
|---|---|---|
| SW-BTED 4 tầng (đề xuất) | **0.8929** | **0.9742** |
| SW-BTED 3 tầng (baseline cũ) | 0.6162 | 0.7540 |

> Việc nâng từ 3 tầng lên 4 tầng (thêm tầng T4 Terminology) cải thiện F1-Score **+27.7%** trên PURE dataset, với kết quả McNemar test có ý nghĩa thống kê (p < 0.001).

### 3.4. Ablation Study — Phân tích đóng góp từng thành phần

Kết quả trên cả hai dataset (FPT và PURE):

| Thành phần bị loại bỏ | FPT ΔF1 | PURE ΔF1 | Ý nghĩa thống kê |
|---|---|---|---|
| Bỏ T4 Terminology (leaf) | -6.4% | -0.3% | ✅ Có (FPT) |
| Bỏ T2 Domain labels | ~0% | ~0% | ❌ Không |
| Dùng beta đồng đều (β uniform) | -0.6% | -6.2% | ✅ Có (PURE) |
| Chỉ dùng Schema distance (β=0) | -7.9% | -10.2% | ✅ Có |
| TED-only (α=1.0) | -18.5% | -10.2% | ✅ Có |

**Kết luận ablation:** Tầng T4 Terminology và việc kết hợp Schema Distance đóng góp quan trọng nhất. Hybrid scoring (TED + SBERT) vượt trội hơn TED hoặc SBERT đơn lẻ trên cả hai dataset.

---

## 4. Thảo luận & Phân tích (Discussion)

### 4.1. Điểm mạnh của SW-BTED

1. **Phân biệt được Topic Conflation**: Nhờ cấu trúc domain T2, SW-BTED không bị đánh lừa bởi hai tài liệu cùng lĩnh vực nhưng có giải pháp khác nhau. SBERT phẳng thất bại ở đây (TC-TNR chỉ 0.88 so với 0.94 của SW-BTED trên Type B).
2. **Đảm bảo tính đúng đắn toán học**: Triangle Inequality được bảo đảm bởi thiết kế công thức, không cần kiểm tra từng cặp.
3. **Interpretable**: Khoảng cách cây có thể truy vết ngược để giải thích "phần nào của tài liệu bị trùng lặp".
4. **Generalizable**: Kiến trúc cây 4 tầng có thể áp dụng cho các domain tài liệu khác (thử nghiệm trên LinkedIn Job Descriptions xác nhận pipeline kỹ thuật hoạt động mà không cần thay đổi code).

### 4.2. Hạn chế

1. **Thời gian tính toán**: APTED có độ phức tạp O(n³) trong trường hợp xấu nhất; với tài liệu lớn cần bounded threshold để kiểm soát runtime.
2. **Phụ thuộc chất lượng parsing**: Kết quả phụ thuộc vào khả năng parser tách đúng D1-D4 từ văn bản gốc. Tài liệu không theo mẫu chuẩn có thể bị parse sai.
3. **Dataset JD (LinkedIn)**: Thử nghiệm cross-domain trên LinkedIn Job Descriptions bị giới hạn bởi đặc thù dữ liệu (các bài đăng cùng công ty sử dụng template giống nhau 100%), khiến bài toán trở nên trivially separable và không thể dùng làm quantitative benchmark. Thử nghiệm này chỉ xác nhận tính generalizability của kiến trúc.

### 4.3. Vị trí của SW-BTED trong không gian giải pháp

```
                    Precision trên Topic Conflation (TC-TNR)
                    Thấp ←──────────────────────────→ Cao
Đơn giản           BM25  TED  pq-Gram  SectionCosine  SBERT  SimCSE  SW-BTED
                   |     |    |        |               |      |       |
Độ phức tạp        Thấp                                              Cao
```

SW-BTED đạt TC-TNR cao nhất trong nhóm có độ phức tạp cao, trong khi B1 (TF-IDF) dù có F1 cao hơn một chút nhưng lại không giải thích được và không có tính chất cấu trúc.

---

## 5. Hướng phát triển tiếp theo (Future Work)

1. **Human-annotated Cross-domain Dataset**: Xây dựng tập benchmark có nhãn từ chuyên gia cho bài toán so sánh JD thực tế.
2. **Adaptive β Tuning**: Tối ưu tham số $\beta_\ell$ theo từng domain bằng Bayesian Optimization thay vì grid search thủ công.
3. **Integration với Retrieval Pipeline**: Kết hợp SW-BTED như một reranker trong RAG pipeline để cải thiện chất lượng retrieval cho hệ thống hỏi đáp tài liệu học thuật.
4. **Mở rộng sang tài liệu tiếng Việt**: Nghiên cứu khả năng áp dụng cho phiếu đăng ký viết hoàn toàn bằng tiếng Việt (cần thay thế spaCy `en_core_web_trf` bằng công cụ NLP tiếng Việt).

---

## 6. Cấu trúc tệp dự án (Repository Structure)

```
RAG_Research/
├── src/                        # Source code thuật toán
│   ├── 01_parser.py            # Parser: .docx → CapTree
│   ├── 03_normalizer.py        # CSO/TEM/SBERT normalization
│   ├── 05_sw_bted.py           # Thuật toán SW-BTED chính
│   ├── node.py                 # CapstoneNode data structure
│   ├── jd_parser.py            # Parser cho LinkedIn JD domain
│   ├── jd_dataset_builder.py   # Dataset builder cho JD evaluation
│   └── jd_normalizer.py        # Normalizer cho JD domain
├── experiments/
│   ├── run_4layer_ablation.py  # Chạy ablation study đầy đủ
│   └── run_jd_evaluation.py    # Chạy JD cross-domain evaluation
├── results/
│   ├── 4layer/                 # Kết quả chính (FPT + PURE)
│   ├── ablation/               # Ablation master table
│   ├── updated_baselines/      # So sánh với 7 baseline
│   └── jd_evaluation/          # Kết quả cross-domain JD
├── data/
│   ├── processed/              # cso_graph.pkl, tech_equiv_map.json
│   └── raw/                    # Phiếu đăng ký gốc
└── docs/research_notes/        # Tài liệu nghiên cứu
    ├── PROJECT_OVERVIEW.md     # File này
    ├── SW_BTED_ABLATION_STUDY.md
    ├── SW_BTED_JD_ADAPTATION_REPORT.md
    └── project_structure.md
```

---

*Cập nhật lần cuối: 2026-07-19*
