# Ghi chú thảo luận — 2026-07-19

> **Phạm vi:** Đánh giá kế hoạch mở rộng dataset (`AGENT_TASKS_dataset_expansion.md`) và phân tích số liệu ablation study hiện có.

---

## 1. Tình trạng Dataset-3 (LinkedIn Job Descriptions)

### Kết luận đã đồng ý
Dataset LinkedIn JD **không thể dùng làm quantitative benchmark** vì đặc thù của dữ liệu:
- Positive pairs (cùng công ty + cùng chức danh) = các bài đăng repost sử dụng chung template → similarity ~0.99–1.0
- Negative pairs (khác công ty, khác ngành) = do HR teams khác nhau viết → similarity ~0.33
- Khoảng cách quá lớn → mọi phương pháp (kể cả Lexical Jaccard thô sơ) đạt F1 = 1.0

Đây **không phải lỗi code** mà là đặc tính nội tại của LinkedIn dataset. Các cách xử lý đã thử (filter `job_id != job_id`, filter `description != description`) đều không giải quyết được vì template text vẫn giống sau normalization.

### Hướng xử lý đã thống nhất
Trình bày Dataset-3 như một **Qualitative Generalizability Test**, không phải benchmark định lượng:
> *"Pipeline SW-CapTree đã xử lý thành công 42,000+ mô tả tuyển dụng mà không cần thay đổi code. Tuy nhiên, do đặc thù template-reuse của LinkedIn, bộ dữ liệu này bị linearly separable và không phù hợp làm benchmark. Future work sẽ dùng human-annotated JD pairs."*

---

## 2. Phân tích kế hoạch mở rộng dataset (`AGENT_TASKS_dataset_expansion.md`)

### Task 0 — Statistical Significance
- ✅ **Nên làm ngay, không có gì để thảo luận thêm**
- Cần thêm p-value / 95% CI vào mọi bảng kết quả (FPT, PURE)
- Không pass peer review nếu không có significance testing

### Task 1 — PAN-PC Plagiarism Corpus
- ⚠️ **Cần làm rõ trước khi triển khai**
- Câu hỏi chưa trả lời: ưu thế của SW-BTED đến từ **T2 Domain structure** hay **T4 CSO/TEM normalization**?
- Nếu từ T4 → PAN-PC có thể chạy được (Option A: 3 tầng không có T2)
- Nếu từ T2 → PAN-PC không có T2 grounded schema, thử nghiệm sẽ không chứng minh được gì mới
- **Số liệu ablation (xem Phần 3)** sẽ trả lời câu hỏi này

### Task 2 — Bug Reports (GitBugs/BugRepo)
- ✅ **Task tốt nhất trong danh sách, nên ưu tiên**
- Ground truth từ developer annotation (không tự gán nhãn)
- Schema T2 grounded trong **Bettenburg et al. 2008** (citable): D1=Problem, D2=Reproduction, D3=Environment, D4=Evidence
- Số lượng pairs lớn (hàng nghìn) → loại bỏ small-N criticism
- Cấu trúc document tương đồng với FPT Capstone → khả năng cao SW-BTED sẽ hoạt động tốt

### Task 3 — CUAD (Legal Contracts)
- ⚠️ **Có vấn đề trong spec cần làm rõ**
- Spec viết T3 = "41 clause categories" nhưng đây là **label metadata**, không phải câu văn bản
- Ba hướng triển khai:

| Hướng | T3 là gì | T4 là gì | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| **A (theo spec)** | Clause category label | Extracted terms từ clause text | Đơn giản, T2-T3 rõ ràng | Cây quá nông, TED dominated bởi structure matching thay vì content |
| **B (recommended)** | Câu văn bản trong clause | Legal terms extracted | Đúng tinh thần SW-BTED nhất | Cần build legal TEM từ đầu |
| **C (pragmatic)** | Clause category label | Bỏ qua T4; dùng SBERT(clause_text) cho Dist_content(T3) | Không cần legal TEM, vẫn capture semantic variance | Mất T4 ablation; ít interpretable |

- **Khuyến nghị hiện tại:** Hướng C nếu muốn nhanh, Hướng B nếu muốn đúng phương pháp
- **Chưa quyết định** — cần thảo luận thêm

---

## 3. Kết quả Ablation Study hiện có — Phân tích đúng

> **Nguồn dữ liệu:** `results/ablation/ablation_master_table.csv` (và bản sao tại `results/4layer/ablation/ablation_master_table.csv` — hai file giống hệt nhau)

### Group A — Layer Ablation

| Variant | FPT F1 | ΔF1 FPT | p-value FPT | Sig? | PURE F1 | ΔF1 PURE | p-value PURE | Sig? |
|---|---|---|---|---|---|---|---|---|
| **A1 — Full 4L (base)** | 0.9939 | — | — | — | 0.8684 | — | — | — |
| A2 — Bỏ T4 (no Terminology) | 0.9296 | **-6.4%** | 0.00098 | ✅ | 0.8714 | +0.3% | 1.0 | ❌ |
| A3 — Bỏ T2 (no Domain labels) | 0.9939 | **0.0%** | 1.0 | ❌ | 0.8739 | +0.5% | 0.664 | ❌ |
| A4 — 3L legacy | 0.8289 | -16.5% | <0.001 | ✅ | 0.6162 | -25.2% | <0.001 | ✅ |

### Kết luận đúng từ số liệu (đính chính nhầm lẫn trước đây)

**T4 Terminology:**
- Có impact **có ý nghĩa thống kê trên FPT** (ΔF1 = -6.4%, p < 0.001)
- **Không có impact trên PURE** (ΔF1 = +0.3%, không có ý nghĩa)
- → T4 đóng góp thực sự vào performance, không chỉ là explainability

**T2 Domain:**
- **Không có impact đo được trên cả hai dataset** (ΔF1 = 0% FPT, +0.5% PURE không có ý nghĩa)
- → T2 không đóng góp vào F1 score, nhưng vẫn cần thiết về mặt thiết kế (cho phép schema distance trong cost function)

> ⚠️ **Điểm bạn nhớ nhầm:** Trong báo cáo nếu có ghi "T4 chỉ đóng góp vào explainability" thì cần đính chính — điều này chỉ đúng với PURE, không đúng với FPT. Cần cập nhật lại nhận xét này trong paper.

### Group B — Cost Function Ablation (tóm tắt)

| Variant | FPT F1 | ΔF1 | Sig? | PURE F1 | ΔF1 | Sig? |
|---|---|---|---|---|---|---|
| B1 — β per-layer (full) | 0.9939 | — | — | 0.8684 | — | — |
| B2 — β uniform (0.5 all) | 0.9879 | -0.6% | ❌ | 0.8061 | **-6.2%** | ✅ |
| B3 — Content only (β=1) | 0.9939 | 0.0% | ❌ | 0.8735 | +0.5% | ❌ |
| B4 — Schema only (β=0) | 0.9151 | **-7.9%** | ✅ | 0.7661 | **-10.2%** | ✅ |

**Kết luận:** β per-layer quan trọng nhất trên PURE. Schema distance (T2) đóng góp quan trọng thông qua **cost function (B4)** dù không đo được qua layer removal (A3). Đây là điểm tinh tế cần làm rõ trong paper.

---

## 4. Các câu hỏi còn chưa trả lời — Cần quyết định

| # | Câu hỏi | Ghi chú |
|---|---|---|
| 1 | Ưu thế SW-BTED đến từ T4 CSO/TEM hay T2 schema structure? | Quan trọng vì ảnh hưởng đến việc có nên làm Task 1 (PAN-PC) không. Số liệu A2 vs A3 gợi ý T4, nhưng B4 gợi ý schema distance cũng quan trọng |
| 2 | Task 3 (CUAD): chọn Hướng A, B hay C? | Hướng B đúng phương pháp nhất nhưng cần nhiều công nhất (legal TEM). Hướng C nhanh nhất |
| 3 | Thứ tự ưu tiên: làm Task 0, Task 2, Task 3 trước? | Task 2 (Bug Reports) được đánh giá là ưu tiên cao nhất vì fix được nhiều vấn đề nhất |

---

## 5. Action Items tiếp theo (đề xuất)

- [ ] **Đính chính nhận xét về T4** trong paper/báo cáo: T4 có impact có ý nghĩa trên FPT (không chỉ là explainability)
- [ ] **Quyết định** hướng triển khai Task 3 CUAD (A/B/C)
- [ ] **Bắt đầu Task 0** (significance testing) — không phụ thuộc vào quyết định nào khác
- [ ] **Cập nhật báo cáo Dataset-3** để framing là Qualitative Generalizability Test
