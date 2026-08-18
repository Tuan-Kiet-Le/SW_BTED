# Kế Hoạch Hoàn Thiện Bài Báo SW-BTED & Nộp Hội Nghị/Tạp Chí Q2

> **Ngày lập kế hoạch:** 29/07/2026  
> **Trạng thái:** Giai đoạn 1 HOÀN THÀNH 100% (Đã khóa số liệu) $\rightarrow$ Đang chuyển sang Giai đoạn 2.  

---

## Giai đoạn 1 — Hoàn thiện thực nghiệm (ĐÃ HOÀN THÀNH 100% — HOÀN HẢO)

### 1.1. Structural-Perturbation Benchmark (Đã hoàn thành — Chiến thắng $p = 1.91 \times 10^{-6}$)
- [x] Xây dựng 20 cặp thử nghiệm xáo trộn cấu trúc (Section Reordering $D_2 \leftrightarrow D_3$).
- [x] Chạy SW-BTED Structural-Only, Full-Doc SBERT, và Hybrid Mode trên các cặp này.
- [x] **Kết quả:** Full-Doc SBERT thất bại $100\%$ (FPR $100.0\%$), SW-BTED Structural-Only đạt độ chính xác tuyệt đối $100\%$ (FPR $0.0\%$). Phép thử McNemar đạt chiến thắng có ý nghĩa thống kê cực kỳ rõ ràng ($p = 1.9073 \times 10^{-6} < 0.001$).

### 1.2. Baseline hiện đại (`BAAI/bge-small-en-v1.5`) (Đã hoàn thành)
- [x] Đánh giá đối chiếu mô hình nhúng hiện đại `bge-small-en-v1.5` trên 138 cặp FPT Real-only.
- [x] Chạy lại pipeline significance-testing: BGE-v1.5 đạt $F1 = 0.9870$, đạt hòa thống kê ($p = 0.3750$) với SW-BTED Structural-Only ($F1 = 0.9498$).

### 1.3. Khóa toàn bộ số liệu cuối cùng (Data Frozen)
- [x] Xuất báo cáo tổng kết 📄 [BAO_CAO_GIAI_DOAN_1_PHASE1_REPORT.md](file:///d:/FPT/Semester_8/SW_BTED_v2/reports/audit/BAO_CAO_GIAI_DOAN_1_PHASE1_REPORT.md).
- [x] Đối chiếu mọi con số với `PATH_TO_Q2_SUBMISSION.md` và khóa số liệu chính thức trước khi bắt đầu Giai đoạn 2.

---

## Giai đoạn 2 — Viết bài báo (~2-3 tuần, đang tiến hành)

- [ ] **Framing trung tâm:** bài báo nói về "structure + limits" — SW-BTED đóng góp gì ngoài embedding phẳng, và giới hạn ở đâu — không phải "SW-BTED đánh bại SOTA."
- [ ] Sử dụng `DRAFT_RESULTS_AND_LIMITATIONS.md`, `RESEARCH_BRIEF_SW-BTED.md`, và `SW_BTED_DEFENSE_QNA.md` làm khung sườn cho Results/Discussion/Limitations.
- [ ] Bổ sung phần Structural-Perturbation Benchmark result ($p = 1.91 \times 10^{-6}$) vào bài báo.
- [ ] Bổ sung phần Related Work so sánh với tree-kernel và structure-aware embedding gần đây.
- [ ] Viết Contributions dựa đúng theo bảng đã verify — không thêm bất kỳ claim nào mạnh hơn bằng chứng hiện có.
- [ ] Chuẩn bị reproducibility package (repo sạch, code significance-testing công khai).

---

## Giai đoạn 3 — Đề xuất Hội nghị / Tạp chí Q2 phù hợp

Dựa trên nội dung bài báo, có 2 nhóm lựa chọn:

### Hội nghị (có deadline cụ thể — đã kiểm tra trực tiếp)

| Hội nghị | Xếp hạng | Deadline nộp bài | Ngày tổ chức | Phù hợp vì |
|---|---|---|---|---|
| **SANER 2027** | CCF B / CORE A | **21/09/2026** | 09/03/2027, Richmond, VA, USA | Rất phù hợp với phần bug-report cross-domain |
| **ECIR 2027** | CCF C / CORE A2 | **02/10/2026** | 21-25/03/2027, Southampton, UK | Phù hợp với phần document similarity / Reproducibility track |

### Tạp chí (rolling submission — ưu tiên chọn)

| Tạp chí | Ghi chú |
|---|---|
| **Journal of Systems and Software** (Elsevier) | Phù hợp phạm vi rộng về software engineering + tool/method mới. |
| **Software Quality Journal** (Springer) | Phù hợp nếu framing nghiêng về "chất lượng/độ tin cậy của phương pháp đo lường". |
| **Journal of Software: Evolution and Process** (Wiley) | Phù hợp với phần cross-domain generalization. |

---

## Dòng thời gian cập nhật

| Tuần | Công việc | Trạng thái |
|---|---|---|
| Tuần 1 (29/07) | Xây dựng và chạy Structural-Perturbation Benchmark & Modern Baseline | ✅ **HOÀN THÀNH 100%** |
| Tuần 2-4 (30/07 – 20/08) | Viết bản thảo đầy đủ (Intro → Method → Results → Discussion → Limitations) | ⏳ **ĐANG THỰC HIỆN** |
| Tuần 5 (20/08 – 27/08) | Sửa bản thảo, chuẩn bị reproducibility package | ⏳ Chờ thực hiện |
| Sau tuần 5 | Nộp Tạp chí Q2/Q1 (Rolling submission) hoặc SANER/ECIR 2027 | ⏳ Chờ thực hiện |
