# Tài liệu Nghiên cứu: Đánh giá Trùng lặp & Phát hiện Đạo văn Đồ án Tốt nghiệp (Capstone Project)

Tài liệu này ghi nhận bối cảnh, mô tả bài toán, hiện trạng hệ thống (Baseline) và các vấn đề cần giải quyết trong nghiên cứu đánh giá trùng lặp đồ án tốt nghiệp.

---

## 1. Bài toán (Problem Statement)
**Mục tiêu:** Đánh giá tỷ lệ trùng lặp (%) và phát hiện đạo văn giữa các file đăng ký Đồ án tốt nghiệp (Capstone Project) viết bằng tiếng Anh tại một trường đại học.

## 2. Cấu trúc dữ liệu đầu vào (Input Data Structure)
Dữ liệu đầu vào là các file có cấu trúc bán định dạng (Semi-structured), phân cấp rõ ràng theo biểu mẫu tiêu chuẩn của nhà trường:
*   **Mục 3.1:** Context (Bối cảnh), Requirements (Yêu cầu).
*   **Mục 3.2:** Proposed Solutions (Giải pháp đề xuất), Applied Theory (Lý thuyết áp dụng), Deliverables (Sản phẩm bàn giao).
*   **Mục 3.3:** Research Problem (Vấn đề nghiên cứu), Methodology (Phương pháp nghiên cứu), Contributions (Đóng góp khoa học).

## 3. Hệ thống hiện tại (Baseline System)
*   **Phương pháp:** Sử dụng **Single-Vector Embedding** (biến nguyên cả bài đồ án thành một vector duy nhất).
*   **Lưu trữ:** Lưu trữ trong cơ sở dữ liệu PostgreSQL sử dụng extension `pgvector`.
*   **Độ đo tương đồng:** Sử dụng **Cosine Similarity** để so sánh khoảng cách giữa các vector.

---

## 4. Các hạn chế và Lý do kết quả bị sai
Hệ thống hiện tại (Baseline) gặp phải hai vấn đề lớn khiến kết quả đánh giá không chính xác:

### Vấn đề 1: Thông tin bị loãng (Information Loss)
*   Một vector biểu diễn (thường từ 768 đến 1536 chiều) không thể nén đầy đủ nội dung của một văn bản học thuật dài mà không làm mất đi các chi tiết cốt lõi.
*   Việc nén toàn bộ văn bản vào một vector duy nhất làm lu mờ các phần chi tiết kỹ thuật quan trọng ở các mục nhỏ.

### Vấn đề 2: Báo động giả (High False Positives)
*   Hệ thống "cào bằng" (flatten) toàn bộ văn bản và không phân biệt vai trò của từng mục.
*   **Hệ quả:** Hai đề tài có phần bối cảnh giống nhau (ví dụ: cùng giải quyết bài toán quản lý bệnh viện hoặc cùng sử dụng chung hệ thuật ngữ chuyên ngành) sẽ bị chấm điểm trùng lặp rất cao. Điều này xảy ra ngay cả khi giải pháp kỹ thuật, thuật toán và phương pháp tiếp cận ở Mục 3.2 và 3.3 của chúng khác nhau hoàn toàn.

---

## 5. Hướng tiếp cận đề xuất (Proposed Solutions)
Để khắc phục các hạn chế trên, hướng đi tiếp theo của nghiên cứu có thể tập trung vào:
1.  **Phân mảnh theo cấu trúc (Structural Chunking):** Tách biệt các mục 3.1, 3.2, 3.3 thành các đoạn độc lập thay vì gộp chung.
2.  **Đánh trọng số tương đồng (Weighted Similarity):** Áp dụng trọng số khác nhau cho các mục. Ví dụ: Sự trùng lặp ở phần giải pháp (Mục 3.2) và phương pháp (Mục 3.3) cần được đánh trọng số cao hơn nhiều so với phần bối cảnh (Mục 3.1) khi tính điểm đạo văn tổng thể.
3.  **Tìm kiếm đa Vector (Multi-Vector / Hierarchical Search):** Biểu diễn mỗi tài liệu bằng tập hợp nhiều vector tương ứng với các phân đoạn cấu trúc của nó.

---

## 6. Hướng thực nghiệm đề xuất: Thuật toán SW-BTED (Schema-Weighted Bounded Tree Edit Distance)

### 6.1. Hạn chế của các phương pháp hiện tại (Limitations of Current Methods)
Các thuật toán Tree Edit Distance (TED) giới hạn hiện nay, bao gồm cả thuật toán State-Of-The-Art (SOTA) mới nhất tại hội nghị ESA 2025, đều giả định rằng hàm chi phí hiệu chỉnh $w(e)$ giữa các nút là đồng đều hoặc tuân theo một hệ thống trọng số tĩnh, độc lập với ngữ cảnh vị trí của nút trong lược đồ tài liệu. 

Tuy nhiên, trong tài liệu đăng ký đồ án tốt nghiệp, mức độ quan trọng của các nút cấu trúc là hoàn toàn khác biệt:
*   Nếu một sinh viên sao chép nguyên vẹn cấu trúc của chương *"Methodology"* hoặc *"Proposed Solution"* từ một đồ án khác, mức độ nghiêm trọng và khả năng xảy ra đạo văn là cực kỳ cao.
*   Ngược lại, việc trùng khớp cấu trúc ở các chương phụ trợ như *"References"* hay *"Gantt Chart"* thường chỉ là do sử dụng chung một biểu mẫu (template) quy định bởi nhà trường. 
*   Nếu áp dụng TED thông thường, các trùng khớp mẫu này sẽ tạo ra độ tương đồng giả (**false positives**), hoặc ngược lại, che mờ đi sự đạo văn tinh vi ở các phần cốt lõi.

### 6.2. Nội dung cải tiến đề xuất (Proposed Contributions)
Nghiên cứu này đề xuất thiết lập một mô hình biểu diễn tài liệu dưới dạng cấu trúc cây gán nhãn mới, trong đó mỗi nút $u$ được ánh xạ trực tiếp với một lớp ngữ cảnh lược đồ tài liệu $C(u)$ (ví dụ: $C(u) \in \{\text{Abstract}, \text{Methodology}, \text{Timeline}, \dots\}$). 

Chúng ta định nghĩa một **Hàm trọng số động theo lược đồ (Schema-Aware Dynamic Weighting Function)** $W_{schema}$ cho ba phép hiệu chỉnh cơ bản:

1.  **Chi phí xóa nút $u$ ($w_{del}$):**
    $$w_{del}(u) = \alpha \cdot \text{IDF}(C(u)) + (1 - \alpha) \cdot \frac{1}{\text{Depth}(u)}$$

2.  **Chi phí chèn nút $v$ ($w_{ins}$):**
    $$w_{ins}(v) = \alpha \cdot \text{IDF}(C(v)) + (1 - \alpha) \cdot \frac{1}{\text{Depth}(v)}$$

3.  **Chi phí thay thế nhãn nút $u$ bằng $v$ ($w_{rep}$):**
    $$w_{rep}(u,v) = \beta \cdot |w_{del}(u) - w_{ins}(v)| + \gamma \cdot \text{Dist}_{lex}(\text{label}(u), \text{label}(v))$$

*Trong đó:*
*   $\text{IDF}(C(u))$ là tần suất nghịch đảo của lớp tài liệu $C(u)$ tính trên toàn bộ cơ sở dữ liệu đồ án của nhà trường để giảm thiểu trọng số của các nút biểu mẫu phổ biến.
*   $\text{Depth}(u)$ là độ sâu của nút trên cây. 
    > [!NOTE]
    > *Lưu ý về mặt công thức:* Trong công thức trên, số hạng tỷ lệ nghịch $\frac{1}{\text{Depth}(u)}$ sẽ làm giảm chi phí xóa/chèn khi nút càng sâu. Nếu muốn các nút ở sâu hơn (mô tả chi tiết thuật toán/mục tiêu cụ thể) có chi phí hiệu chỉnh lớn hơn (mang tính đặc trưng cao hơn), công thức thực tế trong thực nghiệm có thể cần điều chỉnh thành tỷ lệ thuận với $\text{Depth}(u)$ thay vì tỷ lệ nghịch, hoặc định nghĩa lại độ sâu ngược từ lá lên.
*   $\text{Dist}_{lex}$ là khoảng cách từ vựng phi AI (như Jaro-Winkler hoặc Levenshtein) áp dụng trên nhãn nội dung văn bản của nút để bắt được sự thay đổi danh từ diễn đạt (paraphrasing/synonyms).

### 6.3. Đóng góp lý thuyết (Theoretical Contributions)
*   **Chứng minh toán học về tính chất Quasimetric của hàm trọng số động $W_{schema}$:** 
    Đóng góp lý thuyết quan trọng nhất là đưa ra chứng minh toán học chặt chẽ rằng hàm trọng số động $W_{schema}$ (tích hợp tần suất nghịch đảo lớp tài liệu IDF và độ sâu của nút $\text{Depth}(u)$) vẫn bảo toàn đầy đủ các tiên đề của một quasimetric (đặc biệt là bất đẳng thức tam giác định hướng).
*   **Bảo toàn độ phức tạp tối ưu trong cấu trúc hạt nhân hóa:** 
    Nhờ chứng minh được tính chất quasimetric, nghiên cứu chỉ ra rằng bài toán SW-BTED hoàn toàn có thể ánh xạ vào khung thuật toán của Kociumaka và Shahali (ESA 2025). Từ đó chứng minh được rằng việc gán trọng số động theo lược đồ không làm tăng độ phức tạp thời gian, thuật toán vẫn chạy ở mức tối ưu $O(n + k^6 \log k)$.


