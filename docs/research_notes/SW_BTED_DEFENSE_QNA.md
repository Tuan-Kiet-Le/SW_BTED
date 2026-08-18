# SW-BTED: Bộ Câu Hỏi & Chiến Lược Phản Bỏ Trả Lời Hội Đồng (Defense Q&A Strategy)

> **Mục đích:** Tài liệu này tổng hợp 6 câu hỏi phản biện cốt lõi (Reviewer / Defense Committee Questions) về mặt kiến trúc, lý thuyết và tính tổng quát của hệ thống SW-BTED, kèm theo câu trả lời được bảo vệ bằng lập luận khoa học và kết quả thực nghiệm.

---

## 📋 TỔNG QUAN 6 CÂU HỎI PHẢN BIỆN CỐT LÕI

1. **Why APTED? Why not Graph Matching?** (Tại sao lại dùng thuật toán Cây APTED mà không dùng Khớp đồ thị Graph Matching?)
2. **Why Tree? Requirements are not naturally trees.** (Tại sao lại biểu diễn dưới dạng Cây? Văn bản Yêu cầu Phần mềm tự nhiên đâu phải là Cây?)
3. **Why Fixed Taxonomy?** (Tại sao lại dùng Phân vùng miền cố định $D_1 \dots D_4$?)
4. **Why SBERT instead of Modern Embedding / LLM?** (Tại sao lại dùng SBERT `all-MiniLM-L6-v2` mà không dùng mô hình nhúng hiện đại như GPT-4, BGE-m3, Llama-3?)
5. **Why Manual Weights?** (Tại sao lại dùng trọng số thủ công/grid search $\beta_\ell, \alpha$ mà không dùng Deep Learning học tự động?)
6. **Generalization? Nếu sang Healthcare/Legal thì Taxonomy còn đúng không?** (Khả năng tổng quát hóa ra sao? Nếu chuyển sang miền Yêu cầu Y tế hay Hợp đồng Pháp lý thì thế nào?)

---

## ⚔️ CHI TIẾT CÂU TRẢ LỜI & CHIẾN LƯỢC BẢO VỆ

### 1. Why APTED? Why not Graph Matching?

#### ❓ Câu hỏi của Hội đồng:
*"Tại sao bạn lại chọn thuật toán APTED (All-Pairs Tree Edit Distance) để tính khoảng cách chỉnh sửa cây? Tại sao không biểu diễn tài liệu dưới dạng Đồ thị (Graph) và dùng Graph Matching / Graph Edit Distance (GED)?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Độ phức tạp tính toán (Computational Complexity):**
   * Bài toán **Graph Edit Distance (GED)** hoặc Graph Isomorphism trên đồ thị tổng quát là bài toán **NP-hard**. Khi số lượng node tăng lên, các phương pháp khớp đồ thị (Graph Matching) buộc phải dùng thuật toán xấp xỉ (Heuristic) hoặc Neural Graph Networks (GNN), dẫn đến việc **mất đi thuộc tính khoảng cách khoảng cách metric chuẩn**.
   * Trong khi đó, **APTED** (Pawlik & Augsten) là thuật toán tối ưu nhất hiện nay cho Tree Edit Distance với độ phức tạp tối đa $O(n^3)$ và bộ nhớ $O(n^2)$, cho phép tính toán **khoảng cách chính xác 100% (Exact Edit Distance)** trong thời gian thực.

2. **Bảo toàn thuộc tính Không gian Metric (Metric Preserving Property):**
   * Thuật toán APTED khi kết hợp với hàm chi phí lồi của SW-BTED ($w_{rep}^{(\ell)}$) đảm bảo đầy đủ các thuộc tính của một **Metric Space**:
     * *Tính không âm (Non-negativity):* $D(A, B) \ge 0$
     * *Tính đối xứng (Symmetry):* $D(A, B) = D(B, A)$
     * *Bất đẳng thức tam giác (Triangle Inequality):* $D(A, C) \le D(A, B) + D(B, C)$
   * Thuộc tính này là điều kiện tiên quyết để hệ thống thực hiện bước **Pre-filtering (T1)** và thiết lập ngưỡng chặn (Bounded threshold $k$) một cách hợp lệ về mặt toán học. Các mô hình Graph Matching học sâu (Neural GED) thường **vi phạm bất đẳng thức tam giác**, khiến khoảng cách bị méo mó.

3. **Tính phân cấp tự nhiên của văn bản:**
   * Cấu trúc văn bản Yêu cầu phần mềm có tính thứ tự và phân cấp từ trên xuống dưới (Document $\rightarrow$ Domain $\rightarrow$ Requirement $\rightarrow$ Term). Cấu trúc cây là mô hình toán học **vừa đủ (Sufficient)** và **tối ưu chi phí** để biểu diễn phân cấp này mà không cần đến sự phức tạp của đồ thị có chu trình (Cyclic Graph).

---

### 2. Why Tree? Requirements are not naturally trees.

#### ❓ Câu hỏi của Hội đồng:
*"Trong thực tế, các yêu cầu phần mềm (Requirements) có mối quan hệ chéo phức tạp (Cross-cutting concerns, phụ thuộc lẫn nhau). Bản chất Yêu cầu đâu phải là Cây (Tree)? Tại sao lại ép nó vào cấu trúc Cây?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Phân biệt giữa 'Dependency Graph' và 'Document Structural Hierarchy':**
   * **Đúng:** Nếu xét về mặt *mối liên hệ phụ thuộc chức năng (Functional Dependencies)* giữa các tính năng (VD: Feature A gọi API của Feature B), hệ thống là một Đồ thị (Graph).
   * **Tuy nhiên:** SW-BTED **không bài toán phân tích phụ thuộc mã nguồn**, mà là bài toán **Đánh giá độ tương đồng cấu trúc tài liệu đăng ký/đề xuất (Registration Document Similarity)**. 
   * Trong các tiêu chuẩn tài liệu kỹ thuật (IEEE 830, ISO/IEC/IEEE 29148, Đề xuất Capstone FPT), tài liệu **bắt buộc được trình bày theo cấu trúc phân cấp cây chuẩn**:
     $$\text{Tài liệu (T1)} \longrightarrow \text{Các Phân miền/Section (T2)} \longrightarrow \text{Các Câu yêu cầu nguyên tử (T3)} \longrightarrow \text{Các Thuật ngữ/Kỹ thuật (T4)}$$

2. **Cây SW-CapTree là một 'Abstraction Schema' có chủ đích:**
   * Hệ thống không dựng cây ngây thơ từ Parse Tree ngữ pháp (Cây cú pháp vốn rất nhiễu), mà xây dựng một **Cây nhãn cấu trúc 4 tầng (SW-CapTree)**.
   * Cấu trúc cây này giúp bài toán có **tính giải trình cao (Interpretability)**: Cho phép xuất vết (Traceability) chính xác đến từng tầng: miền nào lệch ($T_2$), câu nào bị sửa/thêm/xóa ($T_3$), từ khóa công nghệ nào bị thay thế ($T_4$). Việc này trên Đồ thị phi cấu trúc sẽ tạo ra các ma trận kề rất khó giải thích cho con người.

---

### 3. Why Fixed Taxonomy?

#### ❓ Câu hỏi của Hội đồng:
*"Tại sao tầng T2 lại dùng Phân vùng miền cố định (Fixed Taxonomy với 4 miền D1, D2, D3, D4)? Dùng Taxonomy cố định có làm mất tính linh hoạt không?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Cơ sở lý thuyết vững chắc từ Kỹ nghệ Yêu cầu (RE Theory):**
   * 4 miền ngữ nghĩa cố định ($D_1$: Business Context, $D_2$: Functional Requirements, $D_3$: Technical Realization, $D_4$: Execution Planning) được kế thừa trực tiếp từ các khung chuẩn Yêu cầu Phần mềm quốc tế (**IEEE 830 / ISO 29148**) và quy định đăng ký đề tài Capstone Proposal.

2. **Giải quyết triệt để lỗi 'Topic Conflation' (Lỗi trùng lặp từ vựng chủ đề):**
   * Đây là đóng góp lý thuyết quan trọng nhất của T2. Nếu không có Taxonomy cố định, các phương pháp nhúng phẳng (Flat Embeddings) sẽ gộp chung từ vựng miền Bối cảnh ($D_1$) và miền Chức năng ($D_2$).
   * *Ví dụ thực tế:* Hai đề tài *"App chăm sóc người già"* và *"Nền tảng chăm sóc thú cưng"* có chung từ vựng bối cảnh ("chăm sóc", "người dùng", "thanh toán"). Nếu không chia miền cố định, các phương pháp phẳng cho Similarity rất cao $\rightarrow$ **False Positive**.
   * Nhờ có **Fixed Taxonomy (T2)**, SW-BTED ép $D_1$ so sánh với $D_1$, $D_2$ so sánh với $D_2$. Sự lệch pha ở miền Chức năng ($D_2$) sẽ lập tức bị phát hiện và hạ điểm similarity chung, ngăn chặn báo động giả.

3. **Taxonomy cố định theo Miền (Domain-specific), không phải cố định vĩnh viễn:**
   * Khung $D_1 \dots D_4$ cố định cho thể loại **Software Capstone Proposals**. Khi chuyển sang thể loại khác (ví dụ: Bug Reports), Taxonomy sẽ được cấu hình lại tương ứng với thể loại đó (xem Câu 6).

---

### 4. Why SBERT instead of Modern Embedding / LLM?

#### ❓ Câu hỏi của Hội đồng:
*"Tại sao ở tầng T3 lại dùng SBERT model cũ (`all-MiniLM-L6-v2`) mà không dùng các mô hình nhúng mới nhất như OpenAI `text-embedding-3-small`, BGE-m3, hay Llama-3 Embeddings?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Tính Mô-đun hóa (Plug-and-Play Encoder Architecture):**
   * Đóng góp cốt lõi của SW-BTED là **Kiến trúc cây phân cấp 4 tầng và Hàm chi phí gán trọng số Schema ($w_{rep}^{(\ell)}$)**, chứ KHÔNG phụ thuộc vào một Encoder cụ thể nào. SBERT chỉ đóng vai trò là một mô-đun sinh Dense Embedding ở tầng T3 ($Dist_{content} = 1 - \text{cosine}(Emb_u, Emb_v)$).
   * Bạn hoàn toàn có thể thay `all-MiniLM-L6-v2` bằng `BGE-m3` hay `OpenAI Embedding` mà **không cần thay đổi bất kỳ dòng code nào** trong thuật toán chỉnh sửa cây SW-BTED.

2. **Thực nghiệm chứng minh hiệu năng và chi phí (Efficiency & Cost-Benefit):**
   * Mô hình `all-MiniLM-L6-v2` có kích thước cực nhẹ (384 chiều), chạy suy luận 100% cục bộ (Local CPU) trong vài mili-giây, chi phí $= 0$ VNĐ, không phụ thuộc kết nối Internet hay Cloud API.
   * Ở cấp độ câu yêu cầu nguyên tử ($T_3$), việc so sánh paraphrase tiếng Anh kỹ thuật tiêu chuẩn bằng SBERT đã đạt độ chính xác $98.7\%$. Việc dùng các LLM Embedding đắt đỏ (gấp 100x thời gian và chi phí) **không đem lại sự cải thiện có ý nghĩa thống kê** cho bài toán khớp ý định câu đơn lẻ.

---

### 5. Why Manual Weights?

#### ❓ Câu hỏi của Hội đồng:
*"Tại sao các siêu tham số ($\alpha = 0.60$, $\beta_2=0.0, \beta_3=0.9, \beta_4=1.0$, $w_{del}, w_{ins}$) lại được ấn định/tìm bằng Grid Search thủ công mà không dùng Deep Learning (ví dụ: Neural Edit Distance) để mô hình tự học trọng số?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Đảm bảo tính chất Bất đẳng thức Tam giác (Metric Guarantees):**
   * Trong công thức SW-BTED:
     $$w_{rep}^{(\ell)}(u,v) = \left(w_{del}^{(\ell)}(u) + w_{ins}^{(\ell)}(v)\right) \cdot \left(\beta_\ell \cdot \text{Dist}_{content} + (1-\beta_\ell) \cdot \text{Dist}_{schema}\right)$$
   * Việc đặt $\beta_\ell \in [0, 1]$ và dùng $(1 - \beta_\ell)$ cho Schema Distance **tự động đảm bảo về mặt toán học** rằng:
     $$w_{rep}^{(\ell)}(u,v) \le w_{del}^{(\ell)}(u) + w_{ins}^{(\ell)}(v)$$
   * Các mô hình Deep Learning tự do học trọng số thường **vi phạm điều kiện lồi này**, dẫn đến việc thuật toán APTED bị suy biến (chọn xóa+thêm thay vì thay thế cho mọi node), làm cây mất đi ý nghĩa cấu trúc.

2. **Quy trình tối ưu tham số khách quan (5-Fold Stratified Cross-Validation):**
   * Các tham số $\alpha, \beta_\ell$ **không phải chọn tính cảm quan**, mà được tối ưu bằng **Grid Search trên tập Validation** qua 5-Fold CV (xem báo cáo Ablation Study Group D).
   * Việc dùng trọng số được tối ưu qua Grid Search đảm bảo **tính minh bạch (Transparency)** và **tính giải trình (Determinism)** — bất kỳ kỹ sư nào cũng có thể kiểm toán lại chính xác tại sao một chi phí thay thế node lại được tính như vậy.

---

### 6. Generalization? Nếu sang Healthcare/Legal thì Taxonomy còn đúng không?

#### ❓ Câu hỏi của Hội đồng:
*"Khả năng tổng quát hóa (Generalization) của hệ thống đến đâu? Nếu áp dụng SW-BTED sang miền Y tế (Healthcare SRS) hay Hợp đồng Pháp lý (Legal Contracts - CUAD) thì Taxonomy 4 miền hiện tại có còn đúng không?"*

#### 💡 Trả lời & Lập luận Bảo vệ:

1. **Tuyên bố Ranh giới Đóng góp Minh bạch (Scoped Claim):**
   * SW-BTED **KHÔNG tuyên bố** rằng phân vùng 4 miền $D_1 \dots D_4$ của thể loại Đồ án Capstone có thể áp dụng nguyên si cho mọi thể loại văn bản trên đời.
   * **Đóng góp #3 của bài báo nêu rõ:** *"T2 Domain Schema phải được căn chỉnh dựa trên một Taxonomy độc lập, citable của chính thể loại/ngành đó (Transferable Design Principle)."*

2. **Nguyên lý thích ứng theo Miền (Domain Adaptation Framework):**
   * Thuật toán cốt lõi SW-BTED, cấu trúc cây 4 tầng (Macro Filter $\rightarrow$ Domain Partition $\rightarrow$ Intent Matching $\rightarrow$ Terminology Verification) và công thức chi phí $w_{rep}^{(\ell)}$ **giữ nguyên 100% không đổi**.
   * Khi chuyển sang miền mới, chỉ cần thay đổi **Cấu hình Taxonomy ở tầng T2**:

| Thể Loại Tài Liệu (Genre) | Taxonomy Miền $T_2$ Được Thích Ứng | Cơ Sở Chuẩn (Citable Source) |
| :--- | :--- | :--- |
| **Software Capstone Proposals** | $D_1$: Context, $D_2$: Functional, $D_3$: Technical, $D_4$: Execution | FPT SRS Guideline & IEEE 830 |
| **GitBugs Bug Reports** | $D_1$: Bug Summary, $D_2$: Environment, $D_3$: Expected/Actual Behavior, $D_4$: Stack Trace | Bettenburg et al. (MSR) |
| **Healthcare Software SRS** | $D_1$: Clinical Context, $D_2$: Diagnostic Workflow, $D_3$: Regulatory (HIPAA/FDA), $D_4$: Interoperability (HL7/FHIR) | FDA Medical Device Software Specs |
| **Legal Contracts (CUAD)** | $D_1$: Parties & Recitals, $D_2$: Rights & Obligations, $D_3$: Termination & Liability, $D_4$: Governance & Jurisdiction | Atticus Project CUAD Taxonomy |

3. **Bằng chứng Thực nghiệm về Tính Tổng Quát Hóa (Cross-Domain Empirical Proof):**
   * Tính tổng quát hóa của nguyên lý này đã được **chứng minh thực nghiệm thành công 100% trên tập dữ liệu GitBugs Bug Reports (300 cặp)**: Khi chuyển từ Đồ án Capstone sang Bug Reports, hệ thống chỉ thay đổi Taxonomy T2 theo Bettenburg et al., giữ nguyên toàn bộ thuật toán SW-BTED, và kết quả đạt hòa thống kê ($p = 1.0000$) với SBERT chuyên biệt.

---

## 🎯 TÓM TẮT THÔNG ĐIỆP CỐT LÕI ĐỂ GHI NHỚ KHI BẢO VỆ

> *"SW-BTED là một khung giải pháp **Metric-Preserving Structure-Aware Alignment** giúp bóc tách văn bản theo phân cấp ngữ nghĩa. Hệ thống không nhằm mục đích đánh bại SBERT về điểm số F1 tuyệt đối trên văn bản phẳng, mà giải quyết 2 bài toán lớn mà mô hình phẳng không làm được: **(1) Loại bỏ triệt để lỗi Topic Conflation nhờ phân vùng miền T2**, và **(2) Cung cấp vết giải trình chi tiết từng tầng (Section/Sentence/Keyword) cho con người kiểm toán**."*
