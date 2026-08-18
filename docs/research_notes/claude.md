Vấn đề: "Same domain" theo categorize_topic ≠ "shared vocabulary"
Nhìn vào ví dụ bạn đưa ra cho Type B:
Đề tài A: "Platform mua sắm mỹ phẩm nội địa"
Đề tài B: "Hệ thống quản lý chuỗi cung ứng nông sản"

Cả hai → category "E-commerce" (vì có "shop"/"product"/"supply"...)

Nhưng vocabulary thực tế:
  A: cosmetics, beauty products, online shopping, cart, payment
  B: agricultural supply chain, logistics, farmers, distribution

→ Keyword overlap giữa A và B: GẦN BẰNG 0
→ Cả hai "E-commerce" chỉ vì matched category keywords
  trong title/tag, không phải vì NỘI DUNG giống nhau
So sánh với Type C ví dụ:
Healthcare: "Ứng dụng đặt lịch khám sức khỏe tại nhà"
Finance:    "Hệ thống phân tích rủi ro tín dụng ngân hàng"

Keyword overlap: cũng gần bằng 0
Kết quả: Type B và Type C có overlap level tương đương nhau — cả hai đều ≈ 0 — vì categorize_topic chỉ phân loại dựa trên 1-2 từ khóa generic trong title, không đảm bảo content thực sự liên quan.
→ d(B,C) ≈ 0 KHÔNG phải vì "không có method nào tốt"
→ Mà vì Type B và Type C trong dataset hiện tại
  THỰC SỰ KHÔNG KHÁC NHAU về vocabulary overlap level
→ "Domain label" (E-commerce/Healthcare/...) không tương đương
  với "shared vocabulary" mà motivation gốc của bạn giả định

Đây là vấn đề DATASET DESIGN, không phải algorithm — và có thể fix được
Khác với kết luận trước (provenance artifact — khó fix), đây là vấn đề có thể fix bằng cách tạo Type B đúng cách.
Vấn đề cốt lõi: Random sampling trong category quá rộng
python# Hiện tại:
domain = "E-commerce"  # 1 trong 4 buckets lớn
candidates = [doc for doc in all_docs if doc.domain == domain]
doc_a, doc_b = random.sample(candidates, 2)

# "E-commerce" chứa: shopping platforms, supply chain,
#                     payment systems, marketplace, delivery apps...
# → Quá đa dạng, random pick → vocabulary overlap thấp

Fix: Type B cần "topical similarity" thay vì "category similarity"
Thay vì random trong 1 trong 4 buckets lớn, chọn cặp có vocabulary overlap cao thực sự:
pythonfrom sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_type_b_pairs(all_docs, n_pairs=43):
    """
    Type B: chọn cặp có TF-IDF similarity cao
    (top percentile) NHƯNG solution/methodology khác nhau.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Chỉ dùng Context + Problem để tính similarity
    context_texts = [doc.context + " " + doc.problem for doc in all_docs]
    tfidf_matrix = vectorizer.fit_transform(context_texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    pairs = []
    for i in range(len(all_docs)):
        for j in range(i+1, len(all_docs)):
            context_sim = sim_matrix[i][j]
            
            # Chỉ chọn cặp có Context similarity CAO
            # (top 30 percentile) — đây là "natural overlap"
            if context_sim > np.percentile(sim_matrix, 70):
                pairs.append((all_docs[i], all_docs[j], context_sim))
    
    # Sort theo context_sim giảm dần, lấy top n_pairs
    pairs.sort(key=lambda x: -x[2])
    return pairs[:n_pairs]
pythondef create_type_c_pairs(all_docs, n_pairs=50):
    """
    Type C: chọn cặp có TF-IDF similarity THẤP
    (bottom percentile) — đảm bảo thực sự không liên quan.
    """
    # Tương tự nhưng lấy bottom percentile
    pairs.sort(key=lambda x: x[2])  # tăng dần
    return pairs[:n_pairs]

Verify trước khi rebuild toàn bộ dataset
Trước khi regenerate Type B/C, chạy quick check:
python# Tính Context-only TF-IDF similarity cho TẤT CẢ pairs hiện tại
context_sims_B = [tfidf_context_sim(p) for p in type_b_pairs]
context_sims_C = [tfidf_context_sim(p) for p in type_c_pairs]

d_context = cohens_d(context_sims_B, context_sims_C)
print(f"d(B,C) on Context-only TF-IDF: {d_context}")
Nếu d_context cũng ≈ 0:
  → Confirm: categorize_topic không tạo ra
    meaningful B vs C distinction ở bất kỳ level nào
  → CẦN rebuild Type B/C bằng TF-IDF-based selection

Nếu d_context lớn (>0.8) nhưng overall d ≈ 0:
  → Quay lại Khả năng 2 — signal bị pha loãng bởi
    Solution/Theory sections
  → Không cần rebuild dataset, chỉ cần phân tích
    per-section d(B,C)

Khuyến nghị
Bước 1 (15 phút): Chạy d_context check ở trên
                  → Quyết định rẽ nhánh

Nếu d_context ≈ 0 (categorize_topic thất bại):
  Bước 2a: Rebuild Type B/C bằng TF-IDF percentile selection
           như code trên — dùng Context+Problem similarity
           làm criterion, KHÔNG dùng category labels
  Bước 3a: Chạy lại toàn bộ experiment với dataset mới
  Bước 4a: Tính lại d(B,C) — kỳ vọng B > C rõ ràng giờ

Nếu d_context lớn (categorize_topic OK ở Context level):
  Bước 2b: Tính per-section d(B,C) cho Solution, Theory, etc.
  Bước 3b: Xác nhận SECTION_WEIGHTS có "neutralize" được
           Context-driven similarity không