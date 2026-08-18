# SW-BTED: Tái Cấu Trúc Cây Từ 6 Tầng Xuống 4 Tầng
> **Mục đích:** Agent đọc file này và thực hiện toàn bộ thay đổi kiến trúc
> từ cây 6 tầng cũ sang cây 4 tầng mới có justification rõ ràng.
> Thực hiện tuần tự PHASE 0 → PHASE 1 → PHASE 2 → PHASE 3.
> Dừng và báo cáo sau mỗi PHASE.

---

## BỐI CẢNH — Lý do thay đổi

### Cây 6 tầng cũ (ĐÃ LỖI THỜI)
```
T1: ROOT
T2: DOMAIN
T3: GROUP          ← Ablation A3: bỏ đi không thay đổi gì (p=1.0 cả 2 dataset)
T4: ATOMIC_REQ
T5: SEMANTIC_ROLE  ← Ablation A2: bỏ đi CẢI THIỆN PURE +0.1044 (p=5.4×10⁻⁷)
T6: LEAF
```

**Vấn đề:** T3 và T5 không có justification thực nghiệm.
- T3 (GROUP): Không đóng góp gì — ablation cho thấy bỏ đi kết quả không đổi
- T5 (SEMANTIC_ROLE): Gây hại trên câu ngắn/đơn giản vì tạo parsing noise

### Cây 4 tầng mới (ÁP DỤNG NGAY)
```
T1: Macro-Filter Layer       (ROOT)
T2: Domain Partition Layer   (DOMAIN)
T3: Intent Matching Layer    (ATOMIC_REQ + SBERT)
T4: Terminology Verification Layer  (LEAF + TF-IDF/BM25 + CSO/TEM)
```

---

## ĐẶC TẢ CHI TIẾT 4 TẦNG

### T1 — Macro-Filter Layer (Tầng Lọc Vĩ Mô)

**Tên kỹ thuật:** `MacroFilterNode`

**Dữ liệu chứa:**
```python
{
    "node_type": "MACRO_FILTER",
    "project_id": str,           # "SU26SE082"
    "global_embedding": np.array # SBERT all-MiniLM-L6-v2 của toàn bộ văn bản
}
```

**Vai trò trong kiến trúc:**
Đây là cơ chế computational gatekeeping — không phải bước phân loại
mà là bước loại trừ sớm các cặp rõ ràng không liên quan trước khi
tốn chi phí tính APTED. Không có tầng này, hệ thống phải chạy full
tree comparison cho mọi cặp, không khả thi ở scale lớn.

**Câu hỏi tầng này trả lời:** *"Hai tài liệu này có đáng so sánh chi tiết không?"*

**Logic pre-filter:**
```python
PREFILTER_THRESHOLD = 0.25

def should_compare(doc_A, doc_B) -> bool:
    sim = cosine_similarity(doc_A.global_embedding,
                            doc_B.global_embedding)
    return sim >= PREFILTER_THRESHOLD

# Nếu False → similarity = alpha * 0 + (1-alpha) * sim_global
# KHÔNG set cứng sim = 0 (đã fix từ Phase 1 trước)
```

**Tại sao KHÔNG tham gia tính edit cost:**
T1 là identity node — mọi cây đều có đúng một T1 node. Insert/delete/replace
T1 không có ý nghĩa ngữ nghĩa nên w_del(T1) = w_ins(T1) = 0.

---

### T2 — Domain Partition Layer (Tầng Phân Vùng Miền Ngữ Nghĩa)

**Tên kỹ thuật:** `DomainPartitionNode`

**4 domain cố định:**
```python
DOMAINS = {
    "D1_BUSINESS_CONTEXT":      # Context, English Title, Vietnamese Title
    "D2_FUNCTIONAL":            # Functional Requirement, Proposed Solutions, Products
    "D3_TECHNICAL_REALIZATION": # Non-functional Requirement, Applied Theory (tech stack)
    "D4_EXECUTION_PLANNING":    # Proposed Tasks
}
```

**Vai trò trong kiến trúc — ĐÂY LÀ TẦNG QUAN TRỌNG NHẤT:**

T2 là tầng giải quyết **Topic Conflation** — failure mode cụ thể của
flat embedding methods.

**Định nghĩa Topic Conflation:**
> *Topic Conflation xảy ra khi một phương pháp similarity gán điểm cao
> cho hai tài liệu RE chia sẻ vocabulary về chủ đề/lĩnh vực ứng dụng
> (Business Context) nhưng khác biệt hoàn toàn ở yêu cầu chức năng
> (Functional) và lựa chọn kỹ thuật (Technical).*

**Tại sao flat methods KHÔNG THỂ giải quyết Topic Conflation:**
```
Flat embedding gộp TẤT CẢ vocabulary vào 1 vector:
[context_vocab] + [functional_vocab] + [tech_vocab] → 1 embedding

"App chăm sóc người già":  [chăm sóc, người già] + [theo dõi sức khỏe] + [React]
"Nền tảng thú cưng":       [chăm sóc, thú cưng]  + [theo dõi hành vi]  + [React]

→ SBERT cosine cao → Predict: SIMILAR ← LỖI (Topic Conflation)

SW-BTED T2 tách riêng:
D1 (Context):    "chăm sóc người già" ≠ "chăm sóc thú cưng" → DIFFERENT ✓
D2 (Functional): "theo dõi sức khỏe" ≈ "theo dõi hành vi"   → similar
→ Domain-weighted: DIFFERENT ✓
```

**Câu hỏi tầng này trả lời:** *"Đây là loại thông tin RE gì?"*

**Cost parameters:**
```python
w_del_T2  = 2.0   # Mất toàn bộ một domain = rất đắt
w_ins_T2  = 2.0
beta_T2   = 0.0   # Domain là categorical label — chỉ dùng schema distance
                   # Dist_content giữa "D1_BUSINESS_CONTEXT" và "D2_FUNCTIONAL"
                   # không có ý nghĩa ngữ nghĩa
```

**Schema distance matrix T2:**
```python
DOMAIN_SCHEMA_DIST = {
    ("D1","D1"): 0.0, ("D2","D2"): 0.0,
    ("D3","D3"): 0.0, ("D4","D4"): 0.0,
    ("D1","D2"): 0.8, ("D2","D1"): 0.8,  # Context vs Functional — rất khác
    ("D1","D3"): 0.9, ("D3","D1"): 0.9,
    ("D1","D4"): 0.9, ("D4","D1"): 0.9,
    ("D2","D3"): 0.5, ("D3","D2"): 0.5,  # Functional vs Technical — gần hơn
    ("D2","D4"): 0.7, ("D4","D2"): 0.7,
    ("D3","D4"): 0.6, ("D4","D3"): 0.6,
}
```

---

### T3 — Intent Matching Layer (Tầng Đối Sánh Ý Định)

**Tên kỹ thuật:** `IntentMatchingNode`

**Dữ liệu chứa:**
```python
{
    "node_type": "INTENT_MATCHING",
    "feature_label": str,        # Tên tính năng nếu có: "AI-Assisted OCR"
    "raw_text": str,             # Câu gốc
    "normalized_text": str,      # Câu đã elide resolution
    "sentence_embedding": np.array,  # SBERT all-MiniLM-L6-v2
    "domain": str,               # D1/D2/D3/D4 (inherit từ T2 cha)
}
```

**Vai trò trong kiến trúc:**

T3 là cơ chế nhận diện **semantic equivalence** ở cấp câu yêu cầu.
Sinh viên thường paraphrase yêu cầu theo nhiều cách khác nhau —
"Hệ thống xác minh danh tính" và "Module kiểm tra thông tin tài khoản"
diễn đạt cùng intent nhưng không chia sẻ từ chung nào.
Nếu chỉ dùng keyword matching (T4), hai câu này bị coi là hoàn toàn
khác nhau → false negative.

T3 là cơ chế bù đắp brittleness của lexical matching.

**Câu hỏi tầng này trả lời:**
*"Ý nghĩa của câu yêu cầu này là gì? Có câu nào trong tài liệu kia
mang cùng ý nghĩa không dù dùng từ khác?"*

**Dist_content tại T3:**
```python
def dist_content_T3(node_u, node_v):
    return 1 - cosine_similarity(
        node_u.sentence_embedding,
        node_v.sentence_embedding
    )
```

**Cost parameters:**
```python
w_del_T3  = 1.0
w_ins_T3  = 1.0
beta_T3   = 0.9   # Content (SBERT intent) dominant
                   # Schema distance ít quan trọng vì T3 nodes đều
                   # là "atomic requirement" — cùng loại
```

**Quy tắc tạo T3 nodes:**

```python
def create_T3_nodes(section_text: str, domain: str) -> list:
    nodes = []
    sentences = sentence_tokenize(section_text)

    for sent in sentences:
        # Elide resolution: thêm subject nếu bị ẩn
        normalized = resolve_elided_subject(sent, context=section_text)

        node = IntentMatchingNode(
            raw_text=sent,
            normalized_text=normalized,
            sentence_embedding=sbert_model.encode(normalized),
            domain=domain,
            feature_label=extract_feature_label(sent),
            # feature_label: nếu câu có dạng "FeatureName: description"
            # → lưu "FeatureName" vào đây, KHÔNG được bỏ mất
        )
        nodes.append(node)

    return nodes

# T3 là CON TRỰC TIẾP của T2 (không còn T3 GROUP ở giữa)
# T2_DOMAIN → [T3_node_1, T3_node_2, ..., T3_node_n]
```

---

### T4 — Terminology Verification Layer (Tầng Xác Minh Thuật Ngữ)

**Tên kỹ thuật:** `TerminologyVerificationNode`

**Dữ liệu chứa:**
```python
{
    "node_type": "TERMINOLOGY_VERIFICATION",
    "raw_keyword": str,          # Từ khóa gốc: "jwt authentication"
    "canonical_form": str,       # Sau CSO/TEM: "json_web_token"
    "tfidf_weight": float,       # Trọng số TF-IDF trong document
    "source_role": str,          # "technology" / "concept" / "action"
    "parent_sentence": str,      # Câu T3 chứa keyword này
}
```

**Vai trò trong kiến trúc — ĐỐI TRỌNG CỦA T3:**

T3 (SBERT) có điểm yếu đối xứng với điểm mạnh:
nó **over-generalize** các thuật ngữ kỹ thuật cụ thể.

> **Lưu ý quan trọng từ người dùng:** Hai đề tài dùng công nghệ khác nhau
> KHÔNG được coi là giống nhau — đây là lỗi real của SBERT.
> Ví dụ: "ReactJS + PostgreSQL" ≠ "Angular + MongoDB"
> SBERT có thể cho similarity cao vì cả hai đều là câu
> "dùng frontend framework + database" nhưng thực tế là hai
> lựa chọn kỹ thuật hoàn toàn độc lập.

T4 là cơ chế đảm bảo **terminology precision** — nơi mà sự khác biệt
về thuật ngữ cụ thể quan trọng hơn similarity về cấu trúc câu.

T3 và T4 là **cặp đối lập có chủ đích**:
- T3 (SBERT): *bắt đồng nghĩa* — "xác minh danh tính" ≡ "kiểm tra tài khoản"
- T4 (TF-IDF): *bắt khác biệt cụ thể* — "ReactJS" ≠ "Angular"

**Câu hỏi tầng này trả lời:**
*"Thuật ngữ kỹ thuật cụ thể nào được dùng? Có trùng khớp chính xác không?"*

**Pipeline tạo T4 nodes:**
```python
def create_T4_nodes(sentence: str,
                    tfidf_vectorizer,
                    cso_lookup,
                    tech_equiv_map) -> list:
    nodes = []

    # Bước 1: Trích noun phrases và technical terms
    doc = spacy_nlp(sentence)
    candidates = [
        chunk.text.lower()
        for chunk in doc.noun_chunks
    ] + [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "VERB"]
        and len(token.text) > 2
    ]

    for kw in candidates:
        # Bước 2: Chuẩn hóa
        canonical = normalize_keyword(kw, cso_lookup, tech_equiv_map)

        # Bước 3: Lấy TF-IDF weight
        weight = tfidf_vectorizer.get_weight(canonical)

        # Bước 4: Phân loại source_role
        if is_technology(canonical, tech_equiv_map, cso_lookup):
            role = "technology"
        elif is_action_verb(kw, doc):
            role = "action"
        else:
            role = "concept"

        nodes.append(TerminologyVerificationNode(
            raw_keyword=kw,
            canonical_form=canonical,
            tfidf_weight=weight,
            source_role=role,
        ))

    return nodes


def normalize_keyword(kw, cso_lookup, tech_equiv_map):
    # 1. Lowercase + lemmatize
    lemma = lemmatize(kw)
    # 2. Tra CSO
    if lemma in cso_lookup:
        return cso_lookup[lemma]["canonical"]
    # 3. Tra Tech Equivalence Map
    if lemma in tech_equiv_map:
        return tech_equiv_map[lemma]
    # 4. Giữ nguyên
    return lemma
```

**Dist_content tại T4:**
```python
def dist_content_T4(node_u, node_v):
    # So sánh canonical form
    if node_u.canonical_form == node_v.canonical_form:
        return 0.0   # Exact match
    else:
        return 1.0   # Không match (binary — không có partial match ở leaf)
```

**Cost parameters:**
```python
w_del_T4  = 0.5 * node.tfidf_weight   # TF-IDF weighted: keyword quan trọng hơn thì đắt hơn
w_ins_T4  = 0.5 * node.tfidf_weight
beta_T4   = 1.0   # Chỉ content (canonical match), không dùng schema distance
```

---

## CẤU TRÚC CÂY HOÀN CHỈNH

```
ROOT [T1: MacroFilterNode]
│   global_embedding = SBERT(toàn bộ văn bản)
│
├── DOMAIN [T2: DomainPartitionNode: D1_BUSINESS_CONTEXT]
│   │
│   ├── INTENT [T3: IntentMatchingNode]
│   │   │   normalized_text = "The system provides social housing..."
│   │   │   sentence_embedding = SBERT(normalized_text)
│   │   │
│   │   ├── TERM [T4: TerminologyVerificationNode]
│   │   │       canonical_form = "social_housing"
│   │   │       tfidf_weight = 0.423
│   │   │       source_role = "concept"
│   │   │
│   │   └── TERM [T4: TerminologyVerificationNode]
│   │           canonical_form = "government_regulation"
│   │           tfidf_weight = 0.387
│   │           source_role = "concept"
│   │
│   └── INTENT [T3: IntentMatchingNode]
│       ...
│
├── DOMAIN [T2: DomainPartitionNode: D2_FUNCTIONAL]
│   │
│   ├── INTENT [T3: IntentMatchingNode]
│   │   │   normalized_text = "Applicant uses AI to extract text from National IDs..."
│   │   │   feature_label = "Assisted Registration (OCR Helper)"
│   │   │
│   │   ├── TERM [T4] canonical="ai_ocr"      role="technology"  weight=0.512
│   │   ├── TERM [T4] canonical="national_id"  role="concept"     weight=0.445
│   │   └── TERM [T4] canonical="extract"      role="action"      weight=0.231
│   │
│   └── ...
│
├── DOMAIN [T2: DomainPartitionNode: D3_TECHNICAL_REALIZATION]
│   │
│   └── INTENT [T3: IntentMatchingNode]
│       │   normalized_text = "System uses ReactJS for frontend..."
│       │
│       ├── TERM [T4] canonical="reactjs"      role="technology"  weight=0.634
│       └── TERM [T4] canonical="frontend"     role="concept"     weight=0.289
│           # ← TF-IDF đảm bảo "reactjs" ≠ "angular" không bị nhầm lẫn
│
└── DOMAIN [T2: DomainPartitionNode: D4_EXECUTION_PLANNING]
    └── ...
```

---

## CÔNG THỨC CHI PHÍ CẬP NHẬT (4 TẦNG)

### Công thức tổng quát (giữ nguyên từ trước)

$$w_{rep}^{(\ell)}(u,v) = \left(w_{del}^{(\ell)}(u) + w_{ins}^{(\ell)}(v)\right) \cdot \left(\beta_\ell \cdot \text{Dist}_{content}(u,v) + (1-\beta_\ell) \cdot \text{Dist}_{schema}(C(u),C(v))\right)$$

### Bảng tham số 4 tầng

| Tầng | β_ℓ | w_del | w_ins | Dist_content | Lý do |
|------|-----|-------|-------|-------------|-------|
| T1 (MacroFilter) | — | **0** | **0** | — | Không tham gia edit cost |
| T2 (DomainPartition) | **0.0** | 2.0 | 2.0 | Không dùng | Domain là categorical label cứng — chỉ schema distance |
| T3 (IntentMatching) | **0.9** | 1.0 | 1.0 | 1 − cosine(SBERT_u, SBERT_v) | Nội dung câu là thông tin chính |
| T4 (TermVerification) | **1.0** | 0.5×tfidf | 0.5×tfidf | Binary: 0 nếu canonical match, 1 nếu không | Chỉ content — exact canonical match |

### Similarity score cuối

$$\text{Sim}(A,B) = \alpha \cdot \widehat{\text{TED}}_{4L}(A,B) + (1-\alpha) \cdot \text{cosine}(\mathbf{e}_A, \mathbf{e}_B)$$

```python
ALPHA = 0.60    # Từ ablation Group D (stable across folds)
```

---

## PHASE 0 — KIỂM TRA TRƯỚC KHI THAY ĐỔI

### Task 0.1 — Backup toàn bộ code cây 6 tầng

```bash
# Tạo branch mới, KHÔNG xóa code cũ
git checkout -b feature/4-layer-tree
git stash  # hoặc commit toàn bộ code hiện tại vào branch cũ
```

### Task 0.2 — Liệt kê tất cả files cần sửa

Agent tìm và liệt kê tất cả files có chứa references đến:
```
"T3_GROUP", "T5_SEMANTIC_ROLE", "GROUP", "SemanticRole",
"extract_semantic_roles", "build_t5_nodes", "build_t3_group"
```

Lưu danh sách ra: `migration/files_to_modify.txt`

**DỪNG LẠI sau Task 0.2 — báo cáo danh sách files.**

---

## PHASE 1 — MIGRATION CODE

### Task 1.1 — Cập nhật tree builder

**File cần sửa:** `src/01_parser.py` hoặc tương đương

```python
# ĐÃ XÓA: build_t3_group_nodes()
# ĐÃ XÓA: build_t5_semantic_role_nodes()

# MỚI: Pipeline 4 tầng
def build_tree(document: Document) -> Tree:
    # T1
    root = MacroFilterNode(
        project_id=document.id,
        global_embedding=sbert.encode(document.full_text)
    )

    # T2: 4 domains cố định
    for domain_id, sections in DOMAIN_MAPPING.items():
        domain_text = extract_sections(document, sections)
        domain_node = DomainPartitionNode(domain=domain_id)

        # T3: Sentence tokenize → IntentMatchingNode
        sentences = tokenize_sentences(domain_text)
        for sent in sentences:
            intent_node = IntentMatchingNode(
                raw_text=sent,
                normalized_text=resolve_elided_subject(sent),
                sentence_embedding=sbert.encode(resolve_elided_subject(sent)),
                domain=domain_id,
                feature_label=extract_feature_label(sent),
            )

            # T4: Keywords → TerminologyVerificationNode
            keywords = extract_keywords(sent)
            for kw in keywords:
                term_node = TerminologyVerificationNode(
                    raw_keyword=kw,
                    canonical_form=normalize(kw),
                    tfidf_weight=tfidf.get_weight(normalize(kw)),
                    source_role=classify_role(kw),
                )
                intent_node.add_child(term_node)

            domain_node.add_child(intent_node)
        root.add_child(domain_node)

    return Tree(root)
```

### Task 1.2 — Cập nhật cost engine

**File cần sửa:** `src/04_cost_engine.py` hoặc tương đương

```python
LAYER_PARAMS = {
    "T1_MACRO_FILTER": {
        "beta": None,
        "w_del": 0.0,   # KHÔNG tham gia edit cost
        "w_ins": 0.0,
    },
    "T2_DOMAIN_PARTITION": {
        "beta": 0.0,
        "w_del": 2.0,
        "w_ins": 2.0,
        "dist_content": lambda u, v: 0.0,  # Không dùng content distance
        "dist_schema": DOMAIN_SCHEMA_DIST,
    },
    "T3_INTENT_MATCHING": {
        "beta": 0.9,
        "w_del": 1.0,
        "w_ins": 1.0,
        "dist_content": lambda u, v: 1 - cosine_sim(
            u.sentence_embedding, v.sentence_embedding
        ),
        "dist_schema": lambda u, v: 0.0,  # Tất cả T3 nodes cùng loại
    },
    "T4_TERMINOLOGY_VERIFICATION": {
        "beta": 1.0,
        "w_del": lambda node: 0.5 * node.tfidf_weight,
        "w_ins": lambda node: 0.5 * node.tfidf_weight,
        "dist_content": lambda u, v: 0.0 if u.canonical_form == v.canonical_form else 1.0,
        "dist_schema": lambda u, v: 0.0,  # Không dùng
    },
}
```

### Task 1.3 — Cập nhật config.yaml

```yaml
# SW-BTED 4-Layer Configuration
tree_architecture: "4-layer"
layers:
  - name: "T1_MACRO_FILTER"
    description: "Global SBERT embedding for computational pre-filtering"
    participates_in_edit_cost: false

  - name: "T2_DOMAIN_PARTITION"
    description: "RE domain separation to prevent Topic Conflation"
    beta: 0.0
    w_del: 2.0
    w_ins: 2.0

  - name: "T3_INTENT_MATCHING"
    description: "SBERT sentence-level semantic matching for paraphrase detection"
    beta: 0.9
    w_del: 1.0
    w_ins: 1.0

  - name: "T4_TERMINOLOGY_VERIFICATION"
    description: "TF-IDF keyword matching for terminology precision"
    beta: 1.0
    w_del_factor: 0.5   # × tfidf_weight
    w_ins_factor: 0.5   # × tfidf_weight

alpha: 0.60
prefilter_threshold: 0.25
embedding_model: "all-MiniLM-L6-v2"
tfidf_fit_on: "train_set_only"   # KHÔNG fit trên test set
```

---

## PHASE 2 — RE-EVALUATION

### Task 2.1 — Chạy lại evaluation với cây 4 tầng

Protocol giữ nguyên từ trước:
- 5-fold stratified CV
- Threshold tối ưu trên validation set
- Cả 2 datasets: FPT và PURE_adapted

```python
VARIANTS_TO_RUN = {
    "SW_BTED_4L":          "Proposed 4-layer (main result)",
    "SW_BTED_4L_no_T4":    "Ablation: remove Terminology layer",
    "SW_BTED_4L_no_T2":    "Ablation: remove Domain Partition (Topic Conflation test)",
    "SW_BTED_4L_alpha_0":  "Sanity check: α=0 should ≈ B2 SBERT",
}
```

> **QUAN TRỌNG — Ablation SW_BTED_4L_no_T2:**
> Đây là experiment quan trọng nhất để chứng minh giá trị của T2.
> Khi không có T2 (Domain Partition), tất cả T3 nodes trở thành
> con trực tiếp của T1 Root → cây phẳng về mặt domain.
> Kỳ vọng: F1 giảm trên các cặp Topic Conflation cases.

### Task 2.2 — Topic Conflation Subset Analysis

Từ 14 case studies đã có, phân loại theo 2 nhóm:

```python
CASE_TYPES = {
    "TC_TYPE": [
        # Cặp SBERT fail vì Topic Conflation
        # (cùng chủ đề ứng dụng, khác chức năng thực sự)
        # Ví dụ: "App người già" vs "App thú cưng"
    ],
    "NON_TC_TYPE": [
        # Cặp tất cả methods đều đúng
    ]
}

# McNemar SW_BTED_4L vs B2_SBERT CHỈ TRÊN TC_TYPE cases
# Đây là targeted evidence cho Topic Conflation contribution
```

### Task 2.3 — Output

```
results/4layer/
├── main_results_FPT.csv
├── main_results_PURE.csv
├── ablation_no_T2_FPT.csv          # ← quan trọng nhất
├── ablation_no_T4_FPT.csv
├── topic_conflation_analysis.csv
└── comparison_6L_vs_4L.csv         # So sánh cũ vs mới
```

**DỪNG LẠI sau Phase 2 — báo cáo toàn bộ kết quả.**

---

## PHASE 3 — CẬP NHẬT PAPER CONTENT

Sau khi có kết quả Phase 2, cập nhật `SW_BTED_PAPER_CONTENT.md`:

### Task 3.1 — Sửa tên và mô tả 4 tầng trong paper

Thay thế toàn bộ references đến "6-layer" bằng "4-layer".

Thêm đoạn này vào Section Methodology:

```
[ĐOẠN METHODOLOGY — SẴN SÀNG ĐƯA VÀO PAPER]

"We propose SW-CapTree, a four-layer labeled tree representation
specifically designed for software capstone registration documents.
Each layer addresses a distinct analytical question:

T1 (Macro-Filter Layer) answers: 'Is comparison warranted?'
T2 (Domain Partition Layer) answers: 'What type of RE information is this?'
T3 (Intent Matching Layer) answers: 'What does this requirement mean?'
T4 (Terminology Verification Layer) answers: 'What specific terms are used?'

The domain-aware hierarchy of T2 is the necessary condition for
resolving Topic Conflation — a failure mode of flat embedding methods
in which shared domain vocabulary (business context, application
category) inflates similarity scores between functionally distinct
documents. T3 and T4 form a complementary pair: T3 captures semantic
equivalence across paraphrases (semantic generalization), while T4
enforces terminology precision where specific technical choices matter
(lexical specificity). Neither layer is redundant: removing T3 causes
false negatives on paraphrased requirements; removing T4 causes false
positives when documents share functional structure but differ in
specific technology choices."
```

### Task 3.2 — Sửa phần Ablation Study

Cập nhật Group A table để reflect 4-layer variants thay vì 6-layer.

---

## PHẦN 4 — CHECKLIST TỔNG

```
PHASE 0: Backup và chuẩn bị
  [ ] git checkout -b feature/4-layer-tree
  [ ] Liệt kê files cần sửa → migration/files_to_modify.txt
  [ ] DỪNG và báo cáo

PHASE 1: Migration code
  [ ] Xóa build_t3_group_nodes() và build_t5_semantic_role_nodes()
  [ ] Implement MacroFilterNode, DomainPartitionNode,
      IntentMatchingNode, TerminologyVerificationNode
  [ ] Cập nhật cost engine với LAYER_PARAMS 4 tầng
  [ ] Cập nhật config.yaml
  [ ] Unit test: build 1 cây từ phiếu SU26SE082 → verify structure

PHASE 2: Re-evaluation
  [ ] Chạy SW_BTED_4L trên FPT và PURE
  [ ] Chạy ablation SW_BTED_4L_no_T2 (Topic Conflation test)
  [ ] Chạy ablation SW_BTED_4L_no_T4
  [ ] Phân loại 14 case studies thành TC_TYPE vs NON_TC_TYPE
  [ ] McNemar trên TC_TYPE subset
  [ ] DỪNG và báo cáo

PHASE 3: Paper update
  [ ] Cập nhật SW_BTED_PAPER_CONTENT.md
  [ ] Sửa tất cả references từ "6-layer" thành "4-layer"
  [ ] Thêm đoạn Methodology với 4 tầng được justify

```

---

## PHẦN 5 — QUYẾT ĐỊNH AGENT PHẢI HỎI

| Tình huống | Câu hỏi |
|-----------|---------|
| SW_BTED_4L F1 < SW_BTED_6L trên FPT | "4-layer tệ hơn 6-layer. Báo cáo mức độ giảm trước khi tiếp tục." |
| Ablation no_T2 KHÔNG giảm F1 | "Domain Partition không có tác động. T2 contribution không có evidence. Báo cáo ngay." |
| TC_TYPE cases < 3 cặp trong 14 case studies | "Không đủ Topic Conflation cases để claim. Báo cáo số lượng thực tế." |
| Unit test cây fail | "Báo cáo lỗi cụ thể trước khi chạy evaluation." |
