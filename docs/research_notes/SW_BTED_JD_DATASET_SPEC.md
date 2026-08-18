# SW-CapTree: Dataset-3 (Job Descriptions) Adaptation Spec
# Mục tiêu: Validate generalizability của SW-CapTree framework trên domain mới
> **Quan trọng:** File này chứa các paper tham khảo đã được verify là TỒN TẠI THẬT.
> Agent KHÔNG được thêm bất kỳ paper citation nào khác ngoài danh sách ở Mục 1
> mà không hỏi người dùng xác nhận trước.

---

## MỤC 1 — PAPER THAM KHẢO ĐÃ VERIFY (CHỈ DÙNG CÁC CÁI NÀY)

### P1 — Baseline method gần nhất trong domain JD

**Engelbach, M., Klau, D., Kintz, M., & Ulrich, A. (2024).**
*Combining Embeddings and Domain Knowledge for Job Posting Duplicate Detection.*
arXiv:2406.06257. Presented at 9th International Symposium on Language & Knowledge
Engineering (LKE 2024).
🔗 https://arxiv.org/abs/2406.06257

**Kết quả chính của paper này (dùng để so sánh với SW-CapTree):**
- Phát hiện: không một phương pháp đơn lẻ nào đạt hiệu năng thỏa mãn
- Kết hợp string comparison + deep textual embeddings + curated skill lookup lists
  cho kết quả tốt nhất
- Hệ thống này đang được deploy trong production

**Liên quan đến SW-CapTree:** Paper này confirm rằng domain JD cần kết hợp
lexical (string/keyword) và semantic (embedding) — đây là đúng cấu trúc T3+T4
của SW-CapTree. SW-CapTree có thể được positioned như một framework có
domain-aware hierarchy mà paper này thiếu.

---

### P2 — Hierarchical BERT (background cho hierarchical approach)

**Lu, J., Henchion, M., Bacher, I., & Mac Namee, B. (2021).**
*A Sentence-level Hierarchical BERT Model for Document Classification
with Limited Labelled Data.*
arXiv:2106.06738.
🔗 https://arxiv.org/abs/2106.06738

**Kết quả chính:**
- Hierarchical BERT (HBM) outperforms flat BERT trên long document classification
- Đặc biệt hiệu quả với 50-200 labeled instances (phù hợp với scale của FPT dataset)
- User study confirm rằng salient sentences identified by HBM hữu ích như explanations

**Liên quan đến SW-CapTree:** Paper này support argument rằng hierarchical
organization của documents cải thiện classification hơn flat embedding —
nhưng HBM không có domain-aware T2 layer, không giải quyết Topic Conflation.

---

### P3 — Dataset gốc (LinkedIn Job Postings)

**Arshkon. (2024).** *LinkedIn Job Postings (2023-2024).*
Kaggle Dataset. 123,000+ job postings.
🔗 https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

**Đặc điểm:**
- 123,000+ job postings từ LinkedIn năm 2023-2024
- Các fields: title, job_description, company_name, location, work_type
- Bao gồm skills và benefits files riêng biệt
- License: public, có thể dùng cho research

---

### P4 — PassionNet (benchmark framework đã dùng cho PURE)

**Saleem, S., Asim, M. N., & Dengel, A. (2025).**
*PassionNet: An innovative framework for duplicate and conflicting
requirements identification.*
Expert Systems With Applications, 293, 128684.
🔗 https://doi.org/10.1016/j.eswa.2025.128684

**Liên quan:** Đây là paper benchmark đã được dùng cho PURE dataset adaptation.
Cite để establish rằng SW-CapTree đã được tested trên dataset từ RE domain
trước khi extend sang JD domain.

---

> ⚠️ **CẢNH BÁO CHO AGENT:**
> Nếu agent tìm thấy paper khác trong quá trình làm việc và muốn thêm vào
> References, PHẢI báo cáo URL đầy đủ và tên paper chính xác cho người dùng
> xác nhận TRƯỚC KHI thêm vào bất kỳ report nào.
> KHÔNG tự ý thêm citation chưa được verify.

---

## MỤC 2 — LÝ DO CHỌN JOB DESCRIPTIONS DOMAIN

### 2.1 Topic Conflation xảy ra tự nhiên trong JD

Đây là failure mode chính của flat embedding methods trong domain JD:

```
Ví dụ Topic Conflation trong JD:

JD_A: "Senior Data Scientist tại HealthTech startup"
  → D1: "We are a fast-growing digital health company..."
  → D2: "PhD in Statistics/CS, 5+ years ML experience, Python, SQL"
  → D3: "Build predictive models for patient outcomes, analyze clinical data"
  → D4: "$150k-$200k, equity, health benefits"

JD_B: "Senior Data Scientist tại FinTech startup"
  → D1: "We are a fast-growing financial technology company..."
  → D2: "PhD in Statistics/CS, 5+ years ML experience, Python, SQL"
  → D3: "Build fraud detection models, analyze transaction data"
  → D4: "$150k-$200k, equity, health benefits"

SBERT similarity: ~0.89 (rất cao — cùng role, cùng skills)
Ground truth: DIFFERENT (khác industry, khác data domain, khác use case)

SW-CapTree T2 tách:
  D2 (Requirements): gần giống → similarity cao
  D3 (Responsibilities): khác ("patient outcomes" vs "fraud detection")
  → Domain-weighted: DIFFERENT ✓
```

### 2.2 T2 Labels cho JD domain

```python
JD_DOMAIN_MAPPING = {
    "D1_COMPANY_CONTEXT": [
        "company overview", "about us", "team description",
        "culture", "mission", "industry background"
    ],
    "D2_REQUIREMENTS": [
        "qualifications", "requirements", "education",
        "experience", "skills required", "what you bring"
    ],
    "D3_RESPONSIBILITIES": [
        "responsibilities", "what you'll do", "duties",
        "day-to-day", "key tasks", "role description"
    ],
    "D4_COMPENSATION": [
        "salary", "benefits", "perks", "compensation",
        "what we offer", "why join us"
    ]
}
```

### 2.3 Liên hệ với paper P1 (Engelbach et al. 2024)

Paper P1 dùng curated skill lookup lists — tương đương Tech Equivalence Map
của SW-CapTree ở T4. Điểm khác biệt quan trọng:
- P1: flat combination, không có domain partition
- SW-CapTree: hierarchical domain partition (T2) ngăn Topic Conflation
  trước khi apply skill matching (T4)

Đây là positioning gap rõ ràng để cite P1 trong Related Work.

---

## MỤC 3 — PIPELINE TẠO DATASET-3

### 3.1 Download và filter data

```python
# Source: LinkedIn Job Postings (2023-2024) - Kaggle arshkon
# URL: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

DATASET_CONFIG = {
    "source": "linkedin-job-postings",
    "target_industries": [
        "Information Technology",
        "Healthcare & Medical",
        "Finance & Banking",
        "Education",
        "Marketing & Advertising",
    ],
    "min_description_length": 200,  # words
    "max_description_length": 1000,
    "required_fields": ["title", "description", "company_name"],
}

# Chọn các JDs có cấu trúc rõ ràng (có section headers)
# Filter: loại bỏ JDs không có ít nhất 2 trong 4 domain sections
```

### 3.2 Tạo document pairs — Controlled Synthetic Method

Tương tự cách PURE was adapted. Không dùng random pairing.

**Positive pairs (label=1): Same job, different wording**
```python
def create_positive_pairs(jd_dataset):
    """
    Strategy: Lấy các JDs của cùng một role tại cùng một company
    đăng trên các platforms khác nhau hoặc ở thời điểm khác nhau.

    Nếu không tìm được: tạo synthetic paraphrase bằng cách:
    1. Swap synonyms trong Requirements section
    2. Reorder bullet points trong Responsibilities section
    3. Giữ nguyên Company Context và Compensation
    → Positive pair: same intent, different surface form
    """
    positive_pairs = []
    # Group by (company_name, title, location) để tìm near-duplicates tự nhiên
    groups = jd_dataset.groupby(['company_name', 'title'])
    for _, group in groups:
        if len(group) >= 2:
            # Lấy 2 JDs từ cùng company + title làm positive pair
            for i in range(len(group)-1):
                positive_pairs.append((group.iloc[i], group.iloc[i+1], 1))
    return positive_pairs
```

**Negative pairs — 2 loại:**
```python
def create_negative_pairs(jd_dataset):
    negative_pairs = []

    # Type 1: Easy negative — khác role hoàn toàn
    # (Software Engineer vs Chef de Cuisine)
    # → SBERT cũng phân biệt được

    # Type 2: Hard negative (Topic Conflation type) — cùng role khác industry
    # ĐÂY LÀ LOẠI QUAN TRỌNG NHẤT
    roles_across_industries = {
        "Data Scientist": ["HealthTech", "FinTech", "EdTech", "RetailTech"],
        "Product Manager": ["SaaS", "E-commerce", "Healthcare", "Finance"],
        "Software Engineer": ["Gaming", "FinTech", "HealthTech", "GovTech"],
        "Marketing Manager": ["B2B SaaS", "Consumer Tech", "Healthcare", "Education"],
    }
    # Lấy 2 JDs cùng title nhưng khác industry → hard negative
    # SBERT sẽ cho similarity cao vì cùng role vocabulary
    # SW-CapTree T2 D1_COMPANY_CONTEXT sẽ phân biệt được industry

    return negative_pairs
```

### 3.3 Target dataset size

```python
TARGET = {
    "total_pairs": 200,           # Tương đương FPT dataset
    "positive_ratio": 0.33,       # 66 positive pairs
    "negative_tc_type": 0.40,     # 80 hard negatives (Topic Conflation)
    "negative_easy_type": 0.27,   # 54 easy negatives
    "industries_covered": 5,      # IT, Healthcare, Finance, Education, Marketing
}
```

### 3.4 Labeling verification

```python
# Ground truth labels được assign theo quy tắc rõ ràng:
LABELING_RULES = {
    1: "Same job posting (same company + title) with different wording",
    0: "Different job postings (may share role category but differ in specific requirements)"
}

# Inter-rater agreement: 2 người label độc lập trên 20% sample
# Report Cohen's kappa trong paper
```

---

## MỤC 4 — EVALUATION PROTOCOL

### 4.1 SW-CapTree T2 Mapping cho JD

```yaml
# Thêm vào config.yaml
jd_domain_mapping:
  D1_COMPANY_CONTEXT:
    sections: ["about", "company", "overview", "culture", "mission"]
    weight: 2.0   # Quan trọng cho Topic Conflation detection
  D2_REQUIREMENTS:
    sections: ["requirements", "qualifications", "skills", "education"]
    weight: 2.0
  D3_RESPONSIBILITIES:
    sections: ["responsibilities", "duties", "what you will do", "role"]
    weight: 2.0
  D4_COMPENSATION:
    sections: ["salary", "benefits", "perks", "compensation", "offer"]
    weight: 1.0   # Ít quan trọng hơn cho similarity detection
```

### 4.2 Metrics cần report

```python
METRICS = {
    "overall": ["precision", "recall", "f1", "roc_auc"],
    "subset": {
        "easy_positive": "F1 on pairs with high surface similarity",
        "tc_type_negative": "TNR on Topic Conflation hard negatives",
        # ← ĐÂY LÀ METRIC QUAN TRỌNG NHẤT để chứng minh generalizability
    },
    "cross_dataset": "Compare TC-type TNR across FPT, PURE, JD datasets"
}
```

### 4.3 Baselines cần chạy

Tất cả baselines đã dùng ở FPT + PURE, thêm:

```python
ADDITIONAL_BASELINES = {
    "B_Engelbach": {
        "description": "Engelbach et al. (2024) method: string similarity + embedding + skill lookup",
        "reference": "arXiv:2406.06257",
        "note": "Direct comparison với SOTA trong JD domain"
    }
}
```

---

## MỤC 5 — PAPER NARRATIVE (Đoạn viết sẵn cho paper)

### 5.1 Đoạn Related Work — JD domain

```
[PASTE VÀO PAPER — Related Work section]

"Job posting duplicate detection has been studied as a practical industrial
problem. Engelbach et al. (2024) [P1] demonstrate that no single similarity
approach achieves satisfactory performance for job description deduplication;
rather, a combination of character-level string comparison, textual embeddings,
and curated skill lookup lists is required. Their finding that domain-specific
keyword lists (skill vocabularies) provide significant performance boosts
supports our design choice of the T4 Terminology Verification layer in
SW-CapTree, which uses TF-IDF with domain ontology normalization for
precise skill and technology matching.

However, Engelbach et al. (2024) [P1] address flat deduplication — detecting
exact duplicates of the same posting across platforms. Our task differs:
we address similarity detection between functionally distinct documents
that may share surface vocabulary (Topic Conflation), a failure mode
not addressed by flat embedding approaches."
```

### 5.2 Đoạn Generalizability Claim

```
[PASTE VÀO PAPER — Discussion section]

"The SW-CapTree framework is designed to be domain-agnostic: any structured
document corpus with identifiable domain labels can be analyzed using this
approach. The key requirement is that documents exhibit a consistent section
structure that can be mapped to T2 domain partitions.

We validate this claim across three document types:
(1) Software capstone registration forms (FPT dataset) — with domains
    Business Context / Functional / Technical / Execution Planning;
(2) Software requirement specifications (PURE adapted) — with domain
    Functional / Technical Realization;
(3) Job descriptions (LinkedIn JD dataset) — with domains
    Company Context / Requirements / Responsibilities / Compensation.

Across all three domains, removing the T2 Domain Partition layer (ablation
no_T2) consistently reduces performance on Topic Conflation cases, confirming
that the domain-aware hierarchy is the mechanism that prevents the failure
mode rather than an artifact of the FPT domain."
```

---

## MỤC 6 — CHECKLIST CHO AGENT

```
PHASE 0: Verify dataset
  [ ] Download LinkedIn JD dataset từ Kaggle arshkon
  [ ] Kiểm tra: có field "description" đủ dài không?
  [ ] Kiểm tra: có thể identify 4 domain sections không?
  [ ] Report số lượng JDs sau filter → hỏi người dùng

PHASE 1: Tạo pairs
  [ ] Tạo positive pairs (same job, different wording)
  [ ] Tạo hard negative pairs (same role, different industry)
  [ ] Tạo easy negative pairs (completely different roles)
  [ ] Verify target distribution (200 pairs, 33% positive)
  [ ] Report dataset stats → hỏi người dùng

PHASE 2: Build trees và evaluate
  [ ] Apply JD domain mapping (T2 labels như Mục 4.1)
  [ ] Chạy SW-CapTree với config JD
  [ ] Chạy tất cả baselines (B1-B7) + B_Engelbach
  [ ] Tính TC-type TNR riêng biệt
  [ ] McNemar SW-CapTree vs B2 (SBERT) trên TC subset

PHASE 3: Báo cáo
  [ ] So sánh TC-type TNR: FPT vs PURE vs JD
  [ ] Nếu SW-CapTree > SBERT trên TC-type ở tất cả 3 dataset → generalizability confirmed
  [ ] Tạo cross-dataset comparison table
```

---

## MỤC 7 — CÁC QUYẾT ĐỊNH AGENT PHẢI HỎI

| Tình huống | Câu hỏi |
|-----------|---------|
| Không tìm được positive pairs tự nhiên (same job khác wording) | "Có muốn dùng synthetic paraphrase không? Nếu có, cần confirm labeling strategy." |
| TC-type TNR của SW-CapTree thấp hơn SBERT trên JD | "SW-CapTree không generalize tốt sang JD. Báo cáo trước khi tiếp tục." |
| Agent tìm thêm paper muốn cite | "Cần URL đầy đủ + tên paper để người dùng verify trước khi thêm vào citation." |
| Dataset sau filter < 150 pairs | "Không đủ mẫu. Thử nới lỏng filter (min_length = 150 words)?" |
```
