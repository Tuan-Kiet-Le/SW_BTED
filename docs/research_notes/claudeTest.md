Show more
11:49 PM
Dựa trên toàn bộ phân tích, đây là code fix hoàn chỉnh thay thế categorize_topic bằng TF-IDF percentile selection.

python
# src/dataset_builder.py
# Thay thế toàn bộ logic categorize_topic + random pairing

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import random

def get_context_text(doc: dict) -> str:
    """
    Lấy text từ các sections phản ánh "bối cảnh và bài toán"
    — đây là nơi same-domain pairs có vocabulary overlap.
    """
    parts = []
    for field in ["context", "functional_requirements",
                  "non_functional_requirements"]:
        val = doc.get(field, "")
        if isinstance(val, list):
            val = " ".join(val)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def get_solution_text(doc: dict) -> str:
    """
    Lấy text từ các sections phản ánh "giải pháp kỹ thuật"
    — dùng để verify Type B pairs có Solution KHÁC nhau.
    """
    parts = []
    for field in ["proposed_solutions", "theory", "products"]:
        val = doc.get(field, "")
        if isinstance(val, list):
            val = " ".join(val)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def compute_pairwise_sim(texts: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity matrix."""
    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),   # bigrams giúp capture domain terms
        min_df=1,
        max_df=0.95,
    )
    tfidf = vec.fit_transform(texts)
    return cosine_similarity(tfidf)


def build_negative_pairs(
    all_docs: list[dict],
    n_type_b: int = 50,
    n_type_c: int = 50,
    type_b_percentile: float = 70,   # Top 30% context similarity
    type_c_percentile: float = 25,   # Bottom 25% context similarity
    random_seed: int = 42,
) -> tuple[list, list]:
    """
    Tạo Type B và Type C pairs dựa trên TF-IDF context similarity.

    Type B (Hard Negative — Same Topic):
        Context similarity CAO (top percentile)
        → Hai đề tài có cùng background vocabulary
        → Đây là "bẫy" cho baselines

    Type C (Easy Negative — Different Topic):
        Context similarity THẤP (bottom percentile)
        → Hai đề tài hoàn toàn khác biệt về nội dung

    Cả hai đều có label = 0 (không đạo văn).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    n = len(all_docs)
    doc_ids = [d["topic_code"] for d in all_docs]

    # ── Bước 1: Tính context similarity cho TẤT CẢ cặp ──────────
    context_texts = [get_context_text(d) for d in all_docs]
    ctx_sim = compute_pairwise_sim(context_texts)

    # ── Bước 2: Tính solution similarity để lọc Type B ───────────
    # Type B phải có solution KHÁC nhau (dù context giống)
    solution_texts = [get_solution_text(d) for d in all_docs]
    sol_sim = compute_pairwise_sim(solution_texts)

    # ── Bước 3: Phân loại cặp theo percentile ────────────────────
    # Lấy upper triangle (tránh tự so sánh và duplicate)
    all_pairs_info = []
    for i, j in combinations(range(n), 2):
        all_pairs_info.append({
            "i": i, "j": j,
            "doc_a": doc_ids[i],
            "doc_b": doc_ids[j],
            "ctx_sim": ctx_sim[i][j],
            "sol_sim": sol_sim[i][j],
        })

    ctx_sims = np.array([p["ctx_sim"] for p in all_pairs_info])

    # Threshold cho Type B: context similarity >= percentile 70
    threshold_b = np.percentile(ctx_sims, type_b_percentile)

    # Threshold cho Type C: context similarity <= percentile 25
    threshold_c = np.percentile(ctx_sims, type_c_percentile)

    # ── Bước 4: Lọc candidates ───────────────────────────────────
    # Type B candidates:
    # - Context similarity CAO (>= threshold_b)
    # - Solution similarity THẤP (< median sol_sim)
    #   → Bối cảnh giống nhưng giải pháp khác — đúng test case
    sol_median = np.median([p["sol_sim"] for p in all_pairs_info])

    type_b_candidates = [
        p for p in all_pairs_info
        if p["ctx_sim"] >= threshold_b
        and p["sol_sim"] < sol_median
    ]

    # Type C candidates:
    # - Context similarity THẤP (<= threshold_c)
    type_c_candidates = [
        p for p in all_pairs_info
        if p["ctx_sim"] <= threshold_c
    ]

    # ── Bước 5: Sample để đảm bảo diversity ─────────────────────
    # Sort Type B: context sim cao nhất trước
    type_b_candidates.sort(key=lambda x: -x["ctx_sim"])

    # Sort Type C: context sim thấp nhất trước
    type_c_candidates.sort(key=lambda x: x["ctx_sim"])

    # Lấy top candidates, sau đó shuffle để tránh bias đầu list
    b_pool = type_b_candidates[:min(len(type_b_candidates), n_type_b * 3)]
    c_pool = type_c_candidates[:min(len(type_c_candidates), n_type_c * 3)]
    random.shuffle(b_pool)
    random.shuffle(c_pool)

    type_b_pairs = b_pool[:n_type_b]
    type_c_pairs = c_pool[:n_type_c]

    # ── Bước 6: Validate số lượng đủ không ───────────────────────
    if len(type_b_pairs) < n_type_b:
        print(f"⚠️  Warning: Chỉ tạo được {len(type_b_pairs)}/{n_type_b} "
              f"Type B pairs. Thử giảm type_b_percentile.")

    if len(type_c_pairs) < n_type_c:
        print(f"⚠️  Warning: Chỉ tạo được {len(type_c_pairs)}/{n_type_c} "
              f"Type C pairs. Thử tăng type_c_percentile.")

    return type_b_pairs, type_c_pairs


def report_pair_stats(
    type_b_pairs: list,
    type_c_pairs: list,
) -> None:
    """
    In thống kê để verify dataset quality trước khi dùng.
    """
    b_ctx = [p["ctx_sim"] for p in type_b_pairs]
    c_ctx = [p["ctx_sim"] for p in type_c_pairs]
    b_sol = [p["sol_sim"] for p in type_b_pairs]

    print("=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)
    print(f"Type B ({len(type_b_pairs)} pairs):")
    print(f"  Context sim: mean={np.mean(b_ctx):.3f}, "
          f"std={np.std(b_ctx):.3f}, "
          f"min={np.min(b_ctx):.3f}, "
          f"max={np.max(b_ctx):.3f}")
    print(f"  Solution sim: mean={np.mean(b_sol):.3f}  "
          f"← phải THẤP (khác giải pháp)")

    print(f"\nType C ({len(type_c_pairs)} pairs):")
    print(f"  Context sim: mean={np.mean(c_ctx):.3f}, "
          f"std={np.std(c_ctx):.3f}, "
          f"min={np.min(c_ctx):.3f}, "
          f"max={np.max(c_ctx):.3f}")

    # Cohen's d mới
    pooled_std = np.sqrt((np.std(b_ctx)**2 + np.std(c_ctx)**2) / 2)
    d = (np.mean(b_ctx) - np.mean(c_ctx)) / (pooled_std + 1e-9)
    print(f"\nCohen's d(B,C) on Context: {d:.3f}")
    print(f"  Kỳ vọng: > 0.8 (Large effect)")
    print(f"  {'✅ PASS' if d > 0.8 else '❌ FAIL — tăng type_b_percentile'}")
    print("=" * 55)


def save_pairs_to_csv(
    type_a_pairs: list,
    type_b_pairs: list,
    type_c_pairs: list,
    output_path: str = "data/dataset/pairs.csv",
) -> None:
    """Lưu dataset ra CSV với format chuẩn."""
    import csv

    rows = []

    for p in type_a_pairs:
        rows.append({
            "doc_a": p["doc_a"],
            "doc_b": p["doc_b"],
            "label": 1,
            "type": "A",
        })

    for p in type_b_pairs:
        rows.append({
            "doc_a": p["doc_a"],
            "doc_b": p["doc_b"],
            "label": 0,
            "type": "B",
        })

    for p in type_c_pairs:
        rows.append({
            "doc_a": p["doc_a"],
            "doc_b": p["doc_b"],
            "label": 0,
            "type": "C",
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_a", "doc_b", "label", "type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} pairs to {output_path}")
    print(f"   Type A: {len(type_a_pairs)}")
    print(f"   Type B: {len(type_b_pairs)}")
    print(f"   Type C: {len(type_c_pairs)}")
Cách dùng
python
# scripts/rebuild_pairs.py

import json
from src.dataset_builder import (
    build_negative_pairs,
    report_pair_stats,
    save_pairs_to_csv,
)

# Load all documents
all_docs = json.load(open("data/raw/all_topics.json"))

# Build Type B và Type C mới
type_b_pairs, type_c_pairs = build_negative_pairs(
    all_docs     = all_docs,
    n_type_b     = 50,
    n_type_c     = 50,
    type_b_percentile = 70,  # Điều chỉnh nếu không đủ candidates
    type_c_percentile = 25,
    random_seed  = 42,
)

# Verify trước khi lưu
report_pair_stats(type_b_pairs, type_c_pairs)

# Load Type A pairs đã có (giữ nguyên)
type_a_pairs = json.load(open("data/dataset/type_a_pairs.json"))

# Lưu tất cả
save_pairs_to_csv(type_a_pairs, type_b_pairs, type_c_pairs)
Kết quả kỳ vọng sau khi rebuild
TRƯỚC (categorize_topic):
  Type B context sim: mean ≈ 0.21
  Type C context sim: mean ≈ 0.21
  Cohen's d(B,C) ≈ 0.0  ← không phân biệt được

SAU (TF-IDF percentile):
  Type B context sim: mean ≈ 0.35-0.50  ← cao hơn rõ ràng
  Type C context sim: mean ≈ 0.05-0.15  ← thấp hơn rõ ràng
  Cohen's d(B,C) > 0.8  ← large effect

Từ đó:
  B1/B2 baseline: có thể bị confused bởi Type B
                  (context words giống → predict là plagiarism)
  SW-BTED: Section weight Context=0.00 → ignore context signal
           → ít bị confused hơn → TNR Type B cao hơn B1/B2
  → ĐÂY MỚI LÀ TEST CASE ĐÚNG CHO CONTRIBUTION
Verify sau khi rebuild — checklist
□ report_pair_stats() cho d(B,C) > 0.8
□ Type B: ctx_sim mean > 0.30
□ Type C: ctx_sim mean < 0.15
□ Type B: sol_sim mean < median (solution thực sự khác)
□ Không có pair nào xuất hiện ở cả Type B và Type C
□ Không có pair nào overlap với Type A pairs
□ Chạy main_evaluation.py → quan sát TNR Type B của SW-BTED vs B1/B2