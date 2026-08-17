"""
Baselines (B1-B5) implementations for the plagiarism detection benchmark.
"""
import numpy as np
import apted
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# For Standard TED Cost Model
class StandardCostModel:
    def w_del(self, u) -> float:
        return 1.0
    def w_ins(self, v) -> float:
        return 1.0
    def w_rep(self, u, v) -> float:
        return 0.0 if u.label == v.label else 1.0

# ── Helpers to get texts from trees or full_texts ──
def tree_to_section_texts(tree_node) -> dict:
    section_texts = {}
    for sec in tree_node.children:
        words = [leaf.label for leaf in sec.children]
        section_texts[sec.schema_class] = " ".join(words)
    return section_texts

def tree_to_full_text(tree_node) -> str:
    section_texts = tree_to_section_texts(tree_node)
    return tree_node.label + " " + " ".join(section_texts.values())

def get_document_full_text(doc_code, full_texts, tree_node) -> str:
    if full_texts and doc_code in full_texts:
        sections = full_texts[doc_code]
        title = tree_node.label if tree_node else ""
        text = title + " " + " ".join([t for t in sections.values() if t])
        if text.strip():
            return text
    return tree_to_full_text(tree_node)

def get_document_section_text(doc_code, sec, full_texts, tree_node) -> str:
    if full_texts and doc_code in full_texts:
        return full_texts[doc_code].get(sec, "")
    for child in tree_node.children:
        if child.schema_class == sec:
            return " ".join([leaf.label for leaf in child.children])
    return ""

# ── B1: Cosine TF-IDF ──
def get_cosine_tfidf_similarity(trees_dict, pairs_df, full_texts=None):
    docs = {k: get_document_full_text(k, full_texts, v) for k, v in trees_dict.items()}
    keys = list(docs.keys())
    texts = [docs[k] for k in keys]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    similarities = []
    for _, row in pairs_df.iterrows():
        idx_a = key_to_idx[row.doc_a]
        idx_b = key_to_idx[row.doc_b]
        vec_a = tfidf_matrix[idx_a]
        vec_b = tfidf_matrix[idx_b]
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        similarities.append(sim)
    return similarities

# ── B2: Cosine SBERT ──
def get_sbert_similarity(trees_dict, pairs_df, full_texts=None, model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Warning: SBERT loading failed ({e}). Falling back to TF-IDF cosine similarity for B2.")
        return get_cosine_tfidf_similarity(trees_dict, pairs_df, full_texts)
        
    docs = {k: get_document_full_text(k, full_texts, v) for k, v in trees_dict.items()}
    keys = list(docs.keys())
    texts = [docs[k] for k in keys]
    
    embeddings = model.encode(texts, show_progress_bar=False)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    
    similarities = []
    for _, row in pairs_df.iterrows():
        idx_a = key_to_idx[row.doc_a]
        idx_b = key_to_idx[row.doc_b]
        emb_a = embeddings[idx_a].reshape(1, -1)
        emb_b = embeddings[idx_b].reshape(1, -1)
        sim = cosine_similarity(emb_a, emb_b)[0][0]
        similarities.append(sim)
    return similarities

# ── B3: TED-Standard ──
def iter_nodes(node):
    yield node
    for child in node.children:
        yield from iter_nodes(child)

def _eval_single_sted(args):
    tree_a_dict, tree_b_dict = args
    from src.node import CapstoneNode
    tree_a = CapstoneNode.from_dict(tree_a_dict)
    tree_b = CapstoneNode.from_dict(tree_b_dict)
    cost_model = StandardCostModel()
    config = apted.Config()
    config.rename = lambda u, v: cost_model.w_rep(u, v)
    config.delete = lambda u: cost_model.w_del(u)
    config.insert = lambda v: cost_model.w_ins(v)
    
    runner = apted.APTED(tree_a, tree_b, config)
    dist = runner.compute_edit_distance()
    
    self_a = sum(1 for _ in iter_nodes(tree_a))
    self_b = sum(1 for _ in iter_nodes(tree_b))
    denom = self_a + self_b
    return 1 - dist / denom if denom > 0 else 1.0

def get_standard_ted_similarity(trees_dict, pairs_df):
    from concurrent.futures import ProcessPoolExecutor
    args_list = []
    for _, row in pairs_df.iterrows():
        args_list.append((trees_dict[row.doc_a].to_dict(), trees_dict[row.doc_b].to_dict()))
        
    with ProcessPoolExecutor() as executor:
        similarities = list(executor.map(_eval_single_sted, args_list))
    return similarities

# ── B4: pq-Gram ──
def get_pq_grams(node, p=2, q=3, ancestors=None):
    if ancestors is None:
        ancestors = tuple(["*"] * (p - 1))
    curr_stem = ancestors[-(p-1):] + (node.label,) if p > 1 else (node.label,)
    
    pq_grams = []
    child_labels = [c.label for c in node.children]
    padded_children = ["*"] * (q - 1) + child_labels + ["*"] * (q - 1)
    
    if child_labels:
        for i in range(len(child_labels) + q - 1):
            sibling_window = tuple(padded_children[i : i + q])
            pq_grams.append((curr_stem, sibling_window))
    else:
        pq_grams.append((curr_stem, tuple(["*"] * q)))
        
    for child in node.children:
        pq_grams.extend(get_pq_grams(child, p, q, curr_stem))
    return pq_grams

def get_pqgram_similarity(trees_dict, pairs_df, p=2, q=3):
    similarities = []
    for _, row in pairs_df.iterrows():
        tree_a = trees_dict[row.doc_a]
        tree_b = trees_dict[row.doc_b]
        
        grams_a = get_pq_grams(tree_a, p, q)
        grams_b = get_pq_grams(tree_b, p, q)
        
        c_a = Counter(grams_a)
        c_b = Counter(grams_b)
        
        intersection = sum((c_a & c_b).values())
        union = sum((c_a | c_b).values())
        
        sim = intersection / union if union > 0 else 1.0
        similarities.append(sim)
    return similarities

# ── B5: Section Cosine ──
def get_section_cosine_similarity(trees_dict, pairs_df, full_texts=None):
    SECTION_WEIGHTS = {
        "Context": 0.10,
        "Problem": 0.15,
        "Solution": 0.25,
        "Theory": 0.15,
        "Deliverables": 0.10,
        "Methodology": 0.15,
        "Timeline": 0.05,
        "References": 0.05
    }
    
    all_sections = list(SECTION_WEIGHTS.keys())
    sec_tfidf = {}
    sec_key_to_idx = {}
    
    for sec in all_sections:
        texts = []
        keys = list(trees_dict.keys())
        for k in keys:
            texts.append(get_document_section_text(k, sec, full_texts, trees_dict[k]))
        
        if any(t.strip() != "" for t in texts):
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            sec_tfidf[sec] = tfidf_matrix
            sec_key_to_idx[sec] = {k: i for i, k in enumerate(keys)}
            
    similarities = []
    for _, row in pairs_df.iterrows():
        weighted_sum = 0.0
        weight_total = 0.0
        
        for sec, weight in SECTION_WEIGHTS.items():
            if sec in sec_tfidf:
                idx_map = sec_key_to_idx[sec]
                idx_a = idx_map[row.doc_a]
                idx_b = idx_map[row.doc_b]
                
                text_a = get_document_section_text(row.doc_a, sec, full_texts, trees_dict[row.doc_a]).strip()
                text_b = get_document_section_text(row.doc_b, sec, full_texts, trees_dict[row.doc_b]).strip()
                
                if not text_a and not text_b:
                    sim = 1.0
                elif not text_a or not text_b:
                    sim = 0.0
                else:
                    vec_a = sec_tfidf[sec][idx_a]
                    vec_b = sec_tfidf[sec][idx_b]
                    sim = cosine_similarity(vec_a, vec_b)[0][0]
                    
                weighted_sum += weight * sim
                weight_total += weight
                
        final_sim = weighted_sum / weight_total if weight_total > 0 else 1.0
        similarities.append(final_sim)
    return similarities
