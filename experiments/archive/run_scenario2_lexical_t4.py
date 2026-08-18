import os
import sys
import json
import pickle
import time
import importlib
import pandas as pd
import numpy as np
import collections
import re
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force UTF-8 stdout
if sys.platform.startswith("win"):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

from src.node import CapstoneNode

# ── Dynamic Imports of SW-BTED ──
sw_bted_module = importlib.import_module("src.05_sw_bted")
SWCostModel = sw_bted_module.SWCostModel
normalize_similarity = sw_bted_module.normalize_similarity
dict_to_node = sw_bted_module.dict_to_node

# ── Multiprocessing Globals and Workers ──
_worker_cso_graph = None
_worker_cost_model = None
_worker_trees = {}
_worker_mode = "sbert"
_worker_sim_cache = {}

def _init_worker(cso_graph_path, max_depth_val, trees_dict_raw, mode, sim_cache):
    global _worker_cso_graph, _worker_cost_model, _worker_trees, _worker_mode, _worker_sim_cache
    import pickle
    import importlib
    
    with open(cso_graph_path, "rb") as f:
        cso_data = pickle.load(f)
        _worker_cso_graph = cso_data["graph"]
        
    sw_bted_mod = importlib.import_module("src.05_sw_bted")
    SWCostModel = sw_bted_mod.SWCostModel
    _worker_cost_model = SWCostModel(cso_graph=_worker_cso_graph, max_depth=max_depth_val)
    
    from src.node import CapstoneNode
    _worker_trees = {k: CapstoneNode.from_dict(v) for k, v in trees_dict_raw.items()}
    _worker_mode = mode
    _worker_sim_cache = sim_cache
    
    # Override dist_content dynamically inside the worker process
    original_dist_content = _worker_cost_model.dist_content
    
    def custom_dist_content(u, v):
        if u.depth == 4 and v.depth == 4:
            text_a = u.normalized_text if u.normalized_text else (u.raw_text if u.raw_text else "")
            text_b = v.normalized_text if v.normalized_text else (v.raw_text if v.raw_text else "")
            text_a = text_a.strip()
            text_b = text_b.strip()
            
            if not text_a or not text_b:
                return 1.0
            if text_a.lower() == text_b.lower():
                return 0.0
                
            if _worker_mode == "sbert":
                return original_dist_content(u, v)
                
            # Dictionary lookup
            key = (text_a, text_b) if text_a < text_b else (text_b, text_a)
            if key in _worker_sim_cache:
                sim = _worker_sim_cache[key]
                return float(1.0 - sim)
                
            # Default fallback Jaccard
            import re
            words_a = set(re.findall(r'\w+', text_a.lower()))
            words_b = set(re.findall(r'\w+', text_b.lower()))
            if not words_a or not words_b:
                return 1.0
            j = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
            return float(1.0 - j)
            
        return original_dist_content(u, v)
        
    _worker_cost_model.dist_content = custom_dist_content

def _eval_single_pair(args):
    doc_a, doc_b, alpha, beta_dict = args
    global _worker_cost_model, _worker_trees
    
    if alpha is not None:
        _worker_cost_model.alpha = alpha
    if beta_dict is not None:
        _worker_cost_model.beta = beta_dict
        _worker_cost_model.beta_param = None
        
    tree_a = _worker_trees[doc_a]
    tree_b = _worker_trees[doc_b]
    
    return normalize_similarity(tree_a, tree_b, _worker_cost_model)

# ── Custom BM25 implementation for sentence similarity ──
class CorpusBM25:
    def __init__(self, corpus_texts: List[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus_texts)
        self.doc_tokens = []
        self.doc_lens = []
        self.df = collections.Counter()
        
        for text in corpus_texts:
            tokens = self.tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] += 1
                
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 1.0
        
        # Precompute IDF
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math_log = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
        
    def compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        tf_counter = collections.Counter(doc_tokens)
        score = 0.0
        for q in query_tokens:
            if q not in tf_counter:
                continue
            tf = tf_counter[q]
            idf = self.idf.get(q, 0.0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * tf * (self.k1 + 1.0) / denom
        return score

    def similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        score_ab = self.compute_bm25_score(tokens_a, tokens_b)
        score_ba = self.compute_bm25_score(tokens_b, tokens_a)
        max_a = self.compute_bm25_score(tokens_a, tokens_a)
        max_b = self.compute_bm25_score(tokens_b, tokens_b)
        denom = max_a + max_b
        if denom == 0:
            return 0.0
        return (score_ab + score_ba) / denom

# ── Heuristic Logic for Adaptive T5 ──
def should_activate_t5(sentence: str) -> bool:
    tokens = sentence.split()
    token_count = len(tokens)
    clause_indicators = (
        sentence.count(',') +
        sentence.count(';') +
        sentence.lower().count(' and ') +
        sentence.lower().count(' which ') +
        sentence.lower().count(' that ') +
        sentence.lower().count(' when ') +
        sentence.lower().count(' where ') +
        sentence.lower().count(' to ') +
        sentence.lower().count(' by ') +
        sentence.lower().count(' using ')
    )
    return token_count > 15 and clause_indicators >= 2

# ── In-Memory Transformations ──
def transform_to_5l_norole(root: CapstoneNode) -> CapstoneNode:
    for domain in root.children:
        for child in domain.children:
            if child.depth == 3: # Group
                for t4 in child.children:
                    leaves = []
                    for t5 in t4.children:
                        leaves.extend(t5.children)
                    t4.children = leaves
            elif child.depth == 4: # AtomicReq directly
                leaves = []
                for t5 in child.children:
                    leaves.extend(t5.children)
                child.children = leaves
    return root

def transform_to_adaptive_t5(root: CapstoneNode) -> CapstoneNode:
    def traverse(node):
        if node.depth == 4:
            text = node.normalized_text if node.normalized_text else (node.raw_text if node.raw_text else "")
            if not should_activate_t5(text):
                # Bypass T5
                leaves = []
                for t5 in node.children:
                    leaves.extend(t5.children)
                node.children = leaves
        else:
            for child in node.children:
                traverse(child)
    traverse(root)
    return root

def find_best_threshold(similarities: np.ndarray, labels: np.ndarray) -> float:
    best_thresh = 0.10
    best_f1 = -1.0
    for thresh in np.arange(0.0, 1.01, 0.01):
        preds = [1 if s >= thresh else 0 for s in similarities]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def precompute_t4_similarities(ds_pairs, ds_trees, mode, tfidf_vectorizer=None, corpus_bm25=None):
    sim_cache = {}
    doc_t4_texts = {}
    for doc_id, tree_dict in ds_trees.items():
        root = CapstoneNode.from_dict(tree_dict)
        texts = []
        def collect(n):
            if n.depth == 4:
                txt = n.normalized_text if n.normalized_text else (n.raw_text if n.raw_text else "")
                txt = txt.strip()
                if txt:
                    texts.append(txt)
            for c in n.children:
                collect(c)
        collect(root)
        doc_t4_texts[doc_id] = list(set(texts))

    sentence_pairs = set()
    for _, row in ds_pairs.iterrows():
        texts_a = doc_t4_texts.get(row.doc_a, [])
        texts_b = doc_t4_texts.get(row.doc_b, [])
        for t_a in texts_a:
            for t_b in texts_b:
                if t_a == t_b:
                    continue
                key = (t_a, t_b) if t_a < t_b else (t_b, t_a)
                sentence_pairs.add(key)
                
    print(f"Precomputing {len(sentence_pairs)} unique sentence pairs for mode: {mode}...")
    
    if mode == "tfidf" and tfidf_vectorizer is not None:
        all_unique_texts = list(set(t for pair in sentence_pairs for t in pair))
        if all_unique_texts:
            vectors = tfidf_vectorizer.transform(all_unique_texts)
            text_to_vec = {text: vec for text, vec in zip(all_unique_texts, vectors)}
            for t_a, t_b in sentence_pairs:
                vec_a = text_to_vec[t_a]
                vec_b = text_to_vec[t_b]
                sim = vec_a.dot(vec_b.T)[0, 0]
                sim_cache[f"{t_a}|||{t_b}"] = float(sim)
                
    elif mode == "bm25" and corpus_bm25 is not None:
        for t_a, t_b in sentence_pairs:
            sim = corpus_bm25.similarity(t_a, t_b)
            sim_cache[f"{t_a}|||{t_b}"] = float(sim)
            
    elif mode == "jaccard":
        for t_a, t_b in sentence_pairs:
            words_a = set(re.findall(r'\w+', t_a.lower()))
            words_b = set(re.findall(r'\w+', t_b.lower()))
            if not words_a or not words_b:
                sim = 0.0
            else:
                sim = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
            sim_cache[f"{t_a}|||{t_b}"] = float(sim)
            
    # Convert cache keys back to tuple for lookup inside custom_dist_content
    sim_cache_tuple = {}
    for k, v in sim_cache.items():
        parts = k.split("|||")
        sim_cache_tuple[(parts[0], parts[1])] = v
        
    return sim_cache_tuple

def main():
    print("="*70)
    print("SW-BTED SCENARIO 2 EXPERIMENT: T4 LEXICAL SIMILARITY")
    print("="*70)
    
    # ── Load Datasets ──
    print("\n[1] Loading datasets...")
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw = json.load(open("datasets/pure_adapted/pure_trees.json", encoding="utf-8"))
    
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    
    # Collect all T4 sentences for lexical index fitting
    def collect_all_t4_texts(trees_raw):
        texts = []
        for tree_dict in trees_raw.values():
            root = CapstoneNode.from_dict(tree_dict)
            def collect(n):
                if n.depth == 4:
                    txt = n.normalized_text if n.normalized_text else (n.raw_text if n.raw_text else "")
                    txt = txt.strip()
                    if txt:
                        texts.append(txt)
                for c in n.children:
                    collect(c)
            collect(root)
        return list(set(texts))
        
    fpt_t4_texts = collect_all_t4_texts(fpt_trees_raw)
    pure_t4_texts = collect_all_t4_texts(pure_trees_raw)
    
    print(f"FPT total unique T4 sentences: {len(fpt_t4_texts)}")
    print(f"PURE total unique T4 sentences: {len(pure_t4_texts)}")
    
    # Fit TF-IDF and BM25 globally per dataset
    print("\n[2] Fitting TF-IDF Vectorizers and BM25 Models...")
    fpt_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    fpt_tfidf.fit(fpt_t4_texts)
    
    pure_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    pure_tfidf.fit(pure_t4_texts)
    
    fpt_bm25 = CorpusBM25(fpt_t4_texts)
    pure_bm25 = CorpusBM25(pure_t4_texts)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    num_workers = min(os.cpu_count(), 8)
    
    datasets_info = [
        ("FPT", fpt_pairs, fpt_trees_raw, fpt_tfidf, fpt_bm25),
        ("PURE", pure_pairs, pure_trees_raw, pure_tfidf, pure_bm25)
    ]
    
    modes = ["sbert", "tfidf", "bm25", "jaccard"]
    t5_configs = [
        ("A1", "T5 always ON"),
        ("A_new", "T5 ADAPTIVE")
    ]
    
    results = []
    
    for ds_name, ds_pairs, ds_trees, tfidf_model, bm25_model in datasets_info:
        print("\n" + "="*60)
        print(f"EVALUATING DATASET: {ds_name}")
        print("="*60)
        
        labels = ds_pairs["label"].to_numpy()
        strat_labels = ds_pairs["type"].to_numpy() if "type" in ds_pairs else labels
        
        # Precompute caches for lexical modes to save time
        caches = {
            "sbert": {},
            "jaccard": precompute_t4_similarities(ds_pairs, ds_trees, "jaccard"),
            "tfidf": precompute_t4_similarities(ds_pairs, ds_trees, "tfidf", tfidf_model),
            "bm25": precompute_t4_similarities(ds_pairs, ds_trees, "bm25", corpus_bm25=bm25_model)
        }
        
        for t5_id, t5_desc in t5_configs:
            # Transform trees once for this T5 configuration
            transformed_trees = {}
            for k, v in ds_trees.items():
                root = CapstoneNode.from_dict(v)
                if t5_id == "A1":
                    transformed_trees[k] = root
                elif t5_id == "A_new":
                    transformed_trees[k] = transform_to_adaptive_t5(root)
            
            trees_dict_raw = {k: v.to_dict() for k, v in transformed_trees.items()}
            
            for mode in modes:
                print(f"\n>>> Running configuration: {t5_desc} | Mode: {mode.upper()}...")
                
                # Start multiprocessing pool
                pool = ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_worker,
                    initargs=("data/processed/cso_graph.pkl", max_depth, trees_dict_raw, mode, caches[mode])
                )
                
                alpha = 0.6
                beta_dict = {"T2": 0.0, "T3": 0.6, "T4": 0.9, "T5": 0.0, "T6": 0.8}
                args_list = [(row.doc_a, row.doc_b, alpha, beta_dict) for _, row in ds_pairs.iterrows()]
                
                start_time = time.time()
                similarities = np.array(list(pool.map(_eval_single_pair, args_list)))
                pool.shutdown()
                duration = time.time() - start_time
                
                # 5-Fold Cross Validation
                fold_f1s = []
                fold_precisions = []
                fold_recalls = []
                fold_aucs = []
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                    inner_train, inner_val = next(inner_skf.split(ds_pairs.iloc[train_idx], strat_labels[train_idx]))
                    val_idx = train_idx[inner_val]
                    
                    best_thresh = find_best_threshold(similarities[val_idx], labels[val_idx])
                    
                    test_sims = similarities[test_idx]
                    test_labels = labels[test_idx]
                    preds = np.array([1 if s >= best_thresh else 0 for s in test_sims])
                    
                    p = precision_score(test_labels, preds, zero_division=0)
                    r = recall_score(test_labels, preds, zero_division=0)
                    f1 = f1_score(test_labels, preds, zero_division=0)
                    try:
                        auc = roc_auc_score(test_labels, test_sims)
                    except ValueError:
                        auc = 0.5
                        
                    fold_precisions.append(p)
                    fold_recalls.append(r)
                    fold_f1s.append(f1)
                    fold_aucs.append(auc)
                    
                mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
                mean_p, std_p = np.mean(fold_precisions), np.std(fold_precisions)
                mean_r, std_r = np.mean(fold_recalls), np.std(fold_recalls)
                mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
                
                print(f"  Result F1: {mean_f1:.4f} (±{std_f1:.4f}) | Precision: {mean_p:.4f} | Recall: {mean_r:.4f} | ROC-AUC: {mean_auc:.4f} | Time: {duration:.2f}s")
                
                results.append({
                    "Dataset": ds_name,
                    "T5_Config": t5_desc,
                    "T4_Similarity_Mode": mode.upper(),
                    "F1_Score": f"{mean_f1:.4f} ± {std_f1:.4f}",
                    "F1_mean": float(mean_f1),
                    "Precision": f"{mean_p:.4f} ± {std_p:.4f}",
                    "Recall": f"{mean_r:.4f} ± {std_r:.4f}",
                    "ROC_AUC": f"{mean_auc:.4f} ± {std_auc:.4f}",
                    "Duration_s": round(duration, 2)
                })

    # Save results to a CSV table
    os.makedirs("results/scenario2", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/scenario2/scenario2_results.csv", index=False)
    
    # Print Markdown Summary
    print("\n" + "="*70)
    print("EXPERIMENTAL SUMMARY TABLE")
    print("="*70)
    for ds in ["FPT", "PURE"]:
        print(f"\n--- DATASET: {ds} ---")
        sub_df = df_results[df_results["Dataset"] == ds][["T5_Config", "T4_Similarity_Mode", "F1_Score", "Precision", "Recall", "ROC_AUC"]]
        print(sub_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
