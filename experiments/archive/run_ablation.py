import os
import sys
import json
import pickle
import yaml
import time
import importlib
import spacy
import apted
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binom, wilcoxon
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.node import CapstoneNode

# ── Dynamic Imports of SW-BTED ──
sw_bted_module = importlib.import_module("src.05_sw_bted")
SWCostModel = sw_bted_module.SWCostModel
normalize_similarity = sw_bted_module.normalize_similarity
dict_to_node = sw_bted_module.dict_to_node

# Stopwords set for legacy 3-layer keywords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", 
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", 
    "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", 
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "im", "ive", "if", "in", "into", 
    "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
    "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", 
    "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", 
    "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", 
    "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", 
    "your", "yours", "yourself", "yourselves"
}

# ── Multiprocessing Globals and Workers ──
_worker_cso_graph = None
_worker_cost_model = None
_worker_trees = {}

def _init_worker(cso_graph_path, max_depth_val, trees_dict_raw):
    global _worker_cso_graph, _worker_cost_model, _worker_trees
    import pickle
    import importlib
    
    # Load CSO Graph directly
    with open(cso_graph_path, "rb") as f:
        cso_data = pickle.load(f)
        _worker_cso_graph = cso_data["graph"]
        
    sw_bted_mod = importlib.import_module("src.05_sw_bted")
    SWCostModel = sw_bted_mod.SWCostModel
    _worker_cost_model = SWCostModel(cso_graph=_worker_cso_graph, max_depth=max_depth_val)
    
    # Cache CapstoneNode trees in worker memory
    from src.node import CapstoneNode
    _worker_trees = {k: CapstoneNode.from_dict(v) for k, v in trees_dict_raw.items()}

def _eval_single_pair(args):
    doc_a, doc_b, alpha, beta_dict = args
    global _worker_cost_model, _worker_trees
    
    # Configure cost model dynamically in worker
    if alpha is not None:
        _worker_cost_model.alpha = alpha
    if beta_dict is not None:
        _worker_cost_model.beta = beta_dict
        _worker_cost_model.beta_param = None
        
    tree_a = _worker_trees[doc_a]
    tree_b = _worker_trees[doc_b]
    
    # Optimize for alpha = 0.0 (no structural TED needed)
    if _worker_cost_model.alpha == 0.0:
        if hasattr(tree_a, 'embedding') and hasattr(tree_b, 'embedding') and tree_a.embedding and tree_b.embedding:
            a = np.array(tree_a.embedding)
            b = np.array(tree_b.embedding)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
        return 0.0
        
    return normalize_similarity(tree_a, tree_b, _worker_cost_model)

# Standard TED Cost Model for legacy 3-layer tree
class StandardCostModel:
    def w_del(self, u) -> float:
        return 1.0
    def w_ins(self, v) -> float:
        return 1.0
    def w_rep(self, u, v) -> float:
        return 0.0 if u.label == v.label else 2.0

def _eval_single_sted(args):
    doc_a, doc_b = args
    global _worker_trees
    
    import apted
    tree_a = _worker_trees[doc_a]
    tree_b = _worker_trees[doc_b]
    
    cost_model = StandardCostModel()
    config = apted.Config()
    config.rename = lambda u, v: cost_model.w_rep(u, v)
    config.delete = lambda u: cost_model.w_del(u)
    config.insert = lambda v: cost_model.w_ins(v)
    
    def iter_nodes(node):
        yield node
        for child in node.children:
            yield from iter_nodes(child)
            
    runner = apted.APTED(tree_a, tree_b, config)
    dist = runner.compute_edit_distance()
    
    self_a = sum(1 for _ in iter_nodes(tree_a))
    self_b = sum(1 for _ in iter_nodes(tree_b))
    denom = self_a + self_b
    return 1 - dist / denom if denom > 0 else 1.0


# ── In-Memory Tree Transformation Helpers ──
def clone_tree(node: CapstoneNode) -> CapstoneNode:
    children = [clone_tree(c) for c in node.children]
    return CapstoneNode(
        label=node.label,
        schema_class=node.schema_class,
        depth=node.depth,
        children=children,
        feature_label=node.feature_label,
        raw_text=node.raw_text,
        normalized_text=node.normalized_text,
        embedding=node.embedding,
        cso_ancestors=node.cso_ancestors
    )

def transform_to_5l_norole(root: CapstoneNode) -> CapstoneNode:
    # Bypass T5 (Semantic Role)
    # Loop over T2 Domain children
    for domain in root.children:
        # Loop over Domain children (either T3 Group or T4 AtomicReq directly)
        for child in domain.children:
            if child.depth == 3: # Group
                for t4 in child.children:
                    # Collect all T6 leaves under T5 roles of this T4
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

def transform_to_5l_nogroup(root: CapstoneNode) -> CapstoneNode:
    # Bypass T3 (Group)
    for domain in root.children:
        new_children = []
        for child in domain.children:
            if child.depth == 3: # Group
                # Lift T4 nodes to be direct children of Domain
                new_children.extend(child.children)
            else:
                new_children.append(child)
        domain.children = new_children
    return root

def transform_to_4l(root: CapstoneNode) -> CapstoneNode:
    # Bypass both T3 and T5
    # First, bypass T3 (Group)
    root = transform_to_5l_nogroup(root)
    # Then, bypass T5 (Semantic Role) under all T4 children of Domain
    for domain in root.children:
        for t4 in domain.children:
            leaves = []
            for t5 in t4.children:
                leaves.extend(t5.children)
            t4.children = leaves
    return root


# ── Legacy 3-Layer Tree Construction ──
def extract_keywords_simple(text: str, top_k=8) -> List[str]:
    # Extremely simple keyword extraction: clean, split, lowercase, filter stopwords, take top_k
    if not text:
        return []
    cleaned = text.replace("●", " ").replace("•", " ").replace("-", " ").replace("*", " ").replace("\t", " ")
    words = [w.lower().strip() for w in cleaned.split() if w.strip()]
    filtered = []
    for w in words:
        w_clean = "".join([c for c in w if c.isalnum() or c in ("-", "_")])
        if len(w_clean) > 2 and w_clean not in STOPWORDS:
            filtered.append(w_clean)
            
    # Deduplicate and return top_k
    seen = set()
    unique = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique.append(w)
            if len(unique) >= top_k:
                break
    return unique

def build_3l_tree_fpt(doc_id: str, full_texts: dict) -> CapstoneNode:
    sections = full_texts.get(doc_id, {})
    root = CapstoneNode(label=doc_id, schema_class="Root", depth=1)
    
    sec_names = ["Context", "Solution", "Theory", "FR", "NFR", "Products", "Tasks"]
    
    for sec in sec_names:
        text = sections.get(sec, "")
        if not text:
            if sec == "FR":
                text = sections.get("functional_requirement", "")
            elif sec == "NFR":
                text = sections.get("nonfunctional_requirement", "")
            elif sec == "Tasks":
                text = sections.get("proposed_tasks", "")
                
        if not text:
            continue
            
        sec_node = CapstoneNode(label=sec, schema_class=sec, depth=2)
        root.children.append(sec_node)
        
        # Extract raw keywords
        keywords = extract_keywords_simple(text, top_k=8)
        for kw in keywords:
            kw_node = CapstoneNode(label=kw, schema_class=sec, depth=3)
            sec_node.children.append(kw_node)
            
    return root

def build_3l_tree_pure(doc_id: str, pseudo_docs: dict) -> CapstoneNode:
    doc_data = pseudo_docs.get(doc_id, {})
    requirements = doc_data.get("requirements", [])
    full_text = " ".join(requirements)
    
    root = CapstoneNode(label=doc_id, schema_class="Root", depth=1)
    sec_node = CapstoneNode(label="Functional", schema_class="Functional", depth=2)
    root.children.append(sec_node)
    
    keywords = extract_keywords_simple(full_text, top_k=15)
    for kw in keywords:
        kw_node = CapstoneNode(label=kw, schema_class="Functional", depth=3)
        sec_node.children.append(kw_node)
        
    return root


# ── Leaf Normalization Dynamically for Group C ──
def normalize_leaf_custom(label: str, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups) -> str:
    # 1. Lowercase + remove stopwords
    tokens = label.lower().strip().split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    if not filtered_tokens:
        filtered_tokens = tokens
    phrase = " ".join(filtered_tokens)
    
    res = None
    # 2. CSO lookup
    if use_cso and cso_lookup:
        cso_res = cso_lookup.lookup(phrase)
        if cso_res:
            res = cso_res["cso_concept"]
            
    # 3. Tech Equivalence Map
    if use_tem and not res:
        if phrase in tech_groups:
            res = tech_groups[phrase]
        else:
            for t in filtered_tokens:
                if t in tech_groups:
                    res = tech_groups[t]
                    break
            
    # 4. Lemmatization via spaCy
    if not res:
        doc = nlp_sm(phrase)
        res = " ".join([token.lemma_ for token in doc])
        
    return res

def apply_custom_normalization(unnorm_tree: CapstoneNode, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups) -> CapstoneNode:
    root = clone_tree(unnorm_tree)
    
    # Traverse and find all T6 leaf nodes
    def traverse_and_normalize(node):
        if node.depth == 6:
            norm_lbl = normalize_leaf_custom(node.label, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups)
            node.label = norm_lbl
            # Update schema class
            if norm_lbl in tech_groups.values() or any(k in norm_lbl for k in tech_groups):
                node.schema_class = "TechKeyword"
            else:
                node.schema_class = "ConceptKeyword"
        for child in node.children:
            traverse_and_normalize(child)
            
    traverse_and_normalize(root)
    return root


# ── Giao thức Đánh giá chính (Stratified 5-Fold CV) ──
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

def run_mcnemar_test(y_true, y_pred_sw, y_pred_base):
    b, c = 0, 0
    for gt, p_sw, p_base in zip(y_true, y_pred_sw, y_pred_base):
        sw_correct = (gt == p_sw)
        base_correct = (gt == p_base)
        if sw_correct and not base_correct:
            b += 1
        elif not sw_correct and base_correct:
            c += 1
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    p_value = 2 * binom.cdf(min(b, c), n, 0.5)
    p_value = min(p_value, 1.0)
    return chi2_stat, p_value


def main():
    print("="*60)
    print("SW-BTED MASTER ABLATION STUDY ENGINE (19 VARIANTS)")
    print("="*60)
    
    # ── Load Datasets ──
    print("\n[1] Loading datasets...")
    
    # Dataset-1: FPT Capstone
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    fpt_unnorm_trees_raw = json.load(open("data/dataset/trees_t6_unnormalized.json", encoding="utf-8"))
    fpt_full_texts = json.load(open("data/dataset/full_texts.json", encoding="utf-8"))
    
    # Dataset-2: PURE Adapted
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw = json.load(open("datasets/pure_adapted/pure_trees.json", encoding="utf-8"))
    pure_unnorm_trees_raw = json.load(open("datasets/pure_adapted/pure_trees_unnormalized.json", encoding="utf-8"))
    pure_pseudo_docs = json.load(open("datasets/pure_adapted/pseudo_documents.json", encoding="utf-8"))
    
    # CSO Graph
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    
    print(f"Loaded FPT Dataset: {len(fpt_pairs)} pairs, {len(fpt_trees_raw)} trees.")
    print(f"Loaded PURE Dataset: {len(pure_pairs)} pairs, {len(pure_trees_raw)} trees.")
    
    # Load spaCy for lemmatization in Group C
    print("Loading spaCy model for leaf normalization...")
    nlp_sm = spacy.load("en_core_web_sm")
    
    # Load Tech Equivalence Map
    tech_module = importlib.import_module("src.tech_equivalence")
    TECH_GROUPS = tech_module.TECH_GROUPS
    
    # Initialize CSO Lookup for Group C
    ontology_lookup_module = importlib.import_module("src.03_ontology_lookup")
    CSOLookup = ontology_lookup_module.CSOLookup
    cso_lookup = CSOLookup()
    # Monkey patch to disable fuzzy matching during normalization for speed
    def quick_lookup(label: str) -> bool:
        label_clean = label.strip().lower()
        if label_clean in cso_lookup.label_to_concept:
            concept = cso_lookup.label_to_concept[label_clean]
        elif label_clean in cso_lookup.concept_to_label:
            concept = label_clean
        else:
            return None
        return {
            "cso_concept": concept,
            "label": cso_lookup.concept_to_label[concept]
        }
    cso_lookup.lookup = quick_lookup

    # ── Define 19 Variants Configs ──
    # Format: (group, variant_id, variant_name, description)
    VARIANTS = [
        # Nhóm A: Layer Structure Ablation
        ("A", "A1", "SW-BTED-6L", "Proposed 6-Layer Model (Baseline of Group A)"),
        ("A", "A2", "SW-BTED-5L-noRole", "Bypass T5 (Semantic Role)"),
        ("A", "A3", "SW-BTED-5L-noGroup", "Bypass T3 (Group)"),
        ("A", "A4", "SW-BTED-4L", "Bypass both T3 and T5"),
        ("A", "A5", "SW-BTED-3L", "Legacy 3-Layer tree with uniform costs"),
        
        # Nhóm B: Cost Function Ablation
        ("B", "B1", "SW-BTED-beta_specific", "Proposed per-layer betas (Baseline of Group B)"),
        ("B", "B2", "SW-BTED-beta_uniform", "Uniform beta = 0.5 for all layers"),
        ("B", "B3", "SW-BTED-beta_content_only", "Beta = 1.0 (Content only, schema dist = 0)"),
        ("B", "B4", "SW-BTED-beta_schema_only", "Beta = 0.0 (Schema only, content dist = 0)"),
        
        # Nhóm C: Normalization Ablation
        ("C", "C1", "SW-BTED-full_norm", "Proposed TEM + CSO normalization (Baseline of Group C)"),
        ("C", "C2", "SW-BTED-no_TEM", "Disables Tech Equivalence Map normalization"),
        ("C", "C3", "SW-BTED-no_CSO", "Disables CSO normalization"),
        ("C", "C4", "SW-BTED-no_norm", "Disables both TEM and CSO normalizations"),
        
        # Nhóm D: Alpha Weight Ablation
        ("D", "D1", "SW-BTED-alpha_0.0", "Alpha = 0.0 (Embedding only, no structural TED)"),
        ("D", "D2", "SW-BTED-alpha_0.2", "Alpha = 0.2 (Embedding dominant)"),
        ("D", "D3", "SW-BTED-alpha_0.4", "Alpha = 0.4 (Embedding favored)"),
        ("D", "D4", "SW-BTED-alpha_0.6", "Proposed Alpha = 0.6 (Baseline of Group D)"),
        ("D", "D5", "SW-BTED-alpha_0.8", "Alpha = 0.8 (Structural dominant)"),
        ("D", "D6", "SW-BTED-alpha_1.0", "Alpha = 1.0 (Structural only, no embedding)")
    ]

    # Global tracking of CV test predictions to perform McNemar tests and Wilxocon tests
    # Format: {dataset_name: {variant_id: {preds: np.array, fold_f1s: list, metrics: dict}}}
    master_results = {
        "FPT": {},
        "PURE": {}
    }

    # Setup 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We will execute the study sequentially. 
    # For each dataset, we run all 19 variants.
    datasets_info = [
        ("FPT", fpt_pairs, fpt_trees_raw, fpt_unnorm_trees_raw, fpt_full_texts, None),
        ("PURE", pure_pairs, pure_trees_raw, pure_unnorm_trees_raw, None, pure_pseudo_docs)
    ]
    
    # Process Pool Executor for multiprocessing
    # Load 8 parallel workers for APTED similarity computations
    num_workers = min(os.cpu_count(), 8)
    print(f"\n[2] Initializing ProcessPoolExecutor with {num_workers} workers...")
    
    for ds_name, ds_pairs, ds_trees, ds_unnorm_trees, ds_full_texts, ds_pseudo_docs in datasets_info:
        print("\n" + "="*50)
        print(f"EVALUATING DATASET: {ds_name}")
        print("="*50)
        
        labels = ds_pairs["label"].to_numpy()
        
        # We need stratified labels to split the folds cleanly
        if "type" in ds_pairs:
            strat_labels = ds_pairs["type"].to_numpy()
        else:
            strat_labels = labels # fallback for PURE if type is not present
            
        # Run variants
        for group, var_id, var_name, var_desc in VARIANTS:
            print(f"\n>>> Running Variant {var_id}: {var_name}...")
            start_time = time.time()
            
            # 1. Transform trees in memory according to the variant
            variant_trees = {}
            
            # Flag to indicate if A5 legacy standard edit distance is used
            is_sted = False
            
            if group == "A":
                if var_id == "A1": # Full 6L
                    variant_trees = {k: CapstoneNode.from_dict(v) for k, v in ds_trees.items()}
                elif var_id == "A2": # 5L noRole
                    for k, v in ds_trees.items():
                        root = CapstoneNode.from_dict(v)
                        variant_trees[k] = transform_to_5l_norole(root)
                elif var_id == "A3": # 5L noGroup
                    for k, v in ds_trees.items():
                        root = CapstoneNode.from_dict(v)
                        variant_trees[k] = transform_to_5l_nogroup(root)
                elif var_id == "A4": # 4L
                    for k, v in ds_trees.items():
                        root = CapstoneNode.from_dict(v)
                        variant_trees[k] = transform_to_4l(root)
                elif var_id == "A5": # 3L Legacy
                    is_sted = True
                    if ds_name == "FPT":
                        for k in ds_trees.keys():
                            variant_trees[k] = build_3l_tree_fpt(k, ds_full_texts)
                    else: # PURE
                        for k in ds_trees.keys():
                            variant_trees[k] = build_3l_tree_pure(k, ds_pseudo_docs)
            elif group == "C":
                # Normalization variants: rebuild normalized trees dynamically from unnormalized
                use_cso = (var_id in ("C1", "C2"))
                use_tem = (var_id in ("C1", "C3"))
                
                print(f"  [Group C] Rebuilding trees in memory: CSO={use_cso}, TEM={use_tem}...")
                for k, v in ds_unnorm_trees.items():
                    unnorm_root = CapstoneNode.from_dict(v)
                    variant_trees[k] = apply_custom_normalization(unnorm_root, cso_lookup, nlp_sm, use_cso, use_tem, TECH_GROUPS)
                    
                # Re-generate global embeddings for Root nodes (average of T4 sentence embeddings)
                # Since we changed leaf labels, we don't need to re-encode sentences as they are identical
                # We just need to load the original T4 embeddings
                for k, root in variant_trees.items():
                    orig_root = CapstoneNode.from_dict(ds_trees[k])
                    # Re-map T4 embeddings from original trees to the newly normalized trees
                    # FPT trees are matched by project ID, requirements are in identical order
                    def copy_t4_embs(orig_node, new_node):
                        if orig_node.depth == 4 and new_node.depth == 4:
                            new_node.embedding = orig_node.embedding
                        for orig_child, new_child in zip(orig_node.children, new_node.children):
                            copy_t4_embs(orig_child, new_child)
                    copy_t4_embs(orig_root, root)
                    root.embedding = orig_root.embedding # copy root embedding
            else:
                # Group B and Group D use standard normalized trees
                variant_trees = {k: CapstoneNode.from_dict(v) for k, v in ds_trees.items()}
                
            # 2. Configure Cost Parameters
            alpha = None
            beta_dict = None
            
            # Default proposed parameters
            default_alpha = 0.6
            default_beta = {"T2": 0.0, "T3": 0.6, "T4": 0.9, "T5": 0.0, "T6": 0.8}
            
            if group == "B":
                alpha = default_alpha
                if var_id == "B1":
                    beta_dict = default_beta
                elif var_id == "B2": # Uniform
                    beta_dict = {"T2": 0.5, "T3": 0.5, "T4": 0.5, "T5": 0.5, "T6": 0.5}
                elif var_id == "B3": # Content only
                    beta_dict = {"T2": 1.0, "T3": 1.0, "T4": 1.0, "T5": 1.0, "T6": 1.0}
                elif var_id == "B4": # Schema only
                    beta_dict = {"T2": 0.0, "T3": 0.0, "T4": 0.0, "T5": 0.0, "T6": 0.0}
            elif group == "D":
                beta_dict = default_beta
                if var_id == "D1":
                    alpha = 0.0
                elif var_id == "D2":
                    alpha = 0.2
                elif var_id == "D3":
                    alpha = 0.4
                elif var_id == "D4":
                    alpha = 0.6
                elif var_id == "D5":
                    alpha = 0.8
                elif var_id == "D6":
                    alpha = 1.0
            else:
                # Group A and Group C use proposed cost weights
                alpha = default_alpha
                beta_dict = default_beta
                
            # 3. Parallel Similarity Computation over all pairs
            # Convert variant CapstoneNode trees back to dict to pass to workers
            variant_trees_dict = {k: v.to_dict() for k, v in variant_trees.items()}
            
            print(f"  Parallelizing pair similarities...")
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=("data/processed/cso_graph.pkl", max_depth, variant_trees_dict)
            )
            
            if is_sted:
                # Legacy 3L Standard Edit Distance
                args_list = [(row.doc_a, row.doc_b) for _, row in ds_pairs.iterrows()]
                similarities = np.array(list(pool.map(_eval_single_sted, args_list)))
            else:
                # SW-BTED
                args_list = [(row.doc_a, row.doc_b, alpha, beta_dict) for _, row in ds_pairs.iterrows()]
                similarities = np.array(list(pool.map(_eval_single_pair, args_list)))
                
            pool.shutdown()
            
            # 4. Stratified 5-Fold Cross-Validation
            fold_f1s = []
            cv_preds = np.zeros(len(ds_pairs), dtype=int)
            
            fold_precisions = []
            fold_recalls = []
            fold_aucs = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                # Inner split: 60% Train, 20% Val
                inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                # Split train_idx to get validation fold
                inner_train, inner_val = next(inner_skf.split(ds_pairs.iloc[train_idx], strat_labels[train_idx]))
                val_idx = train_idx[inner_val]
                
                # Find optimal threshold on validation fold
                best_thresh = find_best_threshold(similarities[val_idx], labels[val_idx])
                
                # Evaluate on test fold
                test_sims = similarities[test_idx]
                test_labels = labels[test_idx]
                preds = np.array([1 if s >= best_thresh else 0 for s in test_sims])
                
                cv_preds[test_idx] = preds
                
                # Metrics
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
                
            # Compile results
            mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
            mean_p, std_p = np.mean(fold_precisions), np.std(fold_precisions)
            mean_r, std_r = np.mean(fold_recalls), np.std(fold_recalls)
            mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
            
            duration = time.time() - start_time
            print(f"  -> Done in {duration:.2f}s. F1-score: {mean_f1:.4f} (±{std_f1:.4f})")
            
            master_results[ds_name][var_id] = {
                "preds": cv_preds,
                "fold_f1s": fold_f1s,
                "metrics": {
                    "precision": f"{mean_p:.4f} ± {std_p:.4f}",
                    "recall": f"{mean_r:.4f} ± {std_r:.4f}",
                    "f1": f"{mean_f1:.4f} ± {std_f1:.4f}",
                    "roc_auc": f"{mean_auc:.4f} ± {std_auc:.4f}",
                    "f1_val": mean_f1
                }
            }

    # ── Perform Statistical Significance Tests & Compile Summaries ──
    print("\n[3] Running statistical tests and compiling summary tables...")
    
    # We define the baselines for each group to compare against
    GROUP_BASELINES = {
        "A": "A1", # 6L proposed
        "B": "B1", # beta_specific proposed
        "C": "C1", # full_norm proposed
        "D": "D4"  # alpha_0.6 proposed
    }
    
    # Bonferroni corrected alpha thresholds for each group
    # alpha_corrected = 0.01 / number_of_comparisons_in_group
    BONFERRONI_THRESHOLDS = {
        "A": 0.01 / 4, # 4 comparisons
        "B": 0.01 / 3, # 3 comparisons
        "C": 0.01 / 3, # 3 comparisons
        "D": 0.01 / 5  # 5 comparisons
    }
    
    os.makedirs("results/ablation/group_A_layer", exist_ok=True)
    os.makedirs("results/ablation/group_B_cost", exist_ok=True)
    os.makedirs("results/ablation/group_C_norm", exist_ok=True)
    os.makedirs("results/ablation/group_D_alpha", exist_ok=True)
    
    os.makedirs("ablationTest/group_A_layer", exist_ok=True)
    os.makedirs("ablationTest/group_B_cost", exist_ok=True)
    os.makedirs("ablationTest/group_C_norm", exist_ok=True)
    os.makedirs("ablationTest/group_D_alpha", exist_ok=True)
    
    # We will build summary tables for each group and a master table
    master_table_rows = []
    
    for ds_name in ["FPT", "PURE"]:
        ds_pairs = fpt_pairs if ds_name == "FPT" else pure_pairs
        y_true = ds_pairs["label"].to_numpy()
        
        # Track group rows to write CSVs
        group_rows = {"A": [], "B": [], "C": [], "D": []}
        
        for group, var_id, var_name, var_desc in VARIANTS:
            res = master_results[ds_name][var_id]
            metrics = res["metrics"]
            
            # Compare with group baseline
            baseline_id = GROUP_BASELINES[group]
            baseline_res = master_results[ds_name][baseline_id]
            
            delta_f1 = metrics["f1_val"] - baseline_res["metrics"]["f1_val"]
            
            # McNemar Test
            chi2, p_val = run_mcnemar_test(y_true, res["preds"], baseline_res["preds"])
            
            # Wilcoxon Test on fold F1s
            if var_id == baseline_id:
                w_stat, w_p = 0.0, 1.0
            else:
                diff = np.array(res["fold_f1s"]) - np.array(baseline_res["fold_f1s"])
                if np.all(diff == 0.0):
                    w_stat, w_p = 0.0, 1.0
                else:
                    try:
                        w_stat, w_p = wilcoxon(res["fold_f1s"], baseline_res["fold_f1s"], alternative='two-sided')
                    except Exception:
                        w_stat, w_p = 0.0, 1.0
            
            # Bonferroni significance check
            bonf_thresh = BONFERRONI_THRESHOLDS[group]
            is_sig = "Yes" if p_val < bonf_thresh else "No"
            
            # Format row for CSV
            row_dict = {
                "Variant": var_name,
                "Variant_ID": var_id,
                "F1_Score": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "ROC_AUC": metrics["roc_auc"],
                "Delta_F1_vs_Baseline": f"{delta_f1:.4f}",
                "McNemar_p_value": f"{p_val:.4e}",
                "Wilcoxon_p_value": f"{w_p:.4e}",
                "Significant_Bonferroni": is_sig
            }
            
            group_rows[group].append(row_dict)
            
            # Master table row
            master_table_rows.append({
                "Dataset": ds_name,
                "Group": group,
                "Variant_ID": var_id,
                "Variant_Name": var_name,
                "F1_Score": metrics["f1"].split(" ± ")[0],
                "F1_Std": metrics["f1"].split(" ± ")[1],
                "Precision": metrics["precision"].split(" ± ")[0],
                "Recall": metrics["recall"].split(" ± ")[0],
                "ROC_AUC": metrics["roc_auc"].split(" ± ")[0],
                "Delta_F1": f"{delta_f1:.4f}",
                "McNemar_p": f"{p_val:.4e}",
                "Significant": is_sig
            })
            
            # Save single variant JSON
            single_var_json = {
                "variant_id": var_id,
                "variant_name": var_name,
                "dataset": ds_name,
                "metrics": {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"]
                },
                "mcnemar_vs_baseline": {
                    "chi2": float(chi2),
                    "p_value": float(p_val),
                    "significant": bool(p_val < bonf_thresh)
                },
                "wilcoxon_vs_baseline": {
                    "statistic": float(w_stat),
                    "p_value": float(w_p)
                },
                "delta_f1_vs_baseline": float(delta_f1)
            }
            
            group_folder = {
                "A": "group_A_layer",
                "B": "group_B_cost",
                "C": "group_C_norm",
                "D": "group_D_alpha"
            }[group]
            
            json_filename = f"{var_id}_{var_name.replace('-', '_').replace(' ', '_')}_{ds_name}.json"
            
            # Save to results/
            with open(f"results/ablation/{group_folder}/{json_filename}", "w", encoding="utf-8") as f:
                json.dump(single_var_json, f, ensure_ascii=False, indent=2)
            # Save to ablationTest/
            with open(f"ablationTest/{group_folder}/{json_filename}", "w", encoding="utf-8") as f:
                json.dump(single_var_json, f, ensure_ascii=False, indent=2)
                
        # Save Group summaries to CSV
        group_folders = {
            "A": "group_A_layer",
            "B": "group_B_cost",
            "C": "group_C_norm",
            "D": "group_D_alpha"
        }
        
        for g, rows in group_rows.items():
            df_g = pd.DataFrame(rows)
            csv_filename = f"group_{g}_summary_{ds_name}.csv"
            
            # Save to results/
            df_g.to_csv(f"results/ablation/{group_folders[g]}/{csv_filename}", index=False)
            # Save to ablationTest/
            df_g.to_csv(f"ablationTest/{group_folders[g]}/{csv_filename}", index=False)
            
    # Save Master Table
    df_master = pd.DataFrame(master_table_rows)
    df_master.to_csv("results/ablation/ablation_master_table.csv", index=False)
    df_master.to_csv("ablationTest/ablation_master_table.csv", index=False)
    
    print("\n" + "="*50)
    print("ABLATION STUDY COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("All results, summaries, and CSV tables successfully written to:")
    print(" - results/ablation/")
    print(" - ablationTest/")
    print("="*50)

if __name__ == "__main__":
    main()
