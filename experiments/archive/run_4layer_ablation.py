import os
import sys
import json
import pickle
import yaml
import time
import copy
import pandas as pd
import numpy as np
import spacy
import importlib
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binom, wilcoxon
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.node import CapstoneNode
importlib.invalidate_caches()
sw_bted = importlib.import_module("src.05_sw_bted")

# Force UTF-8 stdout
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

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

_worker_cso_graph = None
_worker_trees = {}

def _init_worker(cso_graph_path, max_depth_val, trees_dict_raw):
    global _worker_cso_graph, _worker_trees
    with open(cso_graph_path, "rb") as f:
        cso_data = pickle.load(f)
        _worker_cso_graph = cso_data["graph"]
    _worker_trees = {k: CapstoneNode.from_dict(v) for k, v in trees_dict_raw.items()}

# In-memory converter to map old 6-layer trees to 4-layer structures
def convert_6l_to_4l(root: CapstoneNode) -> CapstoneNode:
    new_root = CapstoneNode(
        label=root.label,
        schema_class="MacroFilter",
        depth=1,
        embedding=root.embedding
    )
    domain_nodes = {
        "D1_BUSINESS_CONTEXT": CapstoneNode(label="D1_BUSINESS_CONTEXT", schema_class="D1_BUSINESS_CONTEXT", depth=2),
        "D2_FUNCTIONAL": CapstoneNode(label="D2_FUNCTIONAL", schema_class="D2_FUNCTIONAL", depth=2),
        "D3_TECHNICAL_REALIZATION": CapstoneNode(label="D3_TECHNICAL_REALIZATION", schema_class="D3_TECHNICAL_REALIZATION", depth=2),
        "D4_EXECUTION_PLANNING": CapstoneNode(label="D4_EXECUTION_PLANNING", schema_class="D4_EXECUTION_PLANNING", depth=2)
    }
    mapping = {
        "Context": "D1_BUSINESS_CONTEXT",
        "D1_BUSINESS_CONTEXT": "D1_BUSINESS_CONTEXT",
        "FR": "D2_FUNCTIONAL",
        "Solution": "D2_FUNCTIONAL",
        "Products": "D2_FUNCTIONAL",
        "D2_FUNCTIONAL": "D2_FUNCTIONAL",
        "NFR": "D3_TECHNICAL_REALIZATION",
        "Theory": "D3_TECHNICAL_REALIZATION",
        "D3_TECHNICAL_REALIZATION": "D3_TECHNICAL_REALIZATION",
        "Tasks": "D4_EXECUTION_PLANNING",
        "D4_EXECUTION_PLANNING": "D4_EXECUTION_PLANNING"
    }
    for domain in root.children:
        target_domain_name = mapping.get(domain.schema_class)
        if not target_domain_name:
            continue
        target_domain = domain_nodes[target_domain_name]
        
        atomic_reqs = []
        def collect_atomic_reqs(node):
            if node.depth == 4 or node.schema_class == "AtomicReq":
                atomic_reqs.append(node)
            elif node.depth == 3:
                if not node.children or all(c.depth != 4 for c in node.children):
                    atomic_reqs.append(node)
            for child in node.children:
                collect_atomic_reqs(child)
        collect_atomic_reqs(domain)
        
        for ar in atomic_reqs:
            new_ar = CapstoneNode(
                label=ar.label,
                schema_class="IntentMatching",
                depth=3,
                raw_text=ar.raw_text,
                normalized_text=ar.normalized_text,
                embedding=ar.embedding,
                feature_label=ar.feature_label
            )
            target_domain.children.append(new_ar)
            
            leaves = []
            def collect_leaves(node):
                if node.depth == 6 or node.schema_class in ("ConceptKeyword", "TechKeyword", "TerminologyVerification"):
                    if node.children == []:
                        leaves.append(node)
                for child in node.children:
                    collect_leaves(child)
            collect_leaves(ar)
            
            seen_leaves = set()
            for leaf in leaves:
                lbl = leaf.label
                if lbl in seen_leaves:
                    continue
                seen_leaves.add(lbl)
                
                role = "concept"
                if leaf.schema_class in ("TechKeyword", "Technology"):
                    role = "technology"
                elif leaf.source_role:
                    role = leaf.source_role
                
                tfidf_w = getattr(leaf, 'tfidf_weight', None)
                if tfidf_w is None:
                    tfidf_w = getattr(leaf, 'idf', None)
                if tfidf_w is None:
                    tfidf_w = 0.5
                    
                new_leaf = CapstoneNode(
                    label=leaf.label,
                    schema_class="TerminologyVerification",
                    depth=4,
                    source_role=role,
                    tfidf_weight=tfidf_w
                )
                new_ar.children.append(new_leaf)
                
    for d_name in ["D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"]:
        new_root.children.append(domain_nodes[d_name])
        
    return new_root

def populate_embeddings(trees_raw: dict, model: SentenceTransformer):
    all_t3_nodes = []
    roots = {}
    for code, tree_dict in trees_raw.items():
        root_node = CapstoneNode.from_dict(tree_dict)
        roots[code] = root_node
        def collect(node):
            if node.depth == 3 and node.schema_class == "IntentMatching":
                all_t3_nodes.append(node)
            for child in node.children:
                collect(child)
        collect(root_node)
        
    if all_t3_nodes:
        t3_texts = [n.normalized_text if n.normalized_text else (n.raw_text if n.raw_text else "None") for n in all_t3_nodes]
        t3_embs = model.encode(t3_texts, batch_size=256, show_progress_bar=False)
        for n, emb in zip(all_t3_nodes, t3_embs):
            n.embedding = emb.tolist()
        
    root_texts = []
    root_codes = list(roots.keys())
    for code in root_codes:
        r = roots[code]
        texts = []
        def traverse(node):
            if node.depth == 3:
                val = node.normalized_text if node.normalized_text else node.raw_text
                if val:
                    texts.append(val)
            for child in node.children:
                traverse(child)
        traverse(r)
        doc_text = " ".join(texts) if texts else "None"
        root_texts.append(doc_text)
        
    root_embs = model.encode(root_texts, batch_size=256, show_progress_bar=False)
    for code, emb in zip(root_codes, root_embs):
        roots[code].embedding = emb.tolist()
        
    return {code: r.to_dict() for code, r in roots.items()}

# In-Memory tree transformation helpers
def remove_t4_nodes(root: CapstoneNode) -> CapstoneNode:
    new_root = copy.deepcopy(root)
    def traverse(node):
        if node.depth == 3 and node.schema_class == "IntentMatching":
            node.children = []
        for child in node.children:
            traverse(child)
    traverse(new_root)
    return new_root

def remove_t2_nodes(root: CapstoneNode) -> CapstoneNode:
    new_root = CapstoneNode(
        label=root.label,
        schema_class="MacroFilter",
        depth=1,
        embedding=root.embedding
    )
    t3_nodes = []
    for t2 in root.children:
        for t3 in t2.children:
            t3_clone = copy.deepcopy(t3)
            t3_clone.depth = 2
            for t4 in t3_clone.children:
                t4.depth = 3
            t3_nodes.append(t3_clone)
    new_root.children = t3_nodes
    return new_root

def extract_keywords_simple(text: str, top_k=8) -> List[str]:
    if not text:
        return []
    cleaned = text.replace("●", " ").replace("•", " ").replace("-", " ").replace("*", " ").replace("\t", " ")
    words = [w.lower().strip() for w in cleaned.split() if w.strip()]
    filtered = []
    for w in words:
        w_clean = "".join([c for c in w if c.isalnum() or c in ("-", "_")])
        if len(w_clean) > 2 and w_clean not in STOPWORDS:
            filtered.append(w_clean)
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
            if sec == "FR": text = sections.get("functional_requirement", "")
            elif sec == "NFR": text = sections.get("nonfunctional_requirement", "")
            elif sec == "Tasks": text = sections.get("proposed_tasks", "")
        if not text:
            continue
        sec_node = CapstoneNode(label=sec, schema_class=sec, depth=2)
        root.children.append(sec_node)
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

# Custom Normalization for Group C
_LEAF_NORM_CACHE = {}

def normalize_leaf_custom(label: str, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups) -> str:
    cache_key = (label, use_cso, use_tem)
    if cache_key in _LEAF_NORM_CACHE:
        return _LEAF_NORM_CACHE[cache_key]
        
    tokens = label.lower().strip().split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    if not filtered_tokens:
        filtered_tokens = tokens
    phrase = " ".join(filtered_tokens)
    res = None
    if use_cso and cso_lookup:
        cso_res = cso_lookup.lookup(phrase)
        if cso_res:
            res = cso_res["cso_concept"]
    if use_tem and not res:
        if phrase in tech_groups:
            res = tech_groups[phrase]
        else:
            for t in filtered_tokens:
                if t in tech_groups:
                    res = tech_groups[t]
                    break
    if not res:
        doc = nlp_sm(phrase)
        res = " ".join([token.lemma_ for token in doc])
        
    _LEAF_NORM_CACHE[cache_key] = res
    return res

def apply_custom_normalization(unnorm_tree_6l: CapstoneNode, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups) -> CapstoneNode:
    root = copy.deepcopy(unnorm_tree_6l)
    def traverse_and_normalize(node):
        if node.depth == 6 or node.schema_class in ("ConceptKeyword", "TechKeyword", "Technology"):
            if node.children == []:
                norm_lbl = normalize_leaf_custom(node.label, cso_lookup, nlp_sm, use_cso, use_tem, tech_groups)
                node.label = norm_lbl
                if norm_lbl in tech_groups.values() or any(k in norm_lbl for k in tech_groups):
                    node.schema_class = "TechKeyword"
                else:
                    node.schema_class = "ConceptKeyword"
        for child in node.children:
            traverse_and_normalize(child)
    traverse_and_normalize(root)
    return convert_6l_to_4l(root)

# Custom Cost Model for Ablations
class CustomSWCostModel(sw_bted.SWCostModel):
    def __init__(self, variant: str, cso_graph=None, max_depth=19, beta_override=None):
        super().__init__(cso_graph=cso_graph, max_depth=max_depth)
        self.variant = variant
        self.beta_override = beta_override

    def w_del(self, u: CapstoneNode) -> float:
        if u.depth == 1:
            return 0.0
        if self.variant == "no_T2":
            if u.schema_class == "IntentMatching":
                return 1.0
            if u.schema_class == "TerminologyVerification":
                weight = getattr(u, 'tfidf_weight', None)
                if weight is None: weight = 0.5
                return 0.5 * weight
            return 1.0
        else:
            return super().w_del(u)

    def w_ins(self, v: CapstoneNode) -> float:
        if v.depth == 1:
            return 0.0
        if self.variant == "no_T2":
            if v.schema_class == "IntentMatching":
                return 1.0
            if v.schema_class == "TerminologyVerification":
                weight = getattr(v, 'tfidf_weight', None)
                if weight is None: weight = 0.5
                return 0.5 * weight
            return 1.0
        else:
            return super().w_ins(v)

    def dist_content(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.schema_class != v.schema_class:
            return 1.0
        if u.schema_class == "IntentMatching":
            return 1.0 - sw_bted.cosine_sim(u.embedding, v.embedding)
        elif u.schema_class == "TerminologyVerification":
            return 0.0 if u.label == v.label else 1.0
        elif u.depth == 2 and self.variant != "no_T2":
            return 0.0 if u.label == v.label else 1.0
        return 1.0

    def dist_schema(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.schema_class != v.schema_class:
            return 1.0
        if u.depth == 2 and self.variant != "no_T2":
            return sw_bted.DOMAIN_SCHEMA_DIST.get((u.label, v.label), 1.0)
        return 0.0 if u.schema_class == v.schema_class else 1.0

    def w_rep(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.schema_class != v.schema_class:
            return self.w_del(u) + self.w_ins(v)
        if u.depth == 1:
            return 0.0
        
        if self.beta_override is not None:
            # Group B uniform, content-only, schema-only overrides
            # but Domain T2 should always use beta_l = 0.0 because it is categorical
            if u.depth == 2 and self.variant != "no_T2":
                beta_l = 0.0
            else:
                beta_l = self.beta_override
        else:
            if u.schema_class == "IntentMatching":
                beta_l = 0.9
            elif u.schema_class == "TerminologyVerification":
                beta_l = 1.0
            else:
                beta_l = 0.0
            
        content_d = self.dist_content(u, v)
        schema_d = self.dist_schema(u, v)
        return (self.w_del(u) + self.w_ins(v)) * (beta_l * content_d + (1.0 - beta_l) * schema_d)

def _eval_single_pair(args):
    doc_a, doc_b, variant, alpha, beta_override = args
    global _worker_cso_graph, _worker_trees
    
    cost_model = CustomSWCostModel(variant=variant, cso_graph=_worker_cso_graph, beta_override=beta_override)
    cost_model.alpha = alpha
    
    tree_a = _worker_trees[doc_a]
    tree_b = _worker_trees[doc_b]
    
    if variant == "no_T4":
        tree_a = remove_t4_nodes(tree_a)
        tree_b = remove_t4_nodes(tree_b)
    elif variant == "no_T2":
        tree_a = remove_t2_nodes(tree_a)
        tree_b = remove_t2_nodes(tree_b)
        
    if cost_model.alpha == 0.0:
        if hasattr(tree_a, 'embedding') and hasattr(tree_b, 'embedding') and tree_a.embedding and tree_b.embedding:
            a = np.array(tree_a.embedding)
            b = np.array(tree_b.embedding)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0: return 0.0
            return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
        return 0.0
        
    return sw_bted.normalize_similarity(tree_a, tree_b, cost_model)

class StandardCostModel:
    def w_del(self, u) -> float: return 1.0
    def w_ins(self, v) -> float: return 1.0
    def w_rep(self, u, v) -> float: return 0.0 if u.label == v.label else 2.0

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
        for child in node.children: yield from iter_nodes(child)
    runner = apted.APTED(tree_a, tree_b, config)
    dist = runner.compute_edit_distance()
    denom = sum(1 for _ in iter_nodes(tree_a)) + sum(1 for _ in iter_nodes(tree_b))
    return 1.0 - dist / denom if denom > 0 else 1.0

def find_best_threshold(similarities, labels):
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
        if sw_correct and not base_correct: b += 1
        elif not sw_correct and base_correct: c += 1
    n = b + c
    if n == 0: return 0.0, 1.0
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    p_value = 2 * binom.cdf(min(b, c), n, 0.5)
    p_value = min(p_value, 1.0)
    return chi2_stat, p_value

def main():
    print("="*60)
    print("SW-BTED 4-LAYER MASTER ABLATION ENGINE (19 VARIANTS)")
    print("="*60)
    
    # 1. Load Datasets
    print("\n[1] Loading datasets...")
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw_6l = json.load(open("data/dataset/trees_t6_unnormalized.json", encoding="utf-8"))
    fpt_full_texts = json.load(open("data/dataset/full_texts.json", encoding="utf-8"))
    
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw_6l = json.load(open("datasets/pure_adapted/pure_trees_unnormalized.json", encoding="utf-8"))
    pure_pseudo_docs = json.load(open("datasets/pure_adapted/pseudo_documents.json", encoding="utf-8"))
    
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    max_depth = cso_data.get("max_depth", 19)
    
    # Init CSO Lookup and SpaCy for Group C
    nlp_sm = spacy.load("en_core_web_sm")
    tech_module = importlib.import_module("src.tech_equivalence")
    TECH_GROUPS = tech_module.TECH_GROUPS
    ontology_lookup_module = importlib.import_module("src.03_ontology_lookup")
    cso_lookup = ontology_lookup_module.CSOLookup()
    def quick_lookup(label: str) -> bool:
        label_clean = label.strip().lower()
        if label_clean in cso_lookup.label_to_concept:
            concept = cso_lookup.label_to_concept[label_clean]
        elif label_clean in cso_lookup.concept_to_label:
            concept = label_clean
        else: return None
        return {"cso_concept": concept, "label": cso_lookup.concept_to_label[concept]}
    cso_lookup.lookup = quick_lookup
    
    # Define the 19 variants config: (group, variant_id, variant_name, description)
    VARIANTS = [
        # Group A: Layer Structure Ablation
        ("A", "A1", "SW-BTED_4L", "Proposed 4-layer (baseline of Group A)"),
        ("A", "A2", "SW-BTED_4L_no_T4", "Bypass T4 (Terminology Verification)"),
        ("A", "A3", "SW-BTED_4L_no_T2", "Bypass T2 (Domain Partition)"),
        ("A", "A4", "SW-BTED_3L", "Legacy 3-layer flat model"),
        
        # Group B: Cost Function Ablation
        ("B", "B1", "SW-BTED_beta_specific", "Proposed per-layer betas (baseline of Group B)"),
        ("B", "B2", "SW-BTED_beta_uniform", "Uniform beta = 0.5"),
        ("B", "B3", "SW-BTED_beta_content_only", "Beta = 1.0 (Content only)"),
        ("B", "B4", "SW-BTED_beta_schema_only", "Beta = 0.0 (Schema only)"),
        
        # Group C: Normalization Ablation
        ("C", "C1", "SW-BTED_full_norm", "Proposed TEM + CSO (baseline of Group C)"),
        ("C", "C2", "SW-BTED_no_TEM", "Disables TEM"),
        ("C", "C3", "SW-BTED_no_CSO", "Disables CSO"),
        ("C", "C4", "SW-BTED_no_norm", "Disables both CSO and TEM"),
        
        # Group D: Alpha Weight Ablation
        ("D", "D1", "SW-BTED_alpha_0.0", "Alpha = 0.0 (Embedding only)"),
        ("D", "D2", "SW-BTED_alpha_0.2", "Alpha = 0.2"),
        ("D", "D3", "SW-BTED_alpha_0.4", "Alpha = 0.4"),
        ("D", "D4", "SW-BTED_alpha_0.6", "Proposed Alpha = 0.6 (baseline of Group D)"),
        ("D", "D5", "SW-BTED_alpha_0.8", "Alpha = 0.8"),
        ("D", "D6", "SW-BTED_alpha_1.0", "Alpha = 1.0 (TED only)")
    ]
    
    master_results = {"FPT": {}, "PURE": {}}
    num_workers = min(os.cpu_count(), 8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    datasets_info = [
        ("FPT", fpt_pairs, fpt_trees_raw_6l, fpt_full_texts, None),
        ("PURE", pure_pairs, pure_trees_raw_6l, None, pure_pseudo_docs)
    ]
    
    for ds_name, ds_pairs, ds_trees_6l, ds_full_texts, ds_pseudo_docs in datasets_info:
        print("\n" + "="*50)
        print(f"EVALUATING DATASET: {ds_name}")
        print("="*50)
        
        labels = ds_pairs["label"].to_numpy()
        strat_labels = ds_pairs["type"].to_numpy() if "type" in ds_pairs else labels
        
        for group, var_id, var_name, var_desc in VARIANTS:
            print(f"\n>>> Running Variant {var_id}: {var_name}...")
            start_time = time.time()
            
            # Rebuild/Transform trees
            variant_trees = {}
            is_sted = False
            
            # Default proposed params
            alpha = 0.6
            beta_override = None
            
            if group == "A" and var_id == "A4":
                # Legacy 3L
                is_sted = True
                if ds_name == "FPT":
                    for k in ds_trees_6l.keys():
                        variant_trees[k] = build_3l_tree_fpt(k, ds_full_texts).to_dict()
                else:
                    for k in ds_trees_6l.keys():
                        variant_trees[k] = build_3l_tree_pure(k, ds_pseudo_docs).to_dict()
            elif group == "C":
                use_cso = (var_id in ("C1", "C2"))
                use_tem = (var_id in ("C1", "C3"))
                for k, v in ds_trees_6l.items():
                    node_6l = CapstoneNode.from_dict(v)
                    variant_trees[k] = apply_custom_normalization(node_6l, cso_lookup, nlp_sm, use_cso, use_tem, TECH_GROUPS).to_dict()
            else:
                # Standard 4L Conversion
                for k, v in ds_trees_6l.items():
                    node_6l = CapstoneNode.from_dict(v)
                    variant_trees[k] = convert_6l_to_4l(node_6l).to_dict()
            
            # Configure variant costs
            if group == "B":
                if var_id == "B2": beta_override = 0.5
                elif var_id == "B3": beta_override = 1.0
                elif var_id == "B4": beta_override = 0.0
            elif group == "D":
                if var_id == "D1": alpha = 0.0
                elif var_id == "D2": alpha = 0.2
                elif var_id == "D3": alpha = 0.4
                elif var_id == "D4": alpha = 0.6
                elif var_id == "D5": alpha = 0.8
                elif var_id == "D6": alpha = 1.0
                
            # Populate embeddings dynamically
            if not is_sted:
                variant_trees = populate_embeddings(variant_trees, sbert_model)
                
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=("data/processed/cso_graph.pkl", max_depth, variant_trees)
            )
            
            if is_sted:
                args_list = [(row.doc_a, row.doc_b) for _, row in ds_pairs.iterrows()]
                similarities = np.array(list(pool.map(_eval_single_sted, args_list)))
            else:
                variant_type = "no_T4" if var_id == "A2" else ("no_T2" if var_id == "A3" else "proposed")
                args_list = [(row.doc_a, row.doc_b, variant_type, alpha, beta_override) for _, row in ds_pairs.iterrows()]
                similarities = np.array(list(pool.map(_eval_single_pair, args_list)))
                
            pool.shutdown()
            
            # Stratified 5-Fold CV
            fold_f1s, fold_precisions, fold_recalls, fold_aucs = [], [], [], []
            cv_preds = np.zeros(len(ds_pairs), dtype=int)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                best_thresh = find_best_threshold(similarities[train_idx], labels[train_idx])
                preds = np.array([1 if s >= best_thresh else 0 for s in similarities[test_idx]])
                cv_preds[test_idx] = preds
                
                fold_precisions.append(precision_score(labels[test_idx], preds, zero_division=0))
                fold_recalls.append(recall_score(labels[test_idx], preds, zero_division=0))
                fold_f1s.append(f1_score(labels[test_idx], preds, zero_division=0))
                try:
                    fold_aucs.append(roc_auc_score(labels[test_idx], similarities[test_idx]))
                except ValueError:
                    fold_aucs.append(0.5)
                    
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
            
    # ── Write Results and Summaries ──
    print("\n[3] Compiling results and summaries...")
    os.makedirs("results/4layer/ablation", exist_ok=True)
    
    GROUP_BASELINES = {"A": "A1", "B": "B1", "C": "C1", "D": "D4"}
    BONFERRONI_THRESHOLDS = {"A": 0.01/3, "B": 0.01/3, "C": 0.01/3, "D": 0.01/5}
    
    master_table_rows = []
    
    for ds_info in datasets_info:
        ds_name = ds_info[0]
        ds_pairs = fpt_pairs if ds_name == "FPT" else pure_pairs
        y_true = ds_pairs["label"].to_numpy()
        group_rows = {"A": [], "B": [], "C": [], "D": []}
        
        for group, var_id, var_name, var_desc in VARIANTS:
            res = master_results[ds_name][var_id]
            metrics = res["metrics"]
            baseline_id = GROUP_BASELINES[group]
            baseline_res = master_results[ds_name][baseline_id]
            delta_f1 = metrics["f1_val"] - baseline_res["metrics"]["f1_val"]
            
            chi2, p_val = run_mcnemar_test(y_true, res["preds"], baseline_res["preds"])
            bonf_thresh = BONFERRONI_THRESHOLDS[group]
            is_sig = "Yes" if p_val < bonf_thresh else "No"
            
            row_dict = {
                "Variant": var_name,
                "Variant_ID": var_id,
                "F1_Score": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "ROC_AUC": metrics["roc_auc"],
                "Delta_F1_vs_Baseline": f"{delta_f1:.4f}",
                "McNemar_p_value": f"{p_val:.4e}",
                "Significant_Bonferroni": is_sig
            }
            group_rows[group].append(row_dict)
            
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
            json_filename = f"{var_id}_{var_name.replace('-', '_').replace(' ', '_')}_{ds_name}.json"
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
                "delta_f1_vs_baseline": float(delta_f1)
            }
            dest_json = os.path.abspath(f"results/4layer/ablation/{json_filename}")
            print(f"Writing JSON: {dest_json}")
            with open(dest_json, "w", encoding="utf-8") as f:
                json.dump(single_var_json, f, ensure_ascii=False, indent=2)
                
        # Save group summarization CSVs
        for g, rows in group_rows.items():
            dest_g_csv = os.path.abspath(f"results/4layer/ablation/group_{g}_summary_{ds_name}.csv")
            print(f"Writing Group CSV: {dest_g_csv}")
            pd.DataFrame(rows).to_csv(dest_g_csv, index=False)
            
    dest_master_csv = os.path.abspath("results/4layer/ablation/ablation_master_table.csv")
    print(f"Writing Master CSV: {dest_master_csv}")
    pd.DataFrame(master_table_rows).to_csv(dest_master_csv, index=False)
    print("\n" + "="*50)
    print("4-LAYER ABLATION STUDY COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
