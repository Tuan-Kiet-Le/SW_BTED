import os
import sys
import json
import pickle
import yaml
import time
import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binom
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.node import CapstoneNode
import importlib
sw_bted = importlib.import_module("src.05_sw_bted")

# Force UTF-8 stdout
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Worker Variables
_worker_cso_graph = None
_worker_trees = {}

def _init_worker(cso_graph_path, max_depth_val, trees_dict_raw):
    global _worker_cso_graph, _worker_trees
    with open(cso_graph_path, "rb") as f:
        cso_data = pickle.load(f)
        _worker_cso_graph = cso_data["graph"]
    _worker_trees = {k: CapstoneNode.from_dict(v) for k, v in trees_dict_raw.items()}

# In-memory converter to map old 6-layer trees (like PURE or backup FPT) to new 4-layer representation
def convert_6l_to_4l(root: CapstoneNode) -> CapstoneNode:
    new_root = CapstoneNode(
        label=root.label,
        schema_class="MacroFilter",
        depth=1,
        embedding=root.embedding
    )
    
    # Define the 4 domains
    domain_nodes = {
        "D1_BUSINESS_CONTEXT": CapstoneNode(label="D1_BUSINESS_CONTEXT", schema_class="D1_BUSINESS_CONTEXT", depth=2),
        "D2_FUNCTIONAL": CapstoneNode(label="D2_FUNCTIONAL", schema_class="D2_FUNCTIONAL", depth=2),
        "D3_TECHNICAL_REALIZATION": CapstoneNode(label="D3_TECHNICAL_REALIZATION", schema_class="D3_TECHNICAL_REALIZATION", depth=2),
        "D4_EXECUTION_PLANNING": CapstoneNode(label="D4_EXECUTION_PLANNING", schema_class="D4_EXECUTION_PLANNING", depth=2)
    }
    
    # Map old categories/domains to new ones
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
    
    # Traverse old domain children
    for domain in root.children:
        target_domain_name = mapping.get(domain.schema_class)
        if not target_domain_name:
            continue
        
        target_domain = domain_nodes[target_domain_name]
        
        # Collect all AtomicReq / IntentMatching nodes under this old domain
        atomic_reqs = []
        def collect_atomic_reqs(node):
            if node.depth == 4:
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
            
            # T4 terminology nodes
            leaves = []
            def collect_leaves(node):
                if node.depth == 6 or node.schema_class in ("ConceptKeyword", "TechKeyword", "TerminologyVerification"):
                    if node.children == []: # it is a leaf node
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
                
    # Always append all 4 domains
    for d_name in ["D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"]:
        new_root.children.append(domain_nodes[d_name])
        
    return new_root

from sentence_transformers import SentenceTransformer

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
        
    print(f"  Encoding {len(all_t3_nodes)} T3 Intent nodes...")
    if all_t3_nodes:
        t3_texts = [n.normalized_text if n.normalized_text else (n.raw_text if n.raw_text else "None") for n in all_t3_nodes]
        t3_embs = model.encode(t3_texts, batch_size=256, show_progress_bar=False)
        for n, emb in zip(all_t3_nodes, t3_embs):
            n.embedding = emb.tolist()
        
    print(f"  Encoding {len(roots)} Root nodes...")
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

# Dynamic Tree Bypasses for Ablation Variants
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
    # Collect all T3 nodes under any T2 Domain child
    for t2 in root.children:
        for t3 in t2.children:
            t3_clone = copy.deepcopy(t3)
            # Lift T3 to depth 2
            t3_clone.depth = 2
            # Set all T4 children to depth 3
            for t4 in t3_clone.children:
                t4.depth = 3
            t3_nodes.append(t3_clone)
    new_root.children = t3_nodes
    return new_root

# Custom Cost Model for Ablations
class CustomSWCostModel(sw_bted.SWCostModel):
    def __init__(self, variant: str, cso_graph=None, max_depth=19):
        super().__init__(cso_graph=cso_graph, max_depth=max_depth)
        self.variant = variant

    def w_del(self, u: CapstoneNode) -> float:
        if u.depth == 1:
            return 0.0
        if self.variant == "no_T2":
            # Intent matching is now depth 2
            if u.schema_class == "IntentMatching":
                return 1.0
            # Terminology is now depth 3
            if u.schema_class == "TerminologyVerification":
                weight = getattr(u, 'tfidf_weight', None)
                if weight is None:
                    weight = 0.5
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
                if weight is None:
                    weight = 0.5
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
        elif u.depth == 2 and self.variant != "no_T2":  # Domain
            return 0.0 if u.label == v.label else 1.0
        return 1.0

    def dist_schema(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.schema_class != v.schema_class:
            return 1.0
        if u.depth == 2 and self.variant != "no_T2":  # Domain
            return sw_bted.DOMAIN_SCHEMA_DIST.get((u.label, v.label), 1.0)
        return 0.0 if u.schema_class == v.schema_class else 1.0

    def w_rep(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.schema_class != v.schema_class:
            return self.w_del(u) + self.w_ins(v)
        if u.depth == 1:
            return 0.0
        
        # Configure beta dynamically based on schema_class
        if u.schema_class == "IntentMatching":
            beta_l = 0.9
        elif u.schema_class == "TerminologyVerification":
            beta_l = 1.0
        else:  # Domain Partition
            beta_l = 0.0
            
        content_d = self.dist_content(u, v)
        schema_d = self.dist_schema(u, v)
        return (self.w_del(u) + self.w_ins(v)) * (beta_l * content_d + (1.0 - beta_l) * schema_d)

def _eval_single_pair(args):
    doc_a, doc_b, variant, alpha = args
    global _worker_cso_graph, _worker_trees
    
    cost_model = CustomSWCostModel(variant=variant, cso_graph=_worker_cso_graph)
    cost_model.alpha = alpha
    
    tree_a = _worker_trees[doc_a]
    tree_b = _worker_trees[doc_b]
    
    # Apply structural modifications in-worker based on variant
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
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
        return 0.0
        
    return sw_bted.normalize_similarity(tree_a, tree_b, cost_model)

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
    print("SW-BTED 4-LAYER EVALUATION AND ABLATION ENGINE")
    print("="*60)
    
    # ── Load Datasets ──
    print("\n[1] Loading datasets...")
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw_6l = json.load(open("data/dataset/trees_t6_unnormalized.json", encoding="utf-8"))
        
    print("Converting FPT 6-layer trees to 4-layer structures...")
    fpt_trees_raw = {}
    for code, tree_dict in fpt_trees_raw_6l.items():
        node_6l = CapstoneNode.from_dict(tree_dict)
        node_4l = convert_6l_to_4l(node_6l)
        fpt_trees_raw[code] = node_4l.to_dict()
    
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw_6l = json.load(open("datasets/pure_adapted/pure_trees.json", encoding="utf-8"))
    
    # Convert PURE 6-layer trees to 4-layer representation
    print("Converting PURE 6-layer trees to 4-layer structures...")
    pure_trees_raw = {}
    for code, tree_dict in pure_trees_raw_6l.items():
        node_6l = CapstoneNode.from_dict(tree_dict)
        node_4l = convert_6l_to_4l(node_6l)
        pure_trees_raw[code] = node_4l.to_dict()
        
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    max_depth = cso_data.get("max_depth", 19)
    
    # ── Populate SBERT Embeddings ──
    print("Loading SentenceTransformer model all-MiniLM-L6-v2...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Populating SBERT embeddings for FPT...")
    fpt_trees_raw = populate_embeddings(fpt_trees_raw, sbert_model)
    print("Populating SBERT embeddings for PURE...")
    pure_trees_raw = populate_embeddings(pure_trees_raw, sbert_model)
    
    VARIANTS = {
        "SW_BTED_4L": ("proposed", 0.6),
        "SW_BTED_4L_no_T4": ("no_T4", 0.6),
        "SW_BTED_4L_no_T2": ("no_T2", 0.6),
        "SW_BTED_4L_alpha_0": ("proposed", 0.0)
    }
    
    num_workers = min(os.cpu_count(), 8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    eval_results = {"FPT": {}, "PURE": {}}
    
    datasets_info = [
        ("FPT", fpt_pairs, fpt_trees_raw),
        ("PURE", pure_pairs, pure_trees_raw)
    ]
    
    for ds_name, ds_pairs, ds_trees in datasets_info:
        print("\n" + "="*50)
        print(f"EVALUATING DATASET: {ds_name}")
        print("="*50)
        
        labels = ds_pairs["label"].to_numpy()
        strat_labels = ds_pairs["type"].to_numpy() if "type" in ds_pairs else labels
        
        for var_name, (variant_type, alpha) in VARIANTS.items():
            print(f"\n>>> Running Variant {var_name}...")
            start_time = time.time()
            
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=("data/processed/cso_graph.pkl", max_depth, ds_trees)
            )
            
            args_list = [(row.doc_a, row.doc_b, variant_type, alpha) for _, row in ds_pairs.iterrows()]
            similarities = np.array(list(pool.map(_eval_single_pair, args_list)))
            pool.shutdown()
            
            fold_f1s = []
            fold_precisions = []
            fold_recalls = []
            fold_aucs = []
            cv_preds = np.zeros(len(ds_pairs), dtype=int)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                best_thresh = find_best_threshold(similarities[train_idx], labels[train_idx])
                
                test_sims = similarities[test_idx]
                test_labels = labels[test_idx]
                preds = np.array([1 if s >= best_thresh else 0 for s in test_sims])
                cv_preds[test_idx] = preds
                
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
            
            duration = time.time() - start_time
            print(f"  -> Done in {duration:.2f}s. F1-score: {mean_f1:.4f} (±{std_f1:.4f})")
            
            eval_results[ds_name][var_name] = {
                "similarities": similarities,
                "preds": cv_preds,
                "f1": f"{mean_f1:.4f} ± {std_f1:.4f}",
                "precision": f"{mean_p:.4f} ± {std_p:.4f}",
                "recall": f"{mean_r:.4f} ± {std_r:.4f}",
                "roc_auc": f"{mean_auc:.4f} ± {std_auc:.4f}",
                "f1_val": mean_f1
            }
            
    # ── Write Output CSV Files in results/4layer/ ──
    os.makedirs("results/4layer", exist_ok=True)
    
    # 1. Main Results FPT
    main_fpt_df = pd.DataFrame([
        {"Method": "Proposed SW-BTED 4L", "Precision": eval_results["FPT"]["SW_BTED_4L"]["precision"], "Recall": eval_results["FPT"]["SW_BTED_4L"]["recall"], "F1-Score": eval_results["FPT"]["SW_BTED_4L"]["f1"], "ROC-AUC": eval_results["FPT"]["SW_BTED_4L"]["roc_auc"]}
    ])
    main_fpt_df.to_csv("results/4layer/main_results_FPT.csv", index=False)
    
    # 2. Main Results PURE
    main_pure_df = pd.DataFrame([
        {"Method": "Proposed SW-BTED 4L", "Precision": eval_results["PURE"]["SW_BTED_4L"]["precision"], "Recall": eval_results["PURE"]["SW_BTED_4L"]["recall"], "F1-Score": eval_results["PURE"]["SW_BTED_4L"]["f1"], "ROC-AUC": eval_results["PURE"]["SW_BTED_4L"]["roc_auc"]}
    ])
    main_pure_df.to_csv("results/4layer/main_results_PURE.csv", index=False)
    
    # 3. Ablation no_T2 (Topic Conflation test)
    no_t2_fpt = eval_results["FPT"]["SW_BTED_4L_no_T2"]
    ablation_no_t2_df = pd.DataFrame([
        {"Variant": "SW_BTED_4L_no_T2", "Precision": no_t2_fpt["precision"], "Recall": no_t2_fpt["recall"], "F1-Score": no_t2_fpt["f1"], "ROC-AUC": no_t2_fpt["roc_auc"], "Delta-F1-vs-Proposed": f"{no_t2_fpt['f1_val'] - eval_results['FPT']['SW_BTED_4L']['f1_val']:.4f}"}
    ])
    ablation_no_t2_df.to_csv("results/4layer/ablation_no_T2_FPT.csv", index=False)
    
    # 4. Ablation no_T4
    no_t4_fpt = eval_results["FPT"]["SW_BTED_4L_no_T4"]
    ablation_no_t4_df = pd.DataFrame([
        {"Variant": "SW_BTED_4L_no_T4", "Precision": no_t4_fpt["precision"], "Recall": no_t4_fpt["recall"], "F1-Score": no_t4_fpt["f1"], "ROC-AUC": no_t4_fpt["roc_auc"], "Delta-F1-vs-Proposed": f"{no_t4_fpt['f1_val'] - eval_results['FPT']['SW_BTED_4L']['f1_val']:.4f}"}
    ])
    ablation_no_t4_df.to_csv("results/4layer/ablation_no_T4_FPT.csv", index=False)
    
    # 5. Topic Conflation Subset Analysis
    # Let's extract the 5 specific Topic Conflation pairs from FPT pairs (Group 2 pairs)
    tc_pairs = [
        ("SU26SE048", "SU26SE087"),
        ("SP26SE048", "SU26SE087"),
        ("SP26SE119", "SU26SE067"),
        ("SU26SE169", "SP26SE069"),
        ("SU26SE087", "SP26SE001")
    ]
    
    # Find matching indexes in fpt_pairs
    tc_indexes = []
    for idx, row in fpt_pairs.iterrows():
        for p1, p2 in tc_pairs:
            if (row.doc_a == p1 and row.doc_b == p2) or (row.doc_a == p2 and row.doc_b == p1):
                tc_indexes.append(idx)
                break
                
    # Also define the remainder of pairs as NON_TC_TYPE
    non_tc_indexes = [i for i in range(len(fpt_pairs)) if i not in tc_indexes]
    
    y_true_fpt = fpt_pairs["label"].to_numpy()
    preds_4l = eval_results["FPT"]["SW_BTED_4L"]["preds"]
    preds_no_t2 = eval_results["FPT"]["SW_BTED_4L_no_T2"]["preds"]
    preds_alpha_0 = eval_results["FPT"]["SW_BTED_4L_alpha_0"]["preds"]
    
    # Calculate accuracy on TC subset
    acc_4l_tc = np.mean(y_true_fpt[tc_indexes] == preds_4l[tc_indexes])
    acc_no_t2_tc = np.mean(y_true_fpt[tc_indexes] == preds_no_t2[tc_indexes])
    acc_alpha_0_tc = np.mean(y_true_fpt[tc_indexes] == preds_alpha_0[tc_indexes])
    
    # Run McNemar tests on the Topic Conflation subset
    _, p_val_no_t2 = run_mcnemar_test(y_true_fpt[tc_indexes], preds_4l[tc_indexes], preds_no_t2[tc_indexes])
    _, p_val_alpha_0 = run_mcnemar_test(y_true_fpt[tc_indexes], preds_4l[tc_indexes], preds_alpha_0[tc_indexes])
    
    tc_analysis_df = pd.DataFrame([
        {
            "Subset": "Topic Conflation (TC_TYPE)",
            "SW_BTED_4L Accuracy": f"{acc_4l_tc * 100:.2f}%",
            "SW_BTED_no_T2 Accuracy": f"{acc_no_t2_tc * 100:.2f}%",
            "SBERT (α=0) Accuracy": f"{acc_alpha_0_tc * 100:.2f}%",
            "McNemar p (4L vs no_T2)": f"{p_val_no_t2:.4f}",
            "McNemar p (4L vs SBERT)": f"{p_val_alpha_0:.4f}"
        }
    ])
    tc_analysis_df.to_csv("results/4layer/topic_conflation_analysis.csv", index=False)
    
    # 6. Comparison of 6L vs 4L (6L results taken from historical paper baseline F1 values)
    # 6L results FPT: F1 = 0.9822, PURE: F1 = 0.8143
    comp_df = pd.DataFrame([
        {"Metric": "F1-Score (FPT)", "SW-BTED 6L (Old)": "0.9822", "SW-BTED 4L (New)": f"{eval_results['FPT']['SW_BTED_4L']['f1_val']:.4f}", "Delta": f"{eval_results['FPT']['SW_BTED_4L']['f1_val'] - 0.9822:.4f}"},
        {"Metric": "F1-Score (PURE)", "SW-BTED 6L (Old)": "0.8143", "SW-BTED 4L (New)": f"{eval_results['PURE']['SW_BTED_4L']['f1_val']:.4f}", "Delta": f"{eval_results['PURE']['SW_BTED_4L']['f1_val'] - 0.8143:.4f}"}
    ])
    comp_df.to_csv("results/4layer/comparison_6L_vs_4L.csv", index=False)
    
    print("\n" + "="*50)
    print("PHASE 2 RE-EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
