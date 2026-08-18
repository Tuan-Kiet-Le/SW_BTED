import os
import sys
import json
import pickle
import time
import importlib
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

# ── Dynamic Imports of SW-BTED ──
sw_bted_module = importlib.import_module("src.05_sw_bted")
SWCostModel = sw_bted_module.SWCostModel
normalize_similarity = sw_bted_module.normalize_similarity
dict_to_node = sw_bted_module.dict_to_node

# ── Multiprocessing Globals and Workers ──
_worker_cso_graph = None
_worker_cost_model = None
_worker_trees = {}

def _init_worker(cso_graph_path, max_depth_val, trees_dict_raw):
    global _worker_cso_graph, _worker_cost_model, _worker_trees
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

# ── Heuristic Logic for Adaptive T5 ──
def should_activate_t5(sentence: str) -> bool:
    """
    Quyết định có chạy Semantic Role Extraction (T5) không.
    Trả về True nếu câu đủ phức tạp để T5 có ích.
    """
    tokens = sentence.split()
    token_count = len(tokens)

    # Đếm dấu hiệu cú pháp phức tạp
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

    # Điều kiện kích hoạt T5:
    # (1) Câu đủ dài (> 15 tokens) VÀ
    # (2) Có ít nhất 2 dấu hiệu cú pháp phức tạp (đã cập nhật theo quyết định người dùng)
    return token_count > 15 and clause_indicators >= 2

# ── In-Memory Transformations ──
def transform_to_5l_norole(root: CapstoneNode) -> CapstoneNode:
    # Bypass T5 (Semantic Role)
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

def transform_to_adaptive_t5(root: CapstoneNode, doc_id: str, rates: dict) -> CapstoneNode:
    total_atomic = 0
    t5_activated = 0
    t5_skipped = 0
    
    def traverse(node):
        nonlocal total_atomic, t5_activated, t5_skipped
        if node.depth == 4:
            total_atomic += 1
            text = node.normalized_text if node.normalized_text else (node.raw_text if node.raw_text else "")
            if should_activate_t5(text):
                t5_activated += 1
            else:
                t5_skipped += 1
                # Bypass T5: lift T6 leaves to be children of T4
                leaves = []
                for t5 in node.children:
                    leaves.extend(t5.children)
                node.children = leaves
        else:
            for child in node.children:
                traverse(child)
                
    traverse(root)
    rates[doc_id] = {
        "total_atomic_reqs": total_atomic,
        "t5_activated_count": t5_activated,
        "t5_skipped_count": t5_skipped,
        "t5_activation_rate": round(t5_activated / total_atomic, 4) if total_atomic > 0 else 0.0
    }
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

def calculate_type_metrics(preds, df, labels):
    # Calculate Type A TPR
    idx_a = df[df["type"] == "Type_A"].index if "type" in df else []
    tpr_a = np.sum(preds[idx_a] == 1) / len(idx_a) if len(idx_a) > 0 else 1.0
    
    # Calculate Type B TNR
    idx_b = df[df["type"] == "Type_B"].index if "type" in df else []
    tnr_b = np.sum(preds[idx_b] == 0) / len(idx_b) if len(idx_b) > 0 else 1.0
    
    # Calculate Type C TNR
    idx_c = df[df["type"] == "Type_C"].index if "type" in df else []
    tnr_c = np.sum(preds[idx_c] == 0) / len(idx_c) if len(idx_c) > 0 else 1.0
    
    return tpr_a, tnr_b, tnr_c

def main():
    print("="*60)
    print("SW-BTED TASK 1: ADAPTIVE T5 ACTIVATION EXPERIMENT")
    print("="*60)
    
    # ── Load Datasets ──
    print("\n[1] Loading datasets...")
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw = json.load(open("datasets/pure_adapted/pure_trees.json", encoding="utf-8"))
    
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    
    # Create diagnostics and results directories
    os.makedirs("diagnostics", exist_ok=True)
    os.makedirs("results/adaptive_t5", exist_ok=True)
    
    # ── Calculate and Log Activation Rates ──
    print("\n[2] Computing Adaptive T5 Activation Rates...")
    activation_log = {"FPT": {}, "PURE": {}}
    
    fpt_rates = {}
    pure_rates = {}
    
    # Transform trees and collect rates
    for doc_id, tree_dict in fpt_trees_raw.items():
        root = CapstoneNode.from_dict(tree_dict)
        transform_to_adaptive_t5(root, doc_id, fpt_rates)
        
    for doc_id, tree_dict in pure_trees_raw.items():
        root = CapstoneNode.from_dict(tree_dict)
        transform_to_adaptive_t5(root, doc_id, pure_rates)
        
    fpt_avg_rate = np.mean([info["t5_activation_rate"] for info in fpt_rates.values()])
    pure_avg_rate = np.mean([info["t5_activation_rate"] for info in pure_rates.values()])
    
    activation_log["FPT"] = {
        "summary": {
            "total_documents": len(fpt_rates),
            "avg_activation_rate": round(float(fpt_avg_rate), 4)
        },
        "documents": fpt_rates
    }
    activation_log["PURE"] = {
        "summary": {
            "total_documents": len(pure_rates),
            "avg_activation_rate": round(float(pure_avg_rate), 4)
        },
        "documents": pure_rates
    }
    
    # Save logs
    with open("diagnostics/t5_activation_rates.json", "w", encoding="utf-8") as f:
        json.dump(activation_log, f, ensure_ascii=False, indent=2)
    with open("results/adaptive_t5/t5_activation_rates.json", "w", encoding="utf-8") as f:
        json.dump(activation_log, f, ensure_ascii=False, indent=2)
        
    print(f"FPT Average T5 Activation Rate: {fpt_avg_rate:.4f} (Expected >= 60% but bypassed by user)")
    print(f"PURE Average T5 Activation Rate: {pure_avg_rate:.4f} (Expected <= 50%)")
    
    # ── Check Activation Rate Conditions ──
    # Note: FPT constraint bypassed by user choice, printing warning only
    if fpt_avg_rate < 0.60:
        print(f"WARNING: FPT Activation rate {fpt_avg_rate:.4f} is less than 60%. (Bypassed by user selection).")
    if pure_avg_rate > 0.50:
        print(f"CRITICAL WARNING: PURE Activation rate {pure_avg_rate:.4f} is greater than 50%. STOPPING execution.")
        sys.exit(1)
        
    print("Activation rate conditions met. Proceeding to evaluation...")
    
    # ── Run Evaluation Folds ──
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    num_workers = min(os.cpu_count(), 8)
    
    datasets_info = [
        ("FPT", fpt_pairs, fpt_trees_raw),
        ("PURE", pure_pairs, pure_trees_raw)
    ]
    
    variants = [
        ("A1", "T5 always ON"),
        ("A2", "T5 always OFF"),
        ("A_new", "T5 ADAPTIVE")
    ]
    
    # Store test predictions and metrics for comparisons
    evaluation_results = {
        "FPT": {},
        "PURE": {}
    }
    
    for ds_name, ds_pairs, ds_trees in datasets_info:
        print("\n" + "="*50)
        print(f"RUNNING CV EVALUATION FOR: {ds_name}")
        print("="*50)
        
        labels = ds_pairs["label"].to_numpy()
        strat_labels = ds_pairs["type"].to_numpy() if "type" in ds_pairs else labels
        
        for var_id, var_desc in variants:
            print(f"\n>>> Evaluating {var_id}: {var_desc}...")
            
            # 1. Transform trees in memory
            variant_trees = {}
            rates_temp = {}
            for k, v in ds_trees.items():
                root = CapstoneNode.from_dict(v)
                if var_id == "A1":
                    variant_trees[k] = root
                elif var_id == "A2":
                    variant_trees[k] = transform_to_5l_norole(root)
                elif var_id == "A_new":
                    variant_trees[k] = transform_to_adaptive_t5(root, k, rates_temp)
            
            # 2. Parallel Similarity computation
            variant_trees_dict = {k: v.to_dict() for k, v in variant_trees.items()}
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=("data/processed/cso_graph.pkl", max_depth, variant_trees_dict)
            )
            
            # proposed hyperparams (alpha=0.6, default per-layer betas)
            alpha = 0.6
            beta_dict = {"T2": 0.0, "T3": 0.6, "T4": 0.9, "T5": 0.0, "T6": 0.8}
            args_list = [(row.doc_a, row.doc_b, alpha, beta_dict) for _, row in ds_pairs.iterrows()]
            similarities = np.array(list(pool.map(_eval_single_pair, args_list)))
            pool.shutdown()
            
            # 3. 5-Fold Cross-Validation
            fold_f1s = []
            fold_precisions = []
            fold_recalls = []
            fold_aucs = []
            cv_preds = np.zeros(len(ds_pairs), dtype=int)
            
            fold_tpr_as = []
            fold_tnr_bs = []
            fold_tnr_cs = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                inner_train, inner_val = next(inner_skf.split(ds_pairs.iloc[train_idx], strat_labels[train_idx]))
                val_idx = train_idx[inner_val]
                
                best_thresh = find_best_threshold(similarities[val_idx], labels[val_idx])
                
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
                
                # Type specific metrics (only for test partition of this fold)
                test_df = ds_pairs.iloc[test_idx].reset_index(drop=True)
                test_labels_reset = test_labels
                tpr_a, tnr_b, tnr_c = calculate_type_metrics(preds, test_df, test_labels_reset)
                
                fold_tpr_as.append(tpr_a)
                fold_tnr_bs.append(tnr_b)
                fold_tnr_cs.append(tnr_c)
                
            mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
            mean_p, std_p = np.mean(fold_precisions), np.std(fold_precisions)
            mean_r, std_r = np.mean(fold_recalls), np.std(fold_recalls)
            mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
            
            mean_tpr_a = np.mean(fold_tpr_as)
            mean_tnr_b = np.mean(fold_tnr_bs)
            mean_tnr_c = np.mean(fold_tnr_cs)
            
            print(f"  F1: {mean_f1:.4f} (±{std_f1:.4f})")
            
            evaluation_results[ds_name][var_id] = {
                "preds": cv_preds.tolist(),
                "f1s": fold_f1s,
                "metrics": {
                    "f1": mean_f1,
                    "f1_std": std_f1,
                    "precision": mean_p,
                    "precision_std": std_p,
                    "recall": mean_r,
                    "recall_std": std_r,
                    "roc_auc": mean_auc,
                    "roc_auc_std": std_auc,
                    "type_a_tpr": mean_tpr_a,
                    "type_b_tnr": mean_tnr_b,
                    "type_c_tnr": mean_tnr_c
                }
            }
            
            # Save results to json for detailed audit
            with open(f"results/adaptive_t5/adaptive_t5_{ds_name}_results.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_results[ds_name], f, ensure_ascii=False, indent=2)

    # ── Statistical Testing and Bounded Check ──
    print("\n[3] Running McNemar Significance Tests (A_new vs A1)...")
    comparison_rows = []
    significance_results = {}
    
    for ds_name in ["FPT", "PURE"]:
        ds_pairs = fpt_pairs if ds_name == "FPT" else pure_pairs
        y_true = ds_pairs["label"].to_numpy()
        
        preds_a1 = np.array(evaluation_results[ds_name]["A1"]["preds"])
        preds_anew = np.array(evaluation_results[ds_name]["A_new"]["preds"])
        
        chi2, p_val = run_mcnemar_test(y_true, preds_anew, preds_a1)
        is_significant = bool(p_val < 0.01)
        
        significance_results[ds_name] = {
            "chi2": float(chi2),
            "p_value": float(p_val),
            "significant_bonferroni": is_significant
        }
        
        print(f"{ds_name} McNemar Test: chi2={chi2:.4f}, p-value={p_val:.4e} (Significant: {is_significant})")
        
    with open("results/adaptive_t5/mcnemar_adaptive_vs_A1.json", "w", encoding="utf-8") as f:
        json.dump(significance_results, f, ensure_ascii=False, indent=2)

    # ── Save Master Comparison CSV ──
    print("\n[4] Generating master comparison table CSV...")
    
    rows = []
    for ds_name in ["FPT", "PURE"]:
        for var_id, var_desc in variants:
            m = evaluation_results[ds_name][var_id]["metrics"]
            rows.append({
                "Dataset": ds_name,
                "Variant_ID": var_id,
                "Variant_Name": var_desc,
                "F1_Score": f"{m['f1']:.4f} ± {m['f1_std']:.4f}",
                "Precision": f"{m['precision']:.4f} ± {m['precision_std']:.4f}",
                "Recall": f"{m['recall']:.4f} ± {m['recall_std']:.4f}",
                "ROC_AUC": f"{m['roc_auc']:.4f} ± {m['roc_auc_std']:.4f}",
                "Type_A_TPR": f"{m['type_a_tpr']:.4f}",
                "Type_B_TNR": f"{m['type_b_tnr']:.4f}",
                "Type_C_TNR": f"{m['type_c_tnr']:.4f}",
            })
            
    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv("results/adaptive_t5/adaptive_vs_variants_table.csv", index=False)
    print("Saved comparison table to results/adaptive_t5/adaptive_vs_variants_table.csv")
    
    # Save McNemar CSV
    mcnemar_rows = []
    for ds_name in ["FPT", "PURE"]:
        sig = significance_results[ds_name]
        mcnemar_rows.append({
            "Dataset": ds_name,
            "Comparison": "A_new vs A1",
            "Chi2": f"{sig['chi2']:.4f}",
            "p_value": f"{sig['p_value']:.4e}",
            "Significant_Alpha_0.01": "Yes" if sig["significant_bonferroni"] else "No"
        })
    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv("results/adaptive_t5/mcnemar_adaptive_vs_A1.csv", index=False)
    print("Saved McNemar results to results/adaptive_t5/mcnemar_adaptive_vs_A1.csv")

    print("\n" + "="*50)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
