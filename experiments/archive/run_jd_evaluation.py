import os
import sys
import json
import pickle
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.node import CapstoneNode
import importlib
sw_bted = importlib.import_module("src.05_sw_bted")

def compute_lexical_jaccard(text1: str, text2: str) -> float:
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    if not words1 and not words2:
        return 1.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

def collect_term_leaves(node: CapstoneNode) -> List[str]:
    leaves = []
    def traverse(n):
        if n.depth == 4:
            leaves.append(n.label.lower())
        for child in n.children:
            traverse(child)
    traverse(node)
    return list(set(leaves))

def get_full_document_text(root_node: CapstoneNode) -> str:
    texts = []
    def traverse(node):
        if node.depth == 3:
            val = node.normalized_text if node.normalized_text else node.raw_text
            if val:
                texts.append(val)
        for child in node.children:
            traverse(child)
    traverse(root_node)
    return " ".join(texts)

def main():
    print("==============================================================")
    print("SW-CapTree: Dataset-3 (Job Descriptions) Evaluation Runner")
    print("==============================================================")
    
    pairs_path = "Data/dataset/linkedin_jd/pairs.csv"
    trees_path = "Data/dataset/linkedin_jd/trees.json"
    cso_graph_path = "data/processed/cso_graph.pkl"
    
    if not all(os.path.exists(p) for p in [pairs_path, trees_path, cso_graph_path]):
        print("Error: Missing required evaluation data files.")
        return
        
    # 1. Load Data
    pairs_df = pd.read_csv(pairs_path)
    print(f"Loaded {len(pairs_df)} evaluation pairs.")
    
    with open(trees_path, "r", encoding="utf-8") as f:
        trees_raw = json.load(f)
    trees = {k: CapstoneNode.from_dict(v) for k, v in trees_raw.items()}
    
    with open(cso_graph_path, "rb") as f:
        cso_data = pickle.load(f)
        cso_graph = cso_data["graph"]
        
    # Define cost model
    cost_model = sw_bted.SWCostModel(
        alpha=0.6,
        cso_graph=cso_graph
    )
    
    # 2. Run Predictions
    results = []
    
    # Simple lexical helper regex imports
    import re
    
    for idx, row in pairs_df.iterrows():
        id_A, id_B = str(row["job_id_A"]), str(row["job_id_B"])
        label = row["label"]
        industry_A, industry_B = row["industry_A"], row["industry_B"]
        is_hard_negative = (label == 0 and industry_A != industry_B and row["title_A"] == row["title_B"])
        
        tree_A = trees.get(id_A)
        tree_B = trees.get(id_B)
        
        if not tree_A or not tree_B:
            continue
            
        # SBERT Only similarity
        sbert_sim = sw_bted.cosine_sim(tree_A.embedding, tree_B.embedding)
        
        # TED Only similarity (alpha = 1.0)
        cost_model_ted = sw_bted.SWCostModel(alpha=1.0, cso_graph=cso_graph)
        ted_sim = sw_bted.normalize_similarity(tree_A, tree_B, cost_model_ted)
        
        # SW-CapTree (proposed hybrid)
        sw_captree_sim = sw_bted.normalize_similarity(tree_A, tree_B, cost_model)
        
        # Lexical Jaccard
        text_A = get_full_document_text(tree_A)
        text_B = get_full_document_text(tree_B)
        lexical_sim = compute_lexical_jaccard(text_A, text_B)
        
        # Engelbach (2024) [P1] baseline
        skills_A = set(collect_term_leaves(tree_A))
        skills_B = set(collect_term_leaves(tree_B))
        if not skills_A and not skills_B:
            skill_jaccard = 1.0
        else:
            skill_jaccard = len(skills_A.intersection(skills_B)) / len(skills_A.union(skills_B))
            
        engelbach_sim = 0.4 * sbert_sim + 0.3 * lexical_sim + 0.3 * skill_jaccard
        
        results.append({
            "label": label,
            "is_hard_negative": is_hard_negative,
            "sbert_sim": sbert_sim,
            "ted_sim": ted_sim,
            "sw_captree_sim": sw_captree_sim,
            "lexical_sim": lexical_sim,
            "engelbach_sim": engelbach_sim
        })
        
    res_df = pd.DataFrame(results)
    
    # 3. Compute Metrics for each method
    methods = {
        "Lexical Overlap": "lexical_sim",
        "SBERT Only": "sbert_sim",
        "TED Only": "ted_sim",
        "Engelbach et al. (2024)": "engelbach_sim",
        "SW-CapTree (Proposed)": "sw_captree_sim"
    }
    
    summary_data = []
    
    # Calculate metrics at optimal F1 threshold
    for name, col in methods.items():
        # Find best threshold for F1 on the dataset
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.linspace(0.1, 0.9, 81):
            preds = (res_df[col] >= t).astype(int)
            f1 = f1_score(res_df["label"], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                
        # Compute final metrics at best threshold
        final_preds = (res_df[col] >= best_thresh).astype(int)
        precision = precision_score(res_df["label"], final_preds, zero_division=0)
        recall = recall_score(res_df["label"], final_preds, zero_division=0)
        auc = roc_auc_score(res_df["label"], res_df[col])
        
        # Hard Negatives (Topic Conflation) TNR
        hn_df = res_df[res_df["is_hard_negative"] == True]
        if len(hn_df) > 0:
            hn_preds = (hn_df[col] >= best_thresh).astype(int)
            # TNR = True Negatives / (True Negatives + False Positives)
            tn = (hn_preds == 0).sum()
            tc_tnr = tn / len(hn_df)
        else:
            tc_tnr = 1.0
            
        summary_data.append({
            "Method": name,
            "Threshold": round(best_thresh, 3),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(best_f1, 4),
            "ROC-AUC": round(auc, 4),
            "TC TNR (Hard Negs)": round(tc_tnr, 4)
        })
        
    summary_df = pd.DataFrame(summary_data)
    print("\nEvaluation Results on JD Dataset:")
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs("results/jd_evaluation", exist_ok=True)
    summary_df.to_csv("results/jd_evaluation/summary.csv", index=False)
    print("\nSaved evaluation results to results/jd_evaluation/summary.csv")

if __name__ == "__main__":
    main()
