import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from scipy.stats import binom

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')

# Custom node type import
from src.node import CapstoneNode

# ── Baseline B6: BM25 Similarity ──
def compute_bm25_similarity(doc_A: str, doc_B: str) -> float:
    tokens_A = doc_A.lower().split()
    tokens_B = doc_B.lower().split()
    
    if not tokens_A or not tokens_B:
        return 0.0

    # Direction 1: A queries B
    bm25_B = BM25Okapi([tokens_B])
    score_AtoB = bm25_B.get_scores(tokens_A)[0]

    # Direction 2: B queries A
    bm25_A = BM25Okapi([tokens_A])
    score_BtoA = bm25_A.get_scores(tokens_B)[0]

    # Symmetric score (normalized)
    raw_score = (score_AtoB + score_BtoA) / 2

    # Normalize to [0, 1] using sigmoid
    sim = 1 / (1 + np.exp(-raw_score * 0.1))
    return float(sim)

# Helper to find best threshold
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

# McNemar significance test
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

# Helper to get full texts from FPT json
def get_fpt_full_text(doc_code: str, full_texts: dict, tree_raw: dict) -> str:
    if full_texts and doc_code in full_texts:
        sections = full_texts[doc_code]
        title = tree_raw.get("label", "") if tree_raw else ""
        text = title + " " + " ".join([t for t in sections.values() if t])
        if text.strip():
            return text
    # Fallback to tree label
    return tree_raw.get("label", "")

def main():
    print("="*70)
    print("SW-BTED TASK 3: NEW BASELINES EVALUATION (BM25 & SimCSE)")
    print("="*70)
    
    # ── Load Datasets ──
    print("\n[1] Loading datasets...")
    
    # FPT
    fpt_pairs = pd.read_csv("data/dataset/pairs.csv")
    fpt_trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    fpt_full_texts = json.load(open("data/dataset/full_texts.json", encoding="utf-8"))
    
    # PURE
    pure_pairs = pd.read_csv("datasets/pure_adapted/document_pairs.csv")
    pure_trees_raw = json.load(open("datasets/pure_adapted/pure_trees.json", encoding="utf-8"))
    pure_pseudo_docs = json.load(open("datasets/pure_adapted/pseudo_documents.json", encoding="utf-8"))
    
    # Extract doc text maps
    print("Extracting document full texts...")
    fpt_docs = {k: get_fpt_full_text(k, fpt_full_texts, v) for k, v in fpt_trees_raw.items()}
    pure_docs = {}
    for doc_id, doc_data in pure_pseudo_docs.items():
        requirements = doc_data.get("requirements", [])
        pure_docs[doc_id] = " ".join(requirements)
        
    # ── Initialize SimCSE Model ──
    print("\n[2] Loading SimCSE model (princeton-nlp/sup-simcse-bert-base-uncased)...")
    simcse_model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
    
    # ── Load SW-BTED (A_new) Predictions from Task 1 ──
    print("\n[3] Loading SW-BTED (T5 Adaptive) predictions...")
    with open("results/adaptive_t5/adaptive_t5_FPT_results.json", "r", encoding="utf-8") as f:
        fpt_sw_preds = json.load(f)["A_new"]["preds"]
    with open("results/adaptive_t5/adaptive_t5_PURE_results.json", "r", encoding="utf-8") as f:
        pure_sw_preds = json.load(f)["A_new"]["preds"]
        
    datasets = [
        ("FPT", fpt_pairs, fpt_docs, fpt_sw_preds),
        ("PURE", pure_pairs, pure_docs, pure_sw_preds)
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs("results/updated_baselines", exist_ok=True)
    
    results = []
    mcnemar_results = []
    
    for ds_name, ds_pairs, ds_docs, sw_preds in datasets:
        print("\n" + "="*60)
        print(f"EVALUATING ON DATASET: {ds_name}")
        print("="*60)
        
        labels = ds_pairs["label"].to_numpy()
        strat_labels = ds_pairs["type"].to_numpy() if "type" in ds_pairs else labels
        
        # 1. Compute BM25 similarities
        print("Computing BM25 similarities...")
        bm25_sims = []
        for _, row in ds_pairs.iterrows():
            sim = compute_bm25_similarity(ds_docs[row.doc_a], ds_docs[row.doc_b])
            bm25_sims.append(sim)
        bm25_sims = np.array(bm25_sims)
        
        # 2. Compute SimCSE similarities
        print("Computing SimCSE similarities...")
        keys = list(ds_docs.keys())
        texts = [ds_docs[k] for k in keys]
        embeddings = simcse_model.encode(texts, show_progress_bar=True)
        key_to_idx = {k: i for i, k in enumerate(keys)}
        
        simcse_sims = []
        for _, row in ds_pairs.iterrows():
            idx_a = key_to_idx[row.doc_a]
            idx_b = key_to_idx[row.doc_b]
            emb_a = embeddings[idx_a].reshape(1, -1)
            emb_b = embeddings[idx_b].reshape(1, -1)
            sim = cosine_similarity(emb_a, emb_b)[0][0]
            simcse_sims.append(sim)
        simcse_sims = np.array(simcse_sims)
        
        # Evaluate each baseline
        baselines = [
            ("B6_BM25", bm25_sims),
            ("B7_SimCSE", simcse_sims)
        ]
        
        for name, sims in baselines:
            print(f"\n>>> Running Stratified 5-Fold CV for baseline: {name}...")
            
            fold_f1s = []
            fold_precisions = []
            fold_recalls = []
            fold_aucs = []
            cv_preds = np.zeros(len(ds_pairs), dtype=int)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(ds_pairs, strat_labels)):
                inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                inner_train, inner_val = next(inner_skf.split(ds_pairs.iloc[train_idx], strat_labels[train_idx]))
                val_idx = train_idx[inner_val]
                
                best_thresh = find_best_threshold(sims[val_idx], labels[val_idx])
                
                test_sims = sims[test_idx]
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
            
            print(f"  Result F1: {mean_f1:.4f} (±{std_f1:.4f}) | Precision: {mean_p:.4f} | Recall: {mean_r:.4f} | ROC-AUC: {mean_auc:.4f}")
            
            # McNemar Test against SW-BTED
            chi2, p_val = run_mcnemar_test(labels, sw_preds, cv_preds)
            sig_bonferroni = "Yes" if p_val < 0.01 else "No"
            
            results.append({
                "Dataset": ds_name,
                "Method": name,
                "F1_Score": f"{mean_f1:.4f} ± {std_f1:.4f}",
                "Precision": f"{mean_p:.4f} ± {std_p:.4f}",
                "Recall": f"{mean_r:.4f} ± {std_r:.4f}",
                "ROC_AUC": f"{mean_auc:.4f} ± {std_auc:.4f}",
                "McNemar_p_vs_SW": f"{p_val:.4e}",
                "Significant_Bonferroni": sig_bonferroni
            })
            
            # Save predictions to pair similarities
            ds_pairs[f"sim_{name.lower()}"] = sims
            ds_pairs[f"pred_{name.lower()}"] = cv_preds
            
        # Write dataset pairs back
        ds_pairs.to_csv(f"results/updated_baselines/{ds_name.lower()}_pair_similarities.csv", index=False)
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/updated_baselines/new_baselines_results.csv", index=False)
    
    # ── Update master comparison table ──
    print("\n[4] Building full comparison tables...")
    # Load leak_free results for FPT
    df_leak_free = pd.read_csv("results/results_leak_free.csv")
    
    # We want to add B6 and B7 rows to it.
    # We need to map new results to the same column structure.
    # Columns in results_leak_free: method,threshold_val,precision_mean,precision_std,recall_mean,recall_std,f1_mean,f1_std,roc_auc_mean,roc_auc_std,roc_auc_filtered_mean,roc_auc_filtered_std,typeA_tpr_mean,typeA_tpr_std,typeB_tnr_mean,typeB_tnr_std,typeC_tnr_mean,typeC_tnr_std
    
    new_rows = []
    for idx, row in df_results[df_results["Dataset"] == "FPT"].iterrows():
        method_name = "B6: BM25" if row["Method"] == "B6_BM25" else "B7: SimCSE"
        f1_mean, f1_std = map(float, row["F1_Score"].split(" ± "))
        p_mean, p_std = map(float, row["Precision"].split(" ± "))
        r_mean, r_std = map(float, row["Recall"].split(" ± "))
        auc_mean, auc_std = map(float, row["ROC_AUC"].split(" ± "))
        
        # Fill rest of metrics from cross-validation calculations
        cv_preds = fpt_pairs[f"pred_{row['Method'].lower()}"]
        labels = fpt_pairs["label"].to_numpy()
        
        # Calculate TPR/TNR for FPT
        idx_a = fpt_pairs[fpt_pairs["type"] == "Type_A"].index
        tpr_a = np.sum(cv_preds[idx_a] == 1) / len(idx_a)
        
        idx_b = fpt_pairs[fpt_pairs["type"] == "Type_B"].index
        tnr_b = np.sum(cv_preds[idx_b] == 0) / len(idx_b)
        
        idx_c = fpt_pairs[fpt_pairs["type"] == "Type_C"].index
        tnr_c = np.sum(cv_preds[idx_c] == 0) / len(idx_c)
        
        new_rows.append({
            "method": method_name,
            "threshold_val": 0.5, # dummy, threshold is CV-specific
            "precision_mean": p_mean,
            "precision_std": p_std,
            "recall_mean": r_mean,
            "recall_std": r_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "roc_auc_mean": auc_mean,
            "roc_auc_std": auc_std,
            "roc_auc_filtered_mean": auc_mean, # fallback
            "roc_auc_filtered_std": auc_std,
            "typeA_tpr_mean": tpr_a,
            "typeA_tpr_std": 0.0,
            "typeB_tnr_mean": tnr_b,
            "typeB_tnr_std": 0.0,
            "typeC_tnr_mean": tnr_c,
            "typeC_tnr_std": 0.0
        })
        
    df_new_rows = pd.DataFrame(new_rows)
    df_full_comparison = pd.concat([df_leak_free, df_new_rows], ignore_index=True)
    df_full_comparison.to_csv("results/updated_baselines/full_comparison_table.csv", index=False)
    
    # Save a clean table version as well
    print("\nUpdated Master Table (FPT):")
    print(df_full_comparison[["method", "f1_mean", "precision_mean", "recall_mean", "roc_auc_mean"]].to_string(index=False))
    
    # Also write a summary report file
    report_md = f"""# Task 3: New Baselines Evaluation Report

This report summarizes the performance of two new baselines, B6 (BM25) and B7 (SimCSE), evaluated against the proposed SW-BTED model on both FPT Capstone and PURE datasets.

## FPT Dataset Results

| Method | F1-Score | Precision | Recall | ROC-AUC | McNemar p vs SW-BTED | Significant (Bonferroni) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SW-BTED (Adaptive T5)** | 0.9765 | 0.9556 | 1.0000 | 1.0000 | — | — |
| **B1: Cosine TF-IDF** | 0.9939 | 0.9882 | 1.0000 | 1.0000 | 0.2188 | No |
| **B2: Cosine SBERT** | 0.9593 | 0.9240 | 1.0000 | 1.0000 | 0.7539 | No |
| **B6: BM25** | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['F1_Score'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['Precision'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['Recall'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['ROC_AUC'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['McNemar_p_vs_SW'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B6_BM25')]['Significant_Bonferroni'].values[0]} |
| **B7: SimCSE** | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['F1_Score'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['Precision'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['Recall'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['ROC_AUC'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['McNemar_p_vs_SW'].values[0]} | {df_results[(df_results['Dataset'] == 'FPT') & (df_results['Method'] == 'B7_SimCSE')]['Significant_Bonferroni'].values[0]} |

## PURE Dataset Results

| Method | F1-Score | Precision | Recall | ROC-AUC | McNemar p vs SW-BTED | Significant (Bonferroni) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SW-BTED (Adaptive T5)** | 0.8176 | 0.8610 | 0.7834 | 0.9390 | — | — |
| **B6: BM25** | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['F1_Score'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['Precision'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['Recall'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['ROC_AUC'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['McNemar_p_vs_SW'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B6_BM25')]['Significant_Bonferroni'].values[0]} |
| **B7: SimCSE** | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['F1_Score'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['Precision'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['Recall'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['ROC_AUC'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['McNemar_p_vs_SW'].values[0]} | {df_results[(df_results['Dataset'] == 'PURE') & (df_results['Method'] == 'B7_SimCSE')]['Significant_Bonferroni'].values[0]} |

"""
    with open("results/updated_baselines/new_baselines_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("New baselines report saved successfully to results/updated_baselines/new_baselines_report.md")

if __name__ == "__main__":
    main()
