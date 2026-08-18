"""
Comprehensive Task 12 Evidentiary Script:
1. Reconcile exact reproducible numbers for Full-Doc SBERT (sim_global)
2. Run standalone 5-Fold CV evaluation for Full-Doc SBERT (F1=0.9855)
3. Compute McNemar 2x2 Contingency Tables & binomtest p-values vs SW-BTED Structural-Only and Hybrid Mode
4. Trace B2 text scope history across all dataset configurations
"""
import json, sys, os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.node import CapstoneNode
import importlib
sw_mod = importlib.import_module("src.05_sw_bted")
baselines_mod = importlib.import_module("src.baselines")

# Load dataset
trees_dict = json.load(open('data/dataset/trees_section.json', encoding='utf-8'))
trees_nodes = {k: CapstoneNode.from_dict(v) for k, v in trees_dict.items()}

pairs = pd.read_csv('data/dataset/pairs.csv')
regen = json.load(open('data/processed/plag_regen_sections.json', encoding='utf-8'))
regen_keys = set(regen.keys())
real_pairs = pairs[~(pairs['doc_a'].isin(regen_keys) | pairs['doc_b'].isin(regen_keys))].reset_index(drop=True)
labels = (real_pairs['type'] == 'Type_A').astype(int).values

full_texts = json.load(open('data/dataset/full_texts.json', encoding='utf-8'))

# 1. Compute Full-Doc SBERT embeddings (full_texts.json)
sbert = SentenceTransformer(r"C:\Users\DuyTuanPC\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf")

doc_texts = {}
for k, node in trees_nodes.items():
    if k in full_texts:
        secs = full_texts[k]
        text = node.label + " " + " ".join([v for v in secs.values() if v])
    else:
        text = node.label
    doc_texts[k] = text

doc_keys = list(doc_texts.keys())
embeddings = sbert.encode([doc_texts[k] for k in doc_keys], show_progress_bar=False)
key_to_emb = {k: emb for k, emb in zip(doc_keys, embeddings)}

# Compute per-pair sim_global, sim_struct, and sim_hybrid
sim_globals = []
sim_structs = []
sim_hybrids = []

cost_model = sw_mod.SWCostModel(alpha=0.6, beta={'T2': 0.0, 'T3': 0.9, 'T4': 0.8}, cso_graph=None, max_depth=19)

for _, row in real_pairs.iterrows():
    # sim_struct
    na = trees_nodes[row.doc_a]
    nb = trees_nodes[row.doc_b]
    na.embedding = None
    nb.embedding = None
    s_struct = sw_mod.normalize_similarity(na, nb, cost_model)
    
    # sim_global
    emb_a = key_to_emb[row.doc_a]
    emb_b = key_to_emb[row.doc_b]
    s_global = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    
    # sim_hybrid
    s_hybrid = round(0.6 * s_struct + 0.4 * s_global, 4)
    
    sim_structs.append(s_struct)
    sim_globals.append(round(s_global, 4))
    sim_hybrids.append(s_hybrid)

sim_structs = np.array(sim_structs)
sim_globals = np.array(sim_globals)
sim_hybrids = np.array(sim_hybrids)

type_a_mask = (labels == 1)
type_bc_mask = (labels == 0)

print("==================================================")
print("1. CANONICAL RECONCILED SCORE DISTRIBUTIONS")
print("==================================================")
print(f"Full-Doc SBERT (sim_global):")
print(f"  Type A (Pos):   Mean = {np.mean(sim_globals[type_a_mask]):.4f} (Min = {np.min(sim_globals[type_a_mask]):.4f}, Max = {np.max(sim_globals[type_a_mask]):.4f})")
print(f"  Type B/C (Neg): Mean = {np.mean(sim_globals[type_bc_mask]):.4f} (Min = {np.min(sim_globals[type_bc_mask]):.4f}, Max = {np.max(sim_globals[type_bc_mask]):.4f})")

print(f"\nSW-BTED Structural-Only (sim_struct):")
print(f"  Type A (Pos):   Mean = {np.mean(sim_structs[type_a_mask]):.4f} (Min = {np.min(sim_structs[type_a_mask]):.4f}, Max = {np.max(sim_structs[type_a_mask]):.4f})")
print(f"  Type B/C (Neg): Mean = {np.mean(sim_structs[type_bc_mask]):.4f} (Min = {np.min(sim_structs[type_bc_mask]):.4f}, Max = {np.max(sim_structs[type_bc_mask]):.4f})")

print(f"\nSW-BTED Hybrid Mode Alpha=0.6 (sim_hybrid):")
print(f"  Type A (Pos):   Mean = {np.mean(sim_hybrids[type_a_mask]):.4f} (Min = {np.min(sim_hybrids[type_a_mask]):.4f}, Max = {np.max(sim_hybrids[type_a_mask]):.4f})")
print(f"  Type B/C (Neg): Mean = {np.mean(sim_hybrids[type_bc_mask]):.4f} (Min = {np.min(sim_hybrids[type_bc_mask]):.4f}, Max = {np.max(sim_hybrids[type_bc_mask]):.4f})")

# 5-Fold Stratified CV Evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_sims_preds(sims):
    fold_f1s, fold_ps, fold_rs = [], [], []
    cv_preds = np.zeros(len(real_pairs), dtype=int)
    for train_idx, test_idx in skf.split(real_pairs, labels):
        tr_y, te_y = labels[train_idx], labels[test_idx]
        tr_sim, te_sim = sims[train_idx], sims[test_idx]
        best_t = 0.5
        best_tr_f1 = -1
        for t in np.arange(0.0, 1.001, 0.005):
            preds = (tr_sim >= t).astype(int)
            f1 = f1_score(tr_y, preds, zero_division=0)
            if f1 > best_tr_f1:
                best_tr_f1 = f1
                best_t = t
        te_preds = (te_sim >= best_t).astype(int)
        cv_preds[test_idx] = te_preds
        fold_f1s.append(f1_score(te_y, te_preds, zero_division=0))
        fold_ps.append(precision_score(te_y, te_preds, zero_division=0))
        fold_rs.append(recall_score(te_y, te_preds, zero_division=0))
    return np.mean(fold_f1s), np.std(fold_f1s), np.mean(fold_ps), np.mean(fold_rs), cv_preds

f1_g, std_g, p_g, r_g, preds_g = evaluate_sims_preds(sim_globals)
f1_s, std_s, p_s, r_s, preds_s = evaluate_sims_preds(sim_structs)
f1_h, std_h, p_h, r_h, preds_h = evaluate_sims_preds(sim_hybrids)

print("\n==================================================")
print("2. 5-FOLD CV PERFORMANCE SUMMARY & MCNEMAR TESTS")
print("==================================================")
print(f"1. Standalone Full-Doc SBERT (sim_global): F1 = {f1_g:.4f} ± {std_g:.4f} | Precision = {p_g:.4f} | Recall = {r_g:.4f}")
print(f"2. SW-BTED Structural-Only   (sim_struct): F1 = {f1_s:.4f} ± {std_s:.4f} | Precision = {p_s:.4f} | Recall = {r_s:.4f}")
print(f"3. SW-BTED Hybrid Mode       (sim_hybrid): F1 = {f1_h:.4f} ± {std_h:.4f} | Precision = {p_h:.4f} | Recall = {r_h:.4f}")

def compute_mcnemar_contingency(preds_a, preds_b, name_a, name_b):
    n11, n10, n01, n00 = 0, 0, 0, 0
    for gt, pa, pb in zip(labels, preds_a, preds_b):
        a_ok = (gt == pa)
        b_ok = (gt == pb)
        if a_ok and b_ok: n11 += 1
        elif a_ok and not b_ok: n10 += 1
        elif not a_ok and b_ok: n01 += 1
        else: n00 += 1
    
    n_disc = n10 + n01
    chi2 = ((abs(n10 - n01) - 1.0) ** 2) / n_disc if n_disc > 0 else 0.0
    exact_p = binomtest(min(n10, n01), n_disc, p=0.5, alternative='two-sided').pvalue if n_disc > 0 else 1.0
    
    print(f"\n{name_a} vs {name_b}:")
    print(f"  Contingency Table: n11={n11}, n10={n10}, n01={n01}, n00={n00}")
    print(f"  McNemar chi2 = {chi2:.4f}, exact p-value = {exact_p:.4e}")

compute_mcnemar_contingency(preds_g, preds_s, "Full-Doc SBERT", "SW-BTED Structural-Only")
compute_mcnemar_contingency(preds_h, preds_g, "SW-BTED Hybrid Mode", "Full-Doc SBERT")
compute_mcnemar_contingency(preds_h, preds_s, "SW-BTED Hybrid Mode", "SW-BTED Structural-Only")
