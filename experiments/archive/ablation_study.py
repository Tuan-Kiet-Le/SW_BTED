import itertools
import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import importlib
import pickle
import multiprocessing
import os

# Import 05_sw_bted dynamically because of its numeric prefix
sw_bted_module = importlib.import_module("src.05_sw_bted")
SWCostModel = sw_bted_module.SWCostModel
normalize_similarity = sw_bted_module.normalize_similarity
dict_to_node = sw_bted_module.dict_to_node

# Global dictionary to hold data in worker processes
_worker_data = {}

def init_worker():
    # Load trees
    trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    trees = {k: dict_to_node(v) for k, v in trees_raw.items()}
    # Load CSO graph
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    # Load pairs
    pairs = pd.read_csv("data/dataset/pairs.csv")
    
    _worker_data["trees"] = trees
    _worker_data["cso_graph"] = cso_graph
    _worker_data["max_depth"] = max_depth
    _worker_data["pairs"] = pairs

def evaluate_config(config_params):
    alpha, beta, thresh = config_params
    
    trees = _worker_data["trees"]
    cso_graph = _worker_data["cso_graph"]
    max_depth = _worker_data["max_depth"]
    pairs = _worker_data["pairs"]
    
    cost = SWCostModel(alpha=alpha, beta=beta, gamma=1-beta, cso_graph=cso_graph, max_depth=max_depth)
    preds, labels = [], []

    for _, row in pairs.iterrows():
        sim = normalize_similarity(trees[row.doc_a], trees[row.doc_b], cost)
        preds.append(1 if sim >= thresh else 0)
        labels.append(row.label)

    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    print(f"Finished config: alpha={alpha:.2f}, beta={beta:.2f}, thresh={thresh:.2f} -> F1={f1:.4f}")
    return {
        "alpha": alpha,
        "beta": beta,
        "threshold": thresh,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def main():
    GRID = {
        "alpha": [0.3, 0.5, 0.6, 0.8],
        "beta": [0.3, 0.5, 0.7],
        "threshold": [0.4, 0.5, 0.6, 0.7],
    }

    configs = list(itertools.product(GRID["alpha"], GRID["beta"], GRID["threshold"]))
    num_configs = len(configs)
    print(f"Starting ablation study with {num_configs} configurations...")
    
    # Use multiprocessing to parallelize evaluation
    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Using {num_workers} parallel workers...")
    
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.map(evaluate_config, configs)
        
    df = pd.DataFrame(results).sort_values("f1", ascending=False)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablation.csv", index=False)
    print("\nBest config:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
