"""
Runtime Benchmark
This script measures runtime scaling of the SW-BTED algorithm as a function of the database size N.
"""
import json
import time
import random
import os
import pandas as pd
import numpy as np
import pickle
import importlib

# Dynamic imports
sw_bted_module = importlib.import_module("src.05_sw_bted")
SWCostModel = sw_bted_module.SWCostModel
normalize_similarity = sw_bted_module.normalize_similarity
dict_to_node = sw_bted_module.dict_to_node

def main():
    print("Starting Runtime Benchmark...")
    
    # Load dataset
    trees_raw = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    trees = {k: dict_to_node(v) for k, v in trees_raw.items()}
    
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    
    # Optimal cost model
    cost_model = SWCostModel(alpha=0.3, beta=0.7, gamma=0.3, cso_graph=cso_graph, max_depth=max_depth)
    
    keys = list(trees.keys())
    num_trees = len(keys)
    print(f"Total available trees in database: {num_trees}")
    
    N_sizes = [50, 100, 200, 300]
    N_sizes = [n for n in N_sizes if n < num_trees]
    
    results = []
    num_trials = 5
    print(f"Running {num_trials} trials for each database size N...")
    
    for N in N_sizes:
        trial_times = []
        for trial in range(num_trials):
            query_key = random.choice(keys)
            query_tree = trees[query_key]
            
            db_keys = [k for k in keys if k != query_key]
            db_subset_keys = random.sample(db_keys, N)
            db_subset = [trees[k] for k in db_subset_keys]
            
            start_time = time.perf_counter()
            for db_tree in db_subset:
                _ = normalize_similarity(query_tree, db_tree, cost_model)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            trial_times.append(elapsed)
            
        mean_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        time_per_comparison_ms = (mean_time / N) * 1000
        
        print(f"N = {N:3} | Total Time: {mean_time:.4f}s (±{std_time:.4f}s) | Avg per comparison: {time_per_comparison_ms:.2f} ms")
        
        results.append({
            "Database_Size_N": N,
            "Total_Time_Seconds": mean_time,
            "Std_Time_Seconds": std_time,
            "Avg_Time_Per_Comparison_MS": time_per_comparison_ms
        })
        
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/runtime_benchmark.csv", index=False)
    print("Saved runtime benchmark results to results/runtime_benchmark.csv")

if __name__ == "__main__":
    main()
