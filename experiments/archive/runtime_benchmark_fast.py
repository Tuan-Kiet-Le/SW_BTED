"""
Runtime Benchmark (Fast version)
Measures per-comparison latency and scaling for SW-BTED and all baselines.
"""
import json, time, random, os, pickle, importlib, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np

random.seed(42)

# ── dynamic imports ──────────────────────────────────────────
sw_mod = importlib.import_module("src.05_sw_bted")
SWCostModel      = sw_mod.SWCostModel
normalize_sim    = sw_mod.normalize_similarity
dict_to_node     = sw_mod.dict_to_node

base_mod = importlib.import_module("src.baselines")
get_cosine_tfidf  = base_mod.get_cosine_tfidf_similarity
get_sbert         = base_mod.get_sbert_similarity
get_standard_ted  = base_mod.get_standard_ted_similarity
get_pqgram        = base_mod.get_ged_similarity
get_section_cos   = base_mod.get_section_cosine_similarity

def main():
    print("Loading data...")
    trees_raw  = json.load(open("data/dataset/trees.json", encoding="utf-8"))
    trees      = {k: dict_to_node(v) for k, v in trees_raw.items()}
    full_texts = json.load(open("data/dataset/full_texts.json", encoding="utf-8"))

    cso_data   = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph  = cso_data["graph"]
    max_depth  = cso_data.get("max_depth", 19)

    cost_model = SWCostModel(alpha=0.6, beta=0.7, gamma=0.3,
                             cso_graph=cso_graph, max_depth=max_depth)

    keys = [k for k in trees.keys() if "_plag" not in k]  # use original trees only
    num_trees = len(keys)
    print(f"Original trees available: {num_trees}")

    # ── Warm-up (small, just to load caches) ─────────────────
    print("Warming up (50 comparisons)...")
    for _ in range(50):
        a, b = random.sample(keys, 2)
        normalize_sim(trees[a], trees[b], cost_model)
    print("Warm-up done.")

    N_SIZES  = [10, 25, 50, 100, 150, 200]
    N_SIZES  = [n for n in N_SIZES if n < num_trees]
    NUM_TRIALS = 5

    random.seed(42)
    query_keys = random.sample(keys, NUM_TRIALS)

    # ── Methods to benchmark ──────────────────────────────────
    methods = {
        "SW-BTED": lambda a, b: normalize_sim(trees[a], trees[b], cost_model),
        "B1_TF-IDF": lambda a, b: get_cosine_tfidf(
            {a: trees[a], b: trees[b]},
            pd.DataFrame([{"doc_a": a, "doc_b": b}]),
            full_texts
        )[0],
        "B2_SBERT": lambda a, b: get_sbert(
            {a: trees[a], b: trees[b]},
            pd.DataFrame([{"doc_a": a, "doc_b": b}]),
            full_texts
        )[0],
        "B3_Std-TED": lambda a, b: get_standard_ted(
            {a: trees[a], b: trees[b]},
            pd.DataFrame([{"doc_a": a, "doc_b": b}])
        )[0],
        "B4_GED": lambda a, b: get_pqgram(
            {a: trees[a], b: trees[b]},
            pd.DataFrame([{"doc_a": a, "doc_b": b}])
        )[0],
        "B5_Sec-Cos": lambda a, b: get_section_cos(
            {a: trees[a], b: trees[b]},
            pd.DataFrame([{"doc_a": a, "doc_b": b}]),
            full_texts
        )[0],
    }

    results = []
    for method_name, fn in methods.items():
        print(f"\n[{method_name}] Benchmarking N = {N_SIZES}...")
        for N in N_SIZES:
            trial_times = []
            for trial_idx, qk in enumerate(query_keys):
                pool = [k for k in keys if k != qk]
                random.seed(trial_idx * 100)
                subset = random.sample(pool, N)

                t0 = time.perf_counter()
                for dk in subset:
                    fn(qk, dk)
                elapsed = time.perf_counter() - t0
                trial_times.append(elapsed)

            mean_t  = np.mean(trial_times)
            std_t   = np.std(trial_times)
            ms_each = (mean_t / N) * 1000

            print(f"  N={N:3d} | {mean_t:.3f}s ±{std_t:.3f}s | {ms_each:.2f} ms/comparison")
            results.append({
                "Method": method_name,
                "N": N,
                "Total_Time_s_Mean": round(mean_t, 4),
                "Total_Time_s_Std":  round(std_t, 4),
                "Ms_Per_Comparison": round(ms_each, 3),
            })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/runtime_benchmark.csv", index=False)
    print("\nSaved to results/runtime_benchmark.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
