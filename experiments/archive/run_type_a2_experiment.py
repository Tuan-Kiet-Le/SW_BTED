import json
import os
import sys
import numpy as np
import random
import importlib
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set random seed
random.seed(42)

def main():
    print("Running Type A2 (Cross-section Edge Splicing) Experiment...")
    
    # 1. Load original data
    trees_path = "data/dataset/trees.json"
    full_texts_path = "data/dataset/full_texts.json"
    
    if not os.path.exists(trees_path) or not os.path.exists(full_texts_path):
        print("Error: dataset files not found.")
        return
        
    with open(trees_path, "r", encoding="utf-8") as f:
        trees_raw = json.load(f)
    with open(full_texts_path, "r", encoding="utf-8") as f:
        full_texts = json.load(f)
        
    # Get original documents (no _plag, no _hard)
    original_docs = [k for k in trees_raw.keys() if not k.endswith("_plag") and not k.endswith("_hard")]
    print(f"Found {len(original_docs)} original documents.")
    
    # Select 10 pairs (Doc X -> source of plagiarism, Doc Y -> receiver/host of spliced section)
    # Ensure they are from different topics
    selected_pairs = []
    available_docs = original_docs.copy()
    random.shuffle(available_docs)
    
    for i in range(10):
        doc_x = available_docs[2 * i]
        doc_y = available_docs[2 * i + 1]
        selected_pairs.append((doc_x, doc_y))
        
    print("Selected 10 pairs for splicing:")
    for idx, (x, y) in enumerate(selected_pairs):
        print(f"  {idx+1}: Source X = {x} | Host Y = {y}")
        
    # 2. Build Type A2 mutated documents
    # Splicing: Take Solution section of X and put it into Theory section of Y
    a2_trees = {}
    a2_full_texts = {}
    
    for idx, (doc_x, doc_y) in enumerate(selected_pairs):
        doc_a2_code = f"{doc_y}_a2"
        
        # Splicing text
        y_text = full_texts[doc_y].copy()
        x_text = full_texts[doc_x]
        
        # Put Solution of X into Theory of Y
        y_text["Theory"] = x_text.get("Solution", "")
        a2_full_texts[doc_a2_code] = y_text
        
        # Splicing tree (6-layer)
        x_tree = trees_raw[doc_x]
        y_tree = json.loads(json.dumps(trees_raw[doc_y])) # deep copy
        
        # Find D2_FUNCTIONAL in X
        x_d2 = None
        for child in x_tree.get("children", []):
            if child.get("schema_class") == "D2_FUNCTIONAL":
                x_d2 = child
                break
                
        # Find D3_TECHNICAL_REALIZATION in Y
        y_d3_idx = None
        for c_idx, child in enumerate(y_tree.get("children", [])):
            if child.get("schema_class") == "D3_TECHNICAL_REALIZATION":
                y_d3_idx = c_idx
                break
                
        if x_d2 is not None and y_d3_idx is not None:
            # Clone all children of D2_FUNCTIONAL from X
            spliced_children = json.loads(json.dumps(x_d2.get("children", [])))
            
            # Change any "FunctionalGroup" schema_class and label to "Methodology" to make it valid in D3
            def adapt_to_d3(node):
                if node.get("schema_class") == "FunctionalGroup":
                    node["schema_class"] = "Methodology"
                    node["label"] = "Methodology"
                for c in node.get("children", []):
                    adapt_to_d3(c)
            for c in spliced_children:
                adapt_to_d3(c)
                
            # Replace Y's D3_TECHNICAL_REALIZATION children with these spliced children
            y_tree["children"][y_d3_idx]["children"] = spliced_children
            
        a2_trees[doc_a2_code] = y_tree
        
    # 3. Load SW-BTED module and CSO Graph to calculate similarities
    sw_bted_module = importlib.import_module("src.05_sw_bted")
    SWCostModel = sw_bted_module.SWCostModel
    normalize_similarity = sw_bted_module.normalize_similarity
    dict_to_node = sw_bted_module.dict_to_node
    
    baselines_module = importlib.import_module("src.baselines")
    get_section_cosine_similarity = baselines_module.get_section_cosine_similarity
    
    cso_data = pickle.load(open("data/processed/cso_graph.pkl", "rb"))
    cso_graph = cso_data["graph"]
    max_depth = cso_data.get("max_depth", 19)
    
    alpha, beta = 0.8, 0.7
    cost_model = SWCostModel(alpha=alpha, beta=beta, gamma=1-beta, cso_graph=cso_graph, max_depth=max_depth)
    
    # Calculate similarities for Type A2 pairs
    results = []
    print("\nCalculating similarities for X vs Y_a2 (label = 1):")
    for idx, (doc_x, doc_y) in enumerate(selected_pairs):
        doc_a2_code = f"{doc_y}_a2"
        
        # Convert dictionary format to node objects for SW-BTED
        node_x = dict_to_node(trees_raw[doc_x])
        node_a2 = dict_to_node(a2_trees[doc_a2_code])
        
        # Calculate SW-BTED similarity
        sw_sim = normalize_similarity(node_x, node_a2, cost_model)
        
        # Calculate B5 (Section Cosine) similarity manually for this pair
        # We construct a temporary DataFrame to use the baselines function
        temp_pairs = [{'doc_a': doc_x, 'doc_b': doc_a2_code}]
        import pandas as pd
        df_temp = pd.DataFrame(temp_pairs)
        
        # Section Cosine expects the trees and texts to contain the code
        temp_trees = {doc_x: node_x, doc_a2_code: node_a2}
        temp_texts = {doc_x: full_texts[doc_x], doc_a2_code: a2_full_texts[doc_a2_code]}
        
        seccos_sims = get_section_cosine_similarity(temp_trees, df_temp, temp_texts)
        sec_cos_sim = seccos_sims[0]
        
        results.append({
            "pair": idx + 1,
            "doc_x": doc_x,
            "doc_y": doc_y,
            "sw_sim": sw_sim,
            "sec_cos_sim": sec_cos_sim
        })
        print(f"  Pair {idx+1}: SW-BTED = {sw_sim:.4f} | B5 (Section Cosine) = {sec_cos_sim:.4f}")
        
    # Print average metrics
    avg_sw = np.mean([r["sw_sim"] for r in results])
    avg_seccos = np.mean([r["sec_cos_sim"] for r in results])
    print("\nAverage Similarities on Type A2 (Cross-section Edge Splicing):")
    print(f"  SW-BTED Mean Similarity: {avg_sw:.4f}")
    print(f"  B5 Section Cosine Mean Similarity: {avg_seccos:.4f}")
    
    # Print Detection rate under threshold
    # SW-BTED threshold = 0.10 (optimal from 5-fold CV), B5 threshold = 0.20
    sw_detected = sum(1 for r in results if r["sw_sim"] >= 0.10)
    seccos_detected = sum(1 for r in results if r["sec_cos_sim"] >= 0.20)
    print(f"\nDetection Rate (out of 10):")
    print(f"  SW-BTED (Threshold = 0.10): {sw_detected}/10 ({sw_detected*10}%)")
    print(f"  B5 Section Cosine (Threshold = 0.20): {seccos_detected}/10 ({seccos_detected*10}%)")

if __name__ == "__main__":
    main()
