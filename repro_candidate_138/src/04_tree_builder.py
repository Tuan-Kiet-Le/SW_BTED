import json
import os
import math
from typing import Dict, Any, List
import importlib
# We will import 02_keyword_extractor dynamically because of its numeric prefix
keyword_extractor_module = importlib.import_module("src.02_keyword_extractor")
extract_keywords_from_text = keyword_extractor_module.extract_keywords_from_text
# We will import 03_ontology_lookup dynamically because of its numeric prefix
ontology_lookup_module = importlib.import_module("src.03_ontology_lookup")
CSOLookup = ontology_lookup_module.CSOLookup

MAPPING = {
    "Context": "context",
    "Problem": "research_problem",
    "Solution": "proposed_solutions",
    "Theory": "theory",
    "Deliverables": "products",
    "Methodology": "research_scope_methodology",
    "Timeline": "proposed_tasks",
    "References": "related_works"
}

def node_to_dict(node) -> dict:
    return {
        "label": node["label"],
        "schema_class": node["schema_class"],
        "depth": node["depth"],
        "idf": node["idf"],
        "cso_ancestors": node.get("cso_ancestors", []),
        "children": [node_to_dict(child) for child in node.get("children", [])]
    }

def build_all_trees():
    json_path = os.path.join("data", "processed", "topics.json")
    output_trees_path = os.path.join("data", "dataset", "trees.json")
    output_idf_path = os.path.join("data", "dataset", "idf_weights.json")

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run Stage 1 parser first.")
        return

    print(f"Reading topics from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    N = len(records)
    print(f"Loaded {N} records. Computing schema class IDFs...")

    # Calculate document frequency for each schema class
    df = {}
    for c, field_name in MAPPING.items():
        count = 0
        for r in records:
            val = r.get(field_name)
            if val is not None and str(val).strip() != "":
                count += 1
        df[c] = count

    # Compute smoothed IDF weights
    idf = {}
    for c in MAPPING:
        idf[c] = round(math.log((N + 1) / (df[c] + 1)) + 0.2, 4)
    
    print(f"Computed IDF Weights: {idf}")
    
    # Ensure dataset directory exists
    os.makedirs(os.path.dirname(output_trees_path), exist_ok=True)
    
    # Save IDFs
    with open(output_idf_path, 'w', encoding='utf-8') as f:
        json.dump(idf, f, indent=2)

    # Initialize CSO Lookup
    cso = CSOLookup()

    print("Building document trees...")
    trees = {}
    
    for i, r in enumerate(records):
        doc_id = str(r["id"])
        topic_code = r.get("topic_code", f"TOPIC_{doc_id}")
        title = r.get("title", "Untitled Project")

        # Root Node
        root = {
            "label": title,
            "schema_class": "root",
            "depth": 1,
            "idf": 1.0,
            "cso_ancestors": [],
            "children": []
        }

        # Build Section-level children
        for c, field_name in MAPPING.items():
            text_val = r.get(field_name)
            if text_val is None or str(text_val).strip() == "":
                continue

            section_node = {
                "label": c,
                "schema_class": c,
                "depth": 2,
                "idf": idf[c],
                "cso_ancestors": [],
                "children": []
            }

            # Extract keywords for this section (using globally imported extractor)
            keywords = extract_keywords_from_text(text_val, top_k=8)

            for kw, score in keywords:
                # Lookup in CSO
                cso_res = cso.lookup(kw)
                
                kw_node = {
                    "label": cso_res["cso_concept"] if cso_res else kw,
                    "schema_class": c,
                    "depth": 3,
                    "idf": idf[c],
                    "cso_ancestors": cso_res["ancestors"] if cso_res else [],
                    "children": []
                }
                section_node["children"].append(kw_node)

            root["children"].append(section_node)

        trees[topic_code] = root
        
        if (i + 1) % 50 == 0 or (i + 1) == N:
            print(f"Processed {i + 1}/{N} document trees...")

    with open(output_trees_path, 'w', encoding='utf-8') as f:
        json.dump(trees, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(trees)} trees to {output_trees_path}")

if __name__ == "__main__":
    build_all_trees()
