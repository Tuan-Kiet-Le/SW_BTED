import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import spacy
import importlib
from typing import Dict, List, Any

# Load node definitions
from src.node import CapstoneNode

# Load CSO Lookup
try:
    ontology_lookup_module = importlib.import_module("src.03_ontology_lookup")
    CSOLookup = ontology_lookup_module.CSOLookup
except Exception as e:
    CSOLookup = None

# Load Tech Equivalence Map
try:
    tech_module = importlib.import_module("src.tech_equivalence")
    TECH_GROUPS = tech_module.TECH_GROUPS
except Exception as e:
    TECH_GROUPS = {}

def extract_candidates_from_doc(doc, cso_lookup, tech_map: Dict[str, str]) -> List[Dict[str, str]]:
    candidates = []
    seen_texts = set()
    
    # Classify Noun Phrases
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        if not chunk_lower or chunk_lower in seen_texts:
            continue
        seen_texts.add(chunk_lower)
        
        is_tech = False
        if chunk_lower in tech_map:
            is_tech = True
        else:
            for token in chunk:
                if token.text.lower() in tech_map:
                    is_tech = True
                    break
            if not is_tech and cso_lookup:
                if cso_lookup.lookup(chunk_text):
                    is_tech = True
                    
        role = "technology" if is_tech else "concept"
        candidates.append({"text": chunk_text, "role": role})
        
    # Classify Verbs and individual Nouns/PROPNs
    for token in doc:
        token_lower = token.text.lower()
        if token_lower in seen_texts or len(token_lower) <= 2:
            continue
            
        if token.pos_ in ("NOUN", "PROPN"):
            seen_texts.add(token_lower)
            is_tech = False
            if token_lower in tech_map:
                is_tech = True
            elif cso_lookup and cso_lookup.lookup(token.text):
                is_tech = True
            role = "technology" if is_tech else "concept"
            candidates.append({"text": token.text, "role": role})
            
        elif token.pos_ == "VERB":
            seen_texts.add(token_lower)
            candidates.append({"text": token.lemma_, "role": "action"})
            
    return candidates

def main():
    print("Running Terminology Extractor Stage 2 (4-Layer)...")
    trees_t3_path = "data/dataset/trees_t3.json"
    output_path = "data/dataset/trees_t4_unnormalized.json"
    
    if not os.path.exists(trees_t3_path):
        print(f"Error: {trees_t3_path} not found. Please run Stage 1 parser first.")
        return
        
    print(f"Reading T3 trees from {trees_t3_path}...")
    with open(trees_t3_path, "r", encoding="utf-8") as f:
        trees_raw = json.load(f)
        
    print("Loading spaCy model en_core_web_trf...")
    try:
        nlp = spacy.load("en_core_web_trf")
    except Exception as e:
        print(f"Error loading en_core_web_trf ({e}). Falling back to en_core_web_sm.")
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            print("Fatal: Could not load any spaCy model.")
            return

    cso_lookup = None
    if CSOLookup:
        print("Initializing CSOLookup...")
        cso_lookup = CSOLookup()
        def quick_lookup(label: str) -> bool:
            label_clean = label.strip().lower()
            return (label_clean in cso_lookup.label_to_concept or 
                    label_clean in cso_lookup.concept_to_label)
        cso_lookup.lookup = quick_lookup

    roots = {}
    all_t3_nodes = []
    
    for code, tree_dict in trees_raw.items():
        root_node = CapstoneNode.from_dict(tree_dict)
        roots[code] = root_node
        
        def collect_t3(node):
            if node.depth == 3 and node.schema_class == "IntentMatching":
                all_t3_nodes.append(node)
            for child in node.children:
                collect_t3(child)
        collect_t3(root_node)

    print(f"Collected {len(all_t3_nodes)} Intent Matching nodes. Parsing in batches...")
    
    texts = [node.normalized_text if node.normalized_text else node.raw_text for node in all_t3_nodes]
    texts = [t if t.strip() else "None" for t in texts]
    
    docs = list(nlp.pipe(texts, batch_size=256))
    
    print("Extracting Terminology and building T4 nodes...")
    for node, doc in zip(all_t3_nodes, docs):
        candidates = extract_candidates_from_doc(doc, cso_lookup, TECH_GROUPS)
        
        for cand in candidates:
            t4_node = CapstoneNode(
                label=cand["text"],
                schema_class="TerminologyVerification",
                depth=4,
                source_role=cand["role"]
            )
            node.children.append(t4_node)
                
    trees_t4 = {code: root_node.to_dict() for code, root_node in roots.items()}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trees_t4, f, ensure_ascii=False, indent=2)
    print(f"Saved unnormalized T4 trees to {output_path}")
    print("Stage 2 Terminology Extractor complete.")

if __name__ == "__main__":
    main()
