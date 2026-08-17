import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import spacy
import importlib
from sentence_transformers import SentenceTransformer
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

# Simple stopwords set
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", 
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", 
    "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", 
    "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "im", "ive", "if", "in", "into", 
    "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
    "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", 
    "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", 
    "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", 
    "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", 
    "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", 
    "your", "yours", "yourself", "yourselves"
}

def clean_and_normalize_leaf(label: str, cso_lookup, nlp: spacy.Language) -> str:
    # 1. Lowercase + remove stopwords
    tokens = label.lower().strip().split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    if not filtered_tokens:
        filtered_tokens = tokens
    phrase = " ".join(filtered_tokens)
    
    res = None
    # 2. CSO lookup
    if cso_lookup:
        cso_res = cso_lookup.lookup(phrase)
        if cso_res:
            res = cso_res["cso_concept"]
            
    # 3. Tech Equivalence Map
    if not res:
        if phrase in TECH_GROUPS:
            res = TECH_GROUPS[phrase]
        else:
            for t in filtered_tokens:
                if t in TECH_GROUPS:
                    res = TECH_GROUPS[t]
                    break
            
    # 4. Lemmatization via spaCy
    if not res:
        doc = nlp(phrase)
        res = " ".join([token.lemma_ for token in doc])
        
    return res

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
    print("Running Normalizer Stage 3 (Optimized Batched Mode)...")
    trees_t4_path = "data/dataset/trees_t4_unnormalized.json"
    output_path = "data/dataset/trees.json"
    
    if not os.path.exists(trees_t4_path):
        print(f"Error: {trees_t4_path} not found. Please run Stage 2 terminology extractor first.")
        return
        
    print(f"Reading T4 unnormalized trees from {trees_t4_path}...")
    with open(trees_t4_path, "r", encoding="utf-8") as f:
        trees_raw = json.load(f)
        
    # Load spaCy
    print("Loading spaCy model for lemmatization...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading en_core_web_sm ({e}). Falling back to en_core_web_trf.")
        try:
            nlp = spacy.load("en_core_web_trf")
        except Exception:
            print("Fatal: Could not load any spaCy model.")
            return

    # Load CSO Lookup
    cso_lookup = None
    if CSOLookup:
        print("Initializing CSOLookup...")
        cso_lookup = CSOLookup()
        def quick_lookup(label: str) -> bool:
            label_clean = label.strip().lower()
            if label_clean in cso_lookup.label_to_concept:
                concept = cso_lookup.label_to_concept[label_clean]
            elif label_clean in cso_lookup.concept_to_label:
                concept = label_clean
            else:
                return None
            return {
                "cso_concept": concept,
                "label": cso_lookup.concept_to_label[concept]
            }
        cso_lookup.lookup = quick_lookup

    # Load Sentence Transformer
    print("Loading SentenceTransformer all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect all roots, T3 nodes, and T4 nodes
    roots = {}
    all_t3_nodes = []
    all_t4_nodes = []
    
    for code, tree_dict in trees_raw.items():
        root_node = CapstoneNode.from_dict(tree_dict)
        roots[code] = root_node
        
        def collect_nodes(node):
            if node.depth == 3 and node.schema_class == "IntentMatching":
                all_t3_nodes.append(node)
            elif node.depth == 4 and node.schema_class == "TerminologyVerification":
                all_t4_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        collect_nodes(root_node)
        
    print(f"Collected {len(all_t3_nodes)} T3 Intent nodes, and {len(all_t4_nodes)} T4 Term nodes.")
    
    # 1. Normalize unique T4 term labels
    unique_term_labels = list(set(node.label for node in all_t4_nodes))
    print(f"Normalizing {len(unique_term_labels)} unique term labels...")
    
    norm_map = {}
    for i, lbl in enumerate(unique_term_labels):
        if i % 100 == 0:
            print(f"Normalized {i}/{len(unique_term_labels)}...")
        norm_map[lbl] = clean_and_normalize_leaf(lbl, cso_lookup, nlp)
        
    # Map back to T4 nodes
    print("Updating term nodes in trees...")
    for node in all_t4_nodes:
        norm_lbl = norm_map[node.label]
        node.label = norm_lbl
        
    # Fit TF-IDF globally to assign weights to terms
    print("Computing TF-IDF weights for term nodes...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    doc_texts = []
    doc_codes = list(roots.keys())
    for code in doc_codes:
        doc_texts.append(get_full_document_text(roots[code]))
        
    vectorizer = TfidfVectorizer(stop_words='english')
    if doc_texts:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        feature_names = vectorizer.get_feature_names_out()
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        for doc_idx, code in enumerate(doc_codes):
            root_node = roots[code]
            doc_t4_nodes = []
            def collect_doc_t4(node):
                if node.depth == 4 and node.schema_class == "TerminologyVerification":
                    doc_t4_nodes.append(node)
                for child in node.children:
                    collect_doc_t4(child)
            collect_doc_t4(root_node)
            
            for node in doc_t4_nodes:
                lbl_lower = node.label.lower()
                weight = 0.5  # default fallback
                if lbl_lower in feature_to_idx:
                    feat_idx = feature_to_idx[lbl_lower]
                    weight = float(tfidf_matrix[doc_idx, feat_idx])
                    if weight == 0.0:
                        weight = 0.1
                node.tfidf_weight = weight
    else:
        for node in all_t4_nodes:
            node.tfidf_weight = 0.5

    # 2. Embed T3 Intent nodes
    if all_t3_nodes:
        print(f"Embedding {len(all_t3_nodes)} T3 Intent nodes...")
        t3_texts = [node.normalized_text if node.normalized_text else node.raw_text for node in all_t3_nodes]
        t3_embs = model.encode(t3_texts, batch_size=256, show_progress_bar=True)
        for node, emb in zip(all_t3_nodes, t3_embs):
            node.embedding = emb.tolist()
            
    # 3. Embed Root nodes (T1) using full doc text
    print(f"Embedding {len(roots)} Root nodes...")
    root_texts = []
    root_list = []
    for code, root_node in roots.items():
        doc_text = get_full_document_text(root_node)
        if not doc_text.strip():
            doc_text = "None"
        root_texts.append(doc_text)
        root_list.append(root_node)
        
    root_embs = model.encode(root_texts, batch_size=256, show_progress_bar=True)
    for root_node, emb in zip(root_list, root_embs):
        root_node.embedding = emb.tolist()
        
    # Convert roots back to dicts
    trees_normalized = {code: root_node.to_dict() for code, root_node in roots.items()}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trees_normalized, f, ensure_ascii=False, indent=2)
        
    print(f"Saved normalized and embedded trees to {output_path}")
    print("Stage 3 Normalizer complete.")

if __name__ == "__main__":
    main()
