import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import spacy
import importlib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Any
from src.node import CapstoneNode

# Lazy load CSO & Tech Equivalence
try:
    ontology_lookup_module = importlib.import_module("src.03_ontology_lookup")
    CSOLookup = ontology_lookup_module.CSOLookup
except Exception:
    CSOLookup = None

try:
    tech_module = importlib.import_module("src.tech_equivalence")
    TECH_GROUPS = tech_module.TECH_GROUPS
except Exception:
    TECH_GROUPS = {}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", 
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "his", "how", "i", "if", "in", "into", 
    "is", "it", "its", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "our", "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", 
    "their", "theirs", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", 
    "you", "your", "yours", "yourself", "yourselves"
}

def clean_and_normalize_leaf(label: str, cso_lookup, nlp: spacy.Language) -> str:
    tokens = label.lower().strip().split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    if not filtered_tokens:
        filtered_tokens = tokens
    phrase = " ".join(filtered_tokens)
    
    res = None
    if cso_lookup:
        cso_res = cso_lookup.lookup(phrase)
        if cso_res:
            res = cso_res["cso_concept"]
            
    if not res:
        if phrase in TECH_GROUPS:
            res = TECH_GROUPS[phrase]
        else:
            for t in filtered_tokens:
                if t in TECH_GROUPS:
                    res = TECH_GROUPS[t]
                    break
            
    if not res:
        doc = nlp(phrase)
        res = " ".join([token.lemma_ for token in doc])
        
    return res

def extract_candidates(doc, cso_lookup) -> List[Dict[str, str]]:
    candidates = []
    seen = set()
    
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip().lower()
        if not txt or txt in seen or len(txt) < 3:
            continue
        seen.add(txt)
        is_tech = False
        if txt in TECH_GROUPS:
            is_tech = True
        elif cso_lookup and cso_lookup.lookup(chunk.text):
            is_tech = True
        role = "technology" if is_tech else "concept"
        candidates.append({"text": chunk.text, "role": role})
        
    for token in doc:
        txt = token.text.lower()
        if txt in seen or len(txt) <= 2:
            continue
        if token.pos_ in ("NOUN", "PROPN"):
            seen.add(txt)
            is_tech = False
            if txt in TECH_GROUPS:
                is_tech = True
            elif cso_lookup and cso_lookup.lookup(token.text):
                is_tech = True
            role = "technology" if is_tech else "concept"
            candidates.append({"text": token.text, "role": role})
        elif token.pos_ == "VERB":
            seen.add(txt)
            candidates.append({"text": token.lemma_, "role": "action"})
            
    return candidates

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
    print("Running JD Tree Normalizer & Terminology Extractor Pipeline...")
    
    unnorm_path = "Data/dataset/linkedin_jd/trees_unnormalized.json"
    output_path = "Data/dataset/linkedin_jd/trees.json"
    
    if not os.path.exists(unnorm_path):
        print(f"Error: {unnorm_path} not found.")
        return
        
    with open(unnorm_path, "r", encoding="utf-8") as f:
        trees_raw = json.load(f)
        
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        print("Fatal: Could not load en_core_web_sm.")
        return
        
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

    roots = {}
    all_t3_nodes = []
    
    for code, tree_dict in trees_raw.items():
        root_node = CapstoneNode.from_dict(tree_dict)
        roots[code] = root_node
        
        def collect_t3(node):
            if node.depth == 3:
                all_t3_nodes.append(node)
            for child in node.children:
                collect_t3(child)
        collect_t3(root_node)
        
    print(f"Collected {len(all_t3_nodes)} T3 nodes. Running spaCy Terminology Extraction...")
    t3_texts = [n.raw_text if n.raw_text else n.label for n in all_t3_nodes]
    docs = list(nlp.pipe(t3_texts, batch_size=256))
    
    # Terminology Extraction (Stage 2)
    all_t4_nodes = []
    for node, doc in zip(all_t3_nodes, docs):
        candidates = extract_candidates(doc, cso_lookup)
        for cand in candidates:
            t4_node = CapstoneNode(
                label=cand["text"],
                schema_class="TerminologyVerification",
                depth=4,
                source_role=cand["role"]
            )
            node.children.append(t4_node)
            all_t4_nodes.append(t4_node)
            
    print(f"Extracted {len(all_t4_nodes)} T4 Terminology nodes.")
    
    # Leaf Normalization (Stage 3)
    unique_labels = list(set(node.label for node in all_t4_nodes))
    print(f"Normalizing {len(unique_labels)} unique terms...")
    norm_map = {lbl: clean_and_normalize_leaf(lbl, cso_lookup, nlp) for lbl in unique_labels}
    
    for node in all_t4_nodes:
        node.label = norm_map[node.label]
        
    # Fit TF-IDF
    print("Computing TF-IDF weights...")
    doc_texts = [get_full_document_text(roots[code]) for code in roots]
    doc_codes = list(roots.keys())
    vectorizer = TfidfVectorizer(stop_words='english')
    
    if doc_texts and any(doc_texts):
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        feature_names = vectorizer.get_feature_names_out()
        feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        for doc_idx, code in enumerate(doc_codes):
            root_node = roots[code]
            doc_t4 = []
            def collect_doc_t4(node):
                if node.depth == 4:
                    doc_t4.append(node)
                for child in node.children:
                    collect_doc_t4(child)
            collect_doc_t4(root_node)
            
            for node in doc_t4:
                lbl_lower = node.label.lower()
                weight = 0.5
                if lbl_lower in feature_to_idx:
                    f_idx = feature_to_idx[lbl_lower]
                    weight = float(tfidf_matrix[doc_idx, f_idx])
                    if weight == 0.0:
                        weight = 0.1
                node.tfidf_weight = weight
    else:
        for node in all_t4_nodes:
            node.tfidf_weight = 0.5
            
    # Embeddings
    print("Loading SentenceTransformer all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Embedding {len(all_t3_nodes)} T3 nodes...")
    t3_texts = [n.normalized_text if n.normalized_text else n.raw_text for n in all_t3_nodes]
    if t3_texts:
        t3_embs = model.encode(t3_texts, batch_size=256, show_progress_bar=True)
        for node, emb in zip(all_t3_nodes, t3_embs):
            node.embedding = emb.tolist()
            
    print(f"Embedding {len(roots)} Root nodes...")
    root_texts = [get_full_document_text(roots[code]) for code in roots]
    root_texts = [t if t.strip() else "None" for t in root_texts]
    root_embs = model.encode(root_texts, batch_size=256, show_progress_bar=True)
    for code, emb in zip(roots, root_embs):
        roots[code].embedding = emb.tolist()
        
    # Save trees.json
    trees_normalized = {code: root.to_dict() for code, root in roots.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trees_normalized, f, ensure_ascii=False, indent=2)
    print(f"Saved normalized trees to {output_path}")

if __name__ == "__main__":
    main()
