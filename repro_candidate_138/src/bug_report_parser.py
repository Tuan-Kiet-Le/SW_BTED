"""
Bug Report 4-Tier Tree Parser
Adheres strictly to AGENT_TASKS_cross_domain_validation.md lines 29 & 50:
T1: ROOT (bug_id)
T2: DOMAIN (D1_Problem_Description, D2_Reproduction, D3_Environment, D4_Supporting_Evidence)
T3: SENTENCE (sentences within each domain)
T4: TECHNICAL_TERM / LEAF (extracted terms & canonical keywords)
"""
import os, sys, json, re
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from src.node import CapstoneNode

# Load Tech Equivalence Map
TEM_PATH = "src/bug_tech_equiv_map.json"
TEM_MAP = json.load(open(TEM_PATH, "r", encoding="utf-8")) if os.path.exists(TEM_PATH) else {}

def normalize_leaf(term):
    clean = term.lower().strip()
    return TEM_MAP.get(clean, clean)

def parse_bug_report_to_4tier_tree(bug_id, bug_data):
    """
    Constructs a 4-tier tree for a bug report per AGENT_TASKS_cross_domain_validation.md spec.
    """
    root = CapstoneNode(label=f"BUG_{bug_id}", schema_class="ROOT", depth=1)
    
    # T2 Domains (Bettenburg et al.)
    t2_domains = {
        'D1': CapstoneNode(label="D1_Problem_Description", schema_class="D1_BUSINESS_CONTEXT", depth=2),
        'D2': CapstoneNode(label="D2_Reproduction", schema_class="D2_FUNCTIONAL", depth=2),
        'D3': CapstoneNode(label="D3_Environment", schema_class="D3_TECHNICAL_REALIZATION", depth=2),
        'D4': CapstoneNode(label="D4_Supporting_Evidence", schema_class="D4_EXECUTION_PLANNING", depth=2)
    }
    
    for d_node in t2_domains.values():
        root.children.append(d_node)
        
    for d_key, t2_node in t2_domains.items():
        text = bug_data.get(d_key, "")
        if not text or len(text.strip()) == 0:
            text = f"Default info for {d_key}"
            
        # T3: Sentences within domain
        sentences = [s.strip() for s in re.split(r'[\.\;\n]+', text) if len(s.strip()) > 3]
        if not sentences:
            sentences = [text.strip()]
            
        for sent in sentences[:5]: # Cap at 5 sentences per domain
            t3_sentence = CapstoneNode(label="Sentence", schema_class="AtomicReq", depth=3, normalized_text=sent, raw_text=sent)
            t2_node.children.append(t3_sentence)
            
            # T4: Extracted Technical Terms / Keywords
            words = re.findall(r'\w+', sent.lower())
            keywords = [w for w in words if len(w) > 3][:5]
            if not keywords:
                keywords = ["issue"]
                
            for kw in keywords:
                canon_term = normalize_leaf(kw)
                schema_cls = "TechKeyword" if canon_term in TEM_MAP or kw in TEM_MAP else "ConceptKeyword"
                t4_term = CapstoneNode(label=canon_term, schema_class=schema_cls, depth=4)
                t3_sentence.children.append(t4_term)
                
    return root

if __name__ == "__main__":
    with open("datasets/bug_reports/sample_bugs.json", "r", encoding="utf-8") as f:
        bugs_dict = json.load(f)
        
    sample_id = list(bugs_dict.keys())[0]
    sample_tree = parse_bug_report_to_4tier_tree(sample_id, bugs_dict[sample_id])
    
    def get_all_nodes(node):
        nodes = [node]
        for c in node.children:
            nodes.extend(get_all_nodes(c))
        return nodes
        
    all_nodes = get_all_nodes(sample_tree)
    print(f"Successfully constructed 4-Tier Tree for Bug ID {sample_id}!")
    print(f"Total Tree Nodes: {len(all_nodes)}")
    print("Sample 4-Tier Tree structure (first 10 nodes):")
    for n in all_nodes[:10]:
        print(f" - Depth {n.depth} [{n.schema_class}]: {n.label}")
