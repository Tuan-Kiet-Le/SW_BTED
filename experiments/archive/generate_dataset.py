import json
import os
import random
import pandas as pd
import copy
import pickle
import networkx as nx
import importlib
from dotenv import load_dotenv
from openai import OpenAI

# Set random seed for reproducibility
random.seed(42)

# Load environment variables
load_dotenv()

# Initialize OpenAI client if key is configured
api_key = os.environ.get("OPENAI_API_KEY")
client = None
if api_key and api_key != "your_openai_api_key_here" and api_key.strip() != "":
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully with GPT-4o-mini.")
else:
    print("WARNING: Valid OPENAI_API_KEY not found in .env. LLM-based paraphrasing will be disabled, falling back to rule-based keyword templates.")

# Dynamic imports from src
keyword_extractor_module = importlib.import_module("src.02_keyword_extractor")
extract_keywords_from_text = keyword_extractor_module.extract_keywords_from_text

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

HIGH_WEIGHT_SECTIONS = {"Solution", "Methodology", "Theory"}
LOW_WEIGHT_SECTIONS  = {"Context", "Problem", "Timeline", "References", "Deliverables"}

def categorize_topic(title, keywords):
    text = (title + " " + " ".join(keywords)).lower()
    healthcare = ["health", "medical", "clinic", "patient", "hospital", "doctor", "medicine", "nurse", "disease", "blood", "care"]
    ecommerce = ["shop", "sell", "product", "store", "cart", "payment", "commerce", "trade", "customer", "business", "market", "order", "delivery"]
    education = ["school", "study", "learn", "student", "teacher", "class", "course", "academy", "education", "grade", "exam", "lecture"]
    finance = ["bank", "money", "transaction", "finance", "stock", "invest", "credit", "wallet", "card", "billing", "invoice"]
    scores = {
        "Healthcare": sum(1 for w in healthcare if w in text),
        "E-commerce": sum(1 for w in ecommerce if w in text),
        "Education": sum(1 for w in education if w in text),
        "Finance": sum(1 for w in finance if w in text),
    }
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        return random.choice(["Healthcare", "E-commerce", "Education", "Finance"])
    return best_cat

def get_cso_siblings(concept_label, cso_graph, n=3):
    siblings = set()
    try:
        for parent in cso_graph.predecessors(concept_label):
            for child in cso_graph.successors(parent):
                if child != concept_label:
                    siblings.add(child)
        for child in cso_graph.successors(concept_label):
            siblings.add(child)
    except nx.NetworkXError:
        pass
    siblings = list(siblings)
    random.shuffle(siblings)
    return siblings[:n]

def get_original_keywords(tree_dict, sec_name):
    for child in tree_dict.get("children", []):
        if child.get("schema_class") == sec_name:
            return [leaf.get("label") for leaf in child.get("children", [])]
    return []

def generate_paraphrased_section_text(sec_class, keywords):
    if not keywords:
        return ""
    templates = [
        "Our system implements concepts like {kws}. This is crucial for optimizing our project's workflow.",
        "The proposed approach utilizes {kws} to address the key technical requirements.",
        "We leverage the integration of {kws} to build a scalable and efficient solution.",
        "By incorporating {kws}, we aim to improve performance and overall system design.",
    ]
    kw_str = ", ".join(keywords)
    return random.choice(templates).format(kws=kw_str)

def build_prompt(section_name: str, text: str) -> str:
    base_rules = f"""
You are an expert academic writer in Computer Science.
Your task is to rewrite the following capstone proposal section ("{section_name}") 
to simulate a SOPHISTICATED disguised plagiarism of the original idea.

GLOBAL RULES (apply to ALL sections):
1. Write in the EXACT SAME LANGUAGE as the original (English→English, Vietnamese→Vietnamese).
2. Retain the core scientific idea, research logic, and system purpose.
3. Do NOT output any explanation or preamble. Return ONLY the rewritten text.
4. CRITICAL — Project/System name: The original project may have a name or acronym (e.g., "IQGS", "EduTrack", "MediSys"). 
   You MUST invent a new plausible fake name or acronym for the system (e.g., "HIRE-IQ", "LearnFlow", "HealthHub") 
   and replace the original name consistently throughout the text. Never keep the original system name.
5. Do NOT preserve bullet-point structure if the original uses bullets — 
   convert to numbered lists or prose paragraphs, or use different grouping.
"""

    section_rules = {
        "Context": """
SECTION-SPECIFIC RULES for Context:
6. Completely restructure paragraph organization — merge or split paragraphs differently.
7. Change the narrative angle: if original starts with the problem, start with 
   the industry landscape instead, then arrive at the same problem.
8. Replace all specific statistics or numbers with different plausible values 
   (e.g., "many companies" → "over 60% of enterprises").
9. Use different metaphors and framing while conveying the same motivation.
""",

        "Problem": """
SECTION-SPECIFIC RULES for Problem:
6. Reframe the problem from a different stakeholder perspective 
   (e.g., from HR's view → from job seeker's view, arriving at same conclusion).
7. Change sentence structures completely — no sentence should start with the same word.
8. Replace domain-specific jargon with equivalent academic terminology.
""",

        "Solution": """
SECTION-SPECIFIC RULES for Solution:
6. Keep the same high-level architecture decisions (e.g., if original uses 
   microservices + RAG pipeline, keep that pattern) but describe it differently.
7. Tech Stack Abstraction and Rephrasing: Do NOT just list raw product and tool names. 
   You MUST retain the EXACT technology names (e.g., if it uses "Spring Boot", "PostgreSQL", "RAG"), 
   but embed them deeply within descriptive prose paragraphs explaining their architectural and functional roles 
   (e.g., rewrite as "the framework utilizes Spring Boot to orchestrate microservices, backed by a PostgreSQL infrastructure 
   for persistent storage, integrated with a RAG pipeline"). DO NOT use raw bulleted lists of tool names.
8. Reorganize the bullet points into a different structure 
   (e.g., group by layer: frontend/backend/AI instead of by feature).
9. Change component names while keeping their roles 
   (e.g., "HR Portal" → "Recruiter Dashboard", "Job Seeker Marketplace" → "Candidate Hub").
""",

        "Theory": """
SECTION-SPECIFIC RULES for Theory:
6. Tech Stack Abstraction and Role Description: Do NOT just list raw brand names of technologies 
   separated by slashes or bullets (e.g., "PostgreSQL / Redis / Docker"). 
   You MUST KEEP the exact technology names to preserve technical semantics, BUT you MUST expand and 
   abstract them into complete, complex sentences describing their infrastructure roles 
   (e.g., rewrite as "the system architecture deploys a containerized PostgreSQL relational database management system, 
   which operates alongside a Redis in-memory key-value data store configured for high-performance session caching"). 
   CRITICAL CONSTRAINT: Never swap or change the core technology names, just completely distort the surrounding text and grammatical structure.
7. For document list (SRS, SDD, etc.): rename the documents using alternative 
   standard names (e.g., SRS → Software Requirements Document, 
   SDD → System Design Specification, STD → Test Plan Document).
8. Reorder the technology list: client-side before server-side, or alphabetical.
9. Add one plausible non-core infrastructure technology that the original doesn't mention 
   (e.g., add "Nginx as reverse proxy" or "Prometheus for monitoring").
""",

        "Deliverables": """
SECTION-SPECIFIC RULES for Deliverables:
6. Convert bullet list to a numbered list with brief descriptions added to each item.
7. Merge 2-3 related deliverables into one combined item, or split one into two.
8. Rephrase each deliverable from a user-outcome perspective rather than 
   a technical-output perspective 
   (e.g., "Develop RESTful API" → "A fully operational backend service exposing 
   standardized endpoints to support all platform functionalities").
9. Change the order of deliverables.
""",

        "Timeline": """
SECTION-SPECIFIC RULES for Timeline:
6. CRITICAL — Do NOT preserve the same number of tasks. 
   If original has 7 tasks, output 5 or 6 by merging some.
7. Reorder tasks where logically permissible 
   (e.g., swap Task 2 and Task 3 if they are independent).
8. Rename each task phase completely — do not use "Task 1", "Task 2" pattern 
   if original uses it. Use "Phase 1", "Sprint 1", or "Stage 1" instead.
9. Change the scope boundary of each task 
   (e.g., if original puts "testing" in Task 7, move it to be part of Task 5).
10. Replace technology-specific mentions within tasks with the same prose-embedded style.
""",

        "Methodology": """
SECTION-SPECIFIC RULES for Methodology:
6. If original uses Agile/Scrum → keep Agile but describe it as Kanban or SAFe.
7. Change evaluation metrics names while keeping the same type 
   (e.g., "blinded HR reviewers" → "domain expert panel evaluation").
8. Restructure the methodology flow — describe phases in different order 
   if logically equivalent.
""",
    }

    specific = section_rules.get(section_name, """
SECTION-SPECIFIC RULES:
6. Completely rewrite sentence structures.
7. Replace all specific technology or tool names with equivalents embedded in prose.
8. Change organizational structure of the content.
""")

    return base_rules + specific + f"\nOriginal text:\n{text}"

def paraphrase_via_llm(section_name, text, openai_client, temperature=0.85):
    if not text or not text.strip():
        return ""
    
    prompt = build_prompt(section_name, text)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional academic writing assistant. "
                        "You produce clean, sophisticated academic text. "
                        "You strictly follow all numbered rules in the prompt. "
                        "You NEVER keep the original system/project name. "
                        "You NEVER output explanations — only the rewritten text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API for {section_name}: {e}")
        return text

def build_tree_from_sections(title, sec_texts, idf, cso):
    root = {
        "label": title,
        "schema_class": "root",
        "depth": 1,
        "idf": 1.0,
        "cso_ancestors": [],
        "children": []
    }
    for c in MAPPING:
        text_val = sec_texts.get(c, "")
        if not text_val or str(text_val).strip() == "":
            continue

        section_node = {
            "label": c,
            "schema_class": c,
            "depth": 2,
            "idf": idf[c],
            "cso_ancestors": [],
            "children": []
        }

        keywords = extract_keywords_from_text(text_val, top_k=8)

        for kw, score in keywords:
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
    return root

def generate():
    trees_path = os.path.join("data", "dataset", "trees.json")
    pairs_path = os.path.join("data", "dataset", "pairs.csv")
    topics_path = os.path.join("data", "processed", "topics.json")
    full_texts_path = os.path.join("data", "dataset", "full_texts.json")
    cso_cache_path = os.path.join("data", "processed", "cso_graph.pkl")
    
    if not os.path.exists(trees_path):
        print(f"Error: {trees_path} not found.")
        return
    if not os.path.exists(topics_path):
        print(f"Error: {topics_path} not found.")
        return
        
    with open(trees_path, 'r', encoding='utf-8') as f:
        trees = json.load(f)
    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    cso = CSOLookup()
    cso_graph = cso.graph
    print(f"Loaded CSO graph with {cso_graph.number_of_nodes()} nodes.")
    
    # Filter to original trees only
    original_codes = [k for k in trees.keys() 
                      if not k.endswith("_plag") and not k.endswith("_hard")]
    print(f"Loaded {len(original_codes)} original trees.")
    
    topic_records = {}
    for r in topics:
        code = r.get("topic_code", f"TOPIC_{r['id']}")
        topic_records[code] = r
        
    # Recompute IDF weights on original corpus
    N = len(original_codes)
    df_count = {}
    for c, field_name in MAPPING.items():
        count = 0
        for code in original_codes:
            r = topic_records.get(code, {})
            val = r.get(field_name)
            if val is not None and str(val).strip() != "":
                count += 1
        df_count[c] = count

    import math
    idf = {}
    for c in MAPPING:
        idf[c] = round(math.log((N + 1) / (df_count[c] + 1)) + 0.2, 4)
    print(f"Recomputed IDF: {idf}")
        
    full_texts = {}
    for code in original_codes:
        if code in topic_records:
            r = topic_records[code]
            sec_texts = {}
            for sec, field_name in MAPPING.items():
                val = r.get(field_name)
                sec_texts[sec] = str(val) if val is not None else ""
            full_texts[code] = sec_texts
        else:
            full_texts[code] = {sec: "" for sec in MAPPING}
    
    cat_map = {}
    for code in original_codes:
        tree = trees[code]
        keywords = []
        for sec in tree.get("children", []):
            for leaf in sec.get("children", []):
                keywords.append(leaf.get("label", ""))
        cat_map[code] = categorize_topic(tree.get("label", ""), keywords)
        
    by_cat = {"Healthcare": [], "E-commerce": [], "Education": [], "Finance": []}
    for code, cat in cat_map.items():
        by_cat[cat].append(code)
    print(f"Category counts: { {k: len(v) for k, v in by_cat.items()} }")
    
    # Clean old generated entries
    for key in list(trees.keys()):
        if key.endswith("_plag") or key.endswith("_hard"):
            del trees[key]
    for key in list(full_texts.keys()):
        if key.endswith("_plag") or key.endswith("_hard"):
            del full_texts[key]
    
    pairs = []
    
    # ── Type A: Structural Plagiarism (80 pairs) ──
    print("Generating 80 Type A pairs (structural plagiarism, rewritten text)...")
    type_a_sources = random.sample(original_codes, 80)
    for i, code in enumerate(type_a_sources):
        plag_code = code + "_plag"
        
        # Build mutated text per section
        mutated_text = {}
        for sec_name in MAPPING:
            orig_text = full_texts[code].get(sec_name, "")
            if not orig_text or not orig_text.strip():
                mutated_text[sec_name] = ""
                continue
                
            if client:
                print(f"  [{i+1}/80] Paraphrasing {sec_name} for {code} via GPT-4o-mini...")
                mutated_text[sec_name] = paraphrase_via_llm(sec_name, orig_text, client)
            else:
                # Fallback to rule-based keyword swap
                orig_kws = get_original_keywords(trees[code], sec_name)
                mut_kws = []
                for kw in orig_kws:
                    if random.random() < 0.5:
                        siblings = get_cso_siblings(kw, cso_graph)
                        mut_kws.append(random.choice(siblings) if siblings else kw)
                    else:
                        mut_kws.append(kw)
                mutated_text[sec_name] = generate_paraphrased_section_text(sec_name, mut_kws)
            
        full_texts[plag_code] = mutated_text
        
        # Re-build tree from the mutated text to keep them consistent!
        plag_title = trees[code].get("label", "Untitled") + " (Variant)"
        trees[plag_code] = build_tree_from_sections(plag_title, mutated_text, idf, cso)
        
        pairs.append({"doc_a": code, "doc_b": plag_code, "label": 1, "type": "Type_A"})
        
    # ── Type B: Same-Domain (Real Capstones, No Copy-Paste) (50 pairs) ──
    print("Generating 50 Type B pairs (same domain, realistic separate capstones)...")
    type_b_count = 0
    used_b_pairs = set()
    while type_b_count < 50:
        cat = random.choice([k for k, v in by_cat.items() if len(v) >= 2])
        doc_a, doc_b = random.sample(by_cat[cat], 2)
        pair_key = (min(doc_a, doc_b), max(doc_a, doc_b))
        if pair_key in used_b_pairs:
            continue
        used_b_pairs.add(pair_key)
        
        # No text modification! Just compare doc_a and doc_b directly.
        # Since they are different documents from the same category, they have natural semantic overlap.
        pairs.append({"doc_a": doc_a, "doc_b": doc_b, "label": 0, "type": "Type_B"})
        type_b_count += 1
            
    # ── Type C: Different Domains (50 pairs) ──
    print("Generating 50 Type C pairs (different domains)...")
    type_c_count = 0
    used_c_pairs = set()
    while type_c_count < 50:
        cat_a, cat_b = random.sample(list(by_cat.keys()), 2)
        if not by_cat[cat_a] or not by_cat[cat_b]:
            continue
        doc_a = random.choice(by_cat[cat_a])
        doc_b = random.choice(by_cat[cat_b])
        pair_key = (min(doc_a, doc_b), max(doc_a, doc_b))
        if pair_key in used_c_pairs:
            continue
        used_c_pairs.add(pair_key)
        pairs.append({"doc_a": doc_a, "doc_b": doc_b, "label": 0, "type": "Type_C"})
        type_c_count += 1
            
    # Save everything
    with open(trees_path, 'w', encoding='utf-8') as f:
        json.dump(trees, f, ensure_ascii=False, indent=2)
    with open(full_texts_path, 'w', encoding='utf-8') as f:
        json.dump(full_texts, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame(pairs)
    df.to_csv(pairs_path, index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"  Type A (Structural Plagiarism): {sum(1 for p in pairs if p['type'] == 'Type_A')} pairs")
    print(f"  Type B (Text Overlap Trap):     {sum(1 for p in pairs if p['type'] == 'Type_B')} pairs")
    print(f"  Type C (Different Domain):      {sum(1 for p in pairs if p['type'] == 'Type_C')} pairs")
    print(f"  Total trees: {len(trees)}")

if __name__ == "__main__":
    generate()
