import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import docx
import spacy
from typing import Dict, List, Any
import importlib

# Lazy load node definitions
from src.node import CapstoneNode

# Load Tech Equivalence Map
try:
    tech_module = importlib.import_module("src.tech_equivalence")
    TECH_GROUPS = tech_module.TECH_GROUPS
except Exception as e:
    TECH_GROUPS = {}

# Constants
ISO_25010 = {
    "Performance": ["performance", "latency", "response time", "throughput", "speed", "concurrent", "bandwidth"],
    "Security": ["security", "privacy", "encrypt", "auth", "oauth", "jwt", "access control", "safety", "vulnerability"],
    "Usability": ["usability", "interface", "ux", "ui", "easy to use", "user-friendly", "design", "experience"],
    "Reliability": ["reliability", "stability", "availability", "fault", "recover", "backup", "robustness"],
    "Maintainability": ["maintainability", "modular", "reusability", "analyze", "modify", "testability"],
    "Compatibility": ["compatibility", "interoperability", "co-existence", "browser", "os", "platform"],
    "Scalability": ["scalability", "horizontal", "vertical", "growth", "scale", "load"],
}

METHODOLOGY_KEYWORDS = {
    "agile", "scrum", "kanban", "waterfall", "uml", "modeling", "mvc", "microservice", 
    "architect", "design pattern", "framework design", "spiral", "devops", "iterative", 
    "software life cycle", "sdlc", "testing strategy", "deployment guide", "user requirements", 
    "software requirements", "system architecture", "usecase", "sequence diagram", "class diagram"
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[●\t•*\-]', ' ', text)
    text = re.sub(r'(?:^|(?<=\s))o\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_tech_category(tech_name: str) -> str:
    canonical = TECH_GROUPS.get(tech_name.lower().strip(), tech_name.lower().strip())
    # Heuristics based on canonical names in tech_equivalence
    c_lower = canonical.lower()
    if any(k in c_lower for k in ["backend", "fastapi", "spring", "django", "flask", "express", "nestjs", "laravel", "rails", "dotnet"]):
        return "Backend"
    if any(k in c_lower for k in ["frontend", "react", "vue", "angular", "svelte", "solid", "ssr", "css", "tailwind", "bootstrap", "mui"]):
        return "Frontend"
    if any(k in c_lower for k in ["db", "rdbms", "database", "postgres", "mysql", "sqlite", "oracle", "mongodb", "firestore", "nosql"]):
        return "Database"
    if any(k in c_lower for k in ["cloud", "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "cicd", "actions", "nginx", "proxy", "container"]):
        return "Cloud"
    if any(k in c_lower for k in ["ml", "llm", "vector", "ai", "langchain", "tensorflow", "pytorch", "openai", "gpt", "gemini", "llama"]):
        return "AI"
    return "Other"

def is_group_header(line: str) -> bool:
    line_clean = line.strip()
    if not line_clean:
        return False
    words = line_clean.split()
    if len(words) > 8:
        return False
    if line_clean.endswith(":"):
        return True
    if re.match(r'^(?:\d+|[a-zA-Z])[\.\)]', line_clean):
        return True
    return False

def parse_docx_file(file_path: str) -> Dict[str, Any]:
    doc = docx.Document(file_path)
    sections = {
        "english_title": "",
        "vietnamese_title": "",
        "context": [],
        "functional_requirement": [],
        "nonfunctional_requirement": [],
        "proposed_solutions": [],
        "applied_theory": [],
        "products": [],
        "proposed_tasks": []
    }
    
    current_sec = None
    
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
            
        lower_text = text.lower()
        
        # Heading detection
        if "3.1. capstone project name" in lower_text or "capstone project name" in lower_text:
            current_sec = "title"
            continue
        elif lower_text.startswith("a. context") or lower_text.startswith("context"):
            current_sec = "context"
            continue
        elif "non-functional requirement" in lower_text or "nonfunctional requirement" in lower_text:
            current_sec = "nonfunctional_requirement"
            continue
        elif "functional requirement" in lower_text:
            current_sec = "functional_requirement"
            continue
        elif "proposed solutions" in lower_text or "proposed solution" in lower_text:
            current_sec = "proposed_solutions"
            continue
        elif "applied theory" in lower_text or "applied theories" in lower_text:
            current_sec = "applied_theory"
            continue
        elif "products" in lower_text or "deliverables" in lower_text or "expected deliverables" in lower_text:
            current_sec = "products"
            continue
        elif "proposed tasks" in lower_text or "proposed task" in lower_text or "timeline" in lower_text:
            current_sec = "proposed_tasks"
            continue
        elif lower_text.startswith("4. other comments") or lower_text.startswith("4. other"):
            current_sec = None
            continue
            
        if current_sec == "title":
            if lower_text.startswith("english:"):
                sections["english_title"] = text[len("English:"):].strip()
            elif lower_text.startswith("vietnamese:"):
                sections["vietnamese_title"] = text[len("Vietnamese:"):].strip()
            else:
                if not sections["english_title"]:
                    sections["english_title"] = text
        elif current_sec is not None:
            sections[current_sec].append(text)
            
    return sections

def split_clauses(text: str, nlp: spacy.Language) -> List[str]:
    # If the text is very long, do not run spaCy nlp on it, just return simple splits
    if len(text.split()) > 80:
        return [p.strip() for p in text.split(";") if p.strip()]
        
    # Bullet points might contain compound sentences linked by ';' or 'and'
    try:
        doc = nlp(text)
    except Exception:
        return [text]
        
    clauses = []
    
    # Split by semicolon first
    semicolon_parts = [p.strip() for p in text.split(";") if p.strip()]
    
    for part in semicolon_parts:
        if len(part.split()) > 80:
            clauses.append(part)
            continue
            
        try:
            part_doc = nlp(part)
        except Exception:
            clauses.append(part)
            continue
            
        # Find if 'and' connects two verb predicates
        split_points = []
        for token in part_doc:
            if token.text.lower() == 'and' and token.dep_ == 'cc':
                head = token.head
                # Check if head is a verb and has children that are conjoined verbs
                if head.pos_ == 'VERB':
                    conjs = [child for child in head.children if child.dep_ == 'conj' and child.pos_ == 'VERB']
                    if conjs:
                        split_points.append(token.i)
                        
        if split_points:
            # Simple word-based split at coordinating conjunction index
            words = [t.text for t in part_doc]
            # Split at the first CC that connects verbs
            cc_idx = split_points[0]
            clause1 = " ".join(words[:cc_idx]).strip()
            # Find subject of head verb to copy to clause 2
            subj = ""
            head_verb = part_doc[cc_idx].head
            for child in head_verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = " ".join([t.text for t in child.subtree]) + " "
                    break
            clause2 = subj + " ".join(words[cc_idx+1:]).strip()
            
            clauses.append(clause1)
            clauses.append(clause2)
        else:
            clauses.append(part)
            
    return [c for c in clauses if len(c.split()) > 1]


        
def resolve_elide(clause: str, actor_name: str, nlp: spacy.Language) -> str:
    clause_clean = clause.strip()
    if not clause_clean:
        return clause
        
    lower_clause = clause_clean.lower()
    lower_actor = actor_name.lower()
    
    # Check if the clause already starts with the actor or system indicators
    if (lower_clause.startswith(lower_actor) or 
        lower_clause.startswith("the system") or 
        lower_clause.startswith("the platform") or 
        lower_clause.startswith("system ") or
        lower_clause.startswith("user ")):
        return clause_clean
        
    return f"{actor_name} {clause_clean}"

def build_initial_tree(project_id: str, sections: Dict[str, List[str]], nlp: spacy.Language) -> CapstoneNode:
    root = CapstoneNode(
        label=project_id,
        schema_class="MacroFilter",
        depth=1
    )
    
    # ── T2 Domain 1: Business Context ──
    d1_node = CapstoneNode(
        label="D1_BUSINESS_CONTEXT",
        schema_class="D1_BUSINESS_CONTEXT",
        depth=2
    )
    root.children.append(d1_node)
    
    # Context, English Title, Vietnamese Title go straight to T3 under D1
    title_text = sections.get("english_title", "")
    if title_text:
        d1_node.children.append(CapstoneNode(
            label=title_text,
            schema_class="IntentMatching",
            depth=3,
            raw_text=title_text,
            normalized_text=title_text,
            feature_label="Title"
        ))
        
    vn_title = sections.get("vietnamese_title", "")
    if vn_title:
        d1_node.children.append(CapstoneNode(
            label=vn_title,
            schema_class="IntentMatching",
            depth=3,
            raw_text=vn_title,
            normalized_text=vn_title,
            feature_label="Title_VN"
        ))
        
    context_paras = sections.get("context", [])
    for para in context_paras:
        cleaned = clean_text(para)
        if len(cleaned.split()) > 3:
            d1_node.children.append(CapstoneNode(
                label=para,
                schema_class="IntentMatching",
                depth=3,
                raw_text=para,
                normalized_text=cleaned,
                feature_label="Context"
            ))
            
    # ── T2 Domain 2: Functional ──
    d2_node = CapstoneNode(
        label="D2_FUNCTIONAL",
        schema_class="D2_FUNCTIONAL",
        depth=2
    )
    root.children.append(d2_node)
    
    func_paras = sections.get("functional_requirement", [])
    current_actors = None
    
    for para in func_paras:
        para_clean = clean_text(para)
        if not para_clean:
            continue
            
        if is_group_header(para):
            group_name = para_clean.rstrip(":")
            current_actors = [a.strip() for a in re.split(r'\band\b|,', group_name) if a.strip()]
        else:
            clauses = split_clauses(para_clean, nlp)
            for clause in clauses:
                raw_clause = clause
                if current_actors:
                    for actor in current_actors:
                        resolved_clause = resolve_elide(clause, actor, nlp)
                        d2_node.children.append(CapstoneNode(
                            label=raw_clause,
                            schema_class="IntentMatching",
                            depth=3,
                            raw_text=raw_clause,
                            normalized_text=resolved_clause,
                            feature_label=actor
                        ))
                else:
                    d2_node.children.append(CapstoneNode(
                        label=raw_clause,
                        schema_class="IntentMatching",
                        depth=3,
                        raw_text=raw_clause,
                        normalized_text=clause,
                        feature_label="General"
                    ))
                    
    sol_paras = sections.get("proposed_solutions", [])
    for para in sol_paras:
        para_clean = clean_text(para)
        if not para_clean or is_group_header(para):
            continue
        d2_node.children.append(CapstoneNode(
            label=para,
            schema_class="IntentMatching",
            depth=3,
            raw_text=para,
            normalized_text=para_clean,
            feature_label="ProposedSolution"
        ))
        
    prod_paras = sections.get("products", [])
    for para in prod_paras:
        para_clean = clean_text(para)
        if not para_clean or is_group_header(para):
            continue
        d2_node.children.append(CapstoneNode(
            label=para,
            schema_class="IntentMatching",
            depth=3,
            raw_text=para,
            normalized_text=para_clean,
            feature_label="Product"
        ))
        
    # ── T2 Domain 3: Technical Realization ──
    d3_node = CapstoneNode(
        label="D3_TECHNICAL_REALIZATION",
        schema_class="D3_TECHNICAL_REALIZATION",
        depth=2
    )
    root.children.append(d3_node)
    
    nfr_paras = sections.get("nonfunctional_requirement", [])
    for para in nfr_paras:
        para_clean = clean_text(para)
        if not para_clean:
            continue
            
        qa_category = "Other"
        for qa, keywords in ISO_25010.items():
            if any(k in para_clean.lower() for k in keywords):
                qa_category = qa
                break
                
        cleaned_req = para_clean
        for qa in ISO_25010:
            if cleaned_req.lower().startswith(qa.lower()):
                cleaned_req = re.sub(r'^' + qa + r'[\s\-\*:]+', '', cleaned_req, flags=re.IGNORECASE).strip()
                break
                
        d3_node.children.append(CapstoneNode(
            label=para,
            schema_class="IntentMatching",
            depth=3,
            raw_text=para,
            normalized_text=cleaned_req,
            feature_label=qa_category
        ))
        
    theory_paras = sections.get("applied_theory", [])
    for para in theory_paras:
        para_clean = clean_text(para)
        if not para_clean:
            continue
            
        is_methodology = any(k in para_clean.lower() for k in METHODOLOGY_KEYWORDS)
        
        if is_methodology:
            d3_node.children.append(CapstoneNode(
                label=para,
                schema_class="IntentMatching",
                depth=3,
                raw_text=para,
                normalized_text=para_clean,
                feature_label="Methodology"
            ))
        else:
            tech_cat = get_tech_category(para_clean)
            d3_node.children.append(CapstoneNode(
                label=para,
                schema_class="IntentMatching",
                depth=3,
                raw_text=para,
                normalized_text=para_clean,
                feature_label=tech_cat
            ))
            
    # ── T2 Domain 4: Execution Planning ──
    d4_node = CapstoneNode(
        label="D4_EXECUTION_PLANNING",
        schema_class="D4_EXECUTION_PLANNING",
        depth=2
    )
    root.children.append(d4_node)
    
    task_paras = sections.get("proposed_tasks", [])
    for para in task_paras:
        para_clean = clean_text(para)
        if not para_clean or is_group_header(para):
            continue
        d4_node.children.append(CapstoneNode(
            label=para,
            schema_class="IntentMatching",
            depth=3,
            raw_text=para,
            normalized_text=para_clean,
            feature_label="Task"
        ))
        
    return root

def main():
    print("Running Docx Parser Stage 1...")
    samples_dir = "samples"
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory not found at {samples_dir}")
        return
        
    files = [f for f in os.listdir(samples_dir) if f.endswith(".docx")]
    print(f"Found {len(files)} docx files to parse.")
    
    print("Loading spaCy model en_core_web_sm...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading en_core_web_sm ({e}). Falling back to en_core_web_trf if available.")
        try:
            nlp = spacy.load("en_core_web_trf")
        except Exception:
            print("Fatal: Could not load any spaCy model. Please verify installation.")
            return

    parsed_topics = []
    trees_t3 = {}
    
    for f_name in files:
        file_path = os.path.join(samples_dir, f_name)
        project_id = f_name.split("_")[0]
        
        print(f"Parsing {project_id} from {f_name}...")
        sections = parse_docx_file(file_path)
        
        topic_dict = {
            "topic_code": project_id,
            "title": sections["english_title"],
            "vietnamese_title": sections["vietnamese_title"],
            "context": " ".join(sections["context"]),
            "functional_requirements": " ".join(sections["functional_requirement"]),
            "non_functional_requirements": " ".join(sections["nonfunctional_requirement"]),
            "theory": " ".join(sections["applied_theory"]),
            "products": " ".join(sections["products"]),
            "proposed_tasks": " ".join(sections["proposed_tasks"]),
            "proposed_solutions": " ".join(sections["proposed_solutions"])
        }
        parsed_topics.append(topic_dict)
        
        tree_root = build_initial_tree(project_id, sections, nlp)
        trees_t3[project_id] = tree_root.to_dict()

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/dataset", exist_ok=True)
    
    with open("data/processed/topics.json", "w", encoding="utf-8") as f:
        json.dump(parsed_topics, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(parsed_topics)} topics to data/processed/topics.json")
    
    with open("data/dataset/trees_t3.json", "w", encoding="utf-8") as f:
        json.dump(trees_t3, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(trees_t3)} trees up to T3 to data/dataset/trees_t3.json")
    
    print("Stage 1 Parser complete.")

if __name__ == "__main__":
    main()
