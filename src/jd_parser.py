import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import spacy
from src.node import CapstoneNode

# Load spaCy for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except Exception:
    nlp = None

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[●\t•*\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def partition_jd_description(text: str) -> dict:
    # Heuristic regex patterns for common section headers in JDs
    pattern = r'(?i)\b(Who you are:|Role:|Qualifications:|Responsibilities:|Duties:|Requirements:|What you will do:|Job Type:|Benefits:|Pay:|Schedule:|Experience:|Work Location:|About the company:|About Us|Company Description|What We Offer|What you bring)\b'
    parts = re.split(pattern, text)
    
    sections = {
        "D1_COMPANY_CONTEXT": [],
        "D2_REQUIREMENTS": [],
        "D3_RESPONSIBILITIES": [],
        "D4_COMPENSATION": []
    }
    
    if len(parts) <= 1:
        sections["D1_COMPANY_CONTEXT"].append(text)
        return {k: " ".join(v).strip() for k, v in sections.items()}
        
    sections["D1_COMPANY_CONTEXT"].append(parts[0])
    
    headers_map = {
        r'about|company|overview|us': 'D1_COMPANY_CONTEXT',
        r'who you are|qualification|requirement|skills|experience|education|bring': 'D2_REQUIREMENTS',
        r'role|responsibility|duties|what you will do': 'D3_RESPONSIBILITIES',
        r'job type|benefits|pay|schedule|compensation|salary|location|offer': 'D4_COMPENSATION'
    }
    
    for i in range(1, len(parts), 2):
        header = parts[i].lower()
        content = parts[i+1] if i+1 < len(parts) else ''
        content = content.strip()
        if not content:
            continue
            
        current_sec = 'D1_COMPANY_CONTEXT'
        for pat, sec in headers_map.items():
            if re.search(pat, header):
                current_sec = sec
                break
        sections[current_sec].append(content)
        
    return {k: " ".join(v).strip() for k, v in sections.items()}

def parse_jd_to_tree(job_id: str, title: str, company_name: str, description: str) -> CapstoneNode:
    root = CapstoneNode(
        label=str(job_id),
        schema_class="Project",
        depth=1,
        feature_label=f"{company_name} - {title}"
    )
    
    sections = partition_jd_description(description)
    
    domain_mapping = {
        "D1_COMPANY_CONTEXT": "D1_BUSINESS_CONTEXT",
        "D2_REQUIREMENTS": "D2_FUNCTIONAL",
        "D3_RESPONSIBILITIES": "D3_TECHNICAL_REALIZATION",
        "D4_COMPENSATION": "D4_EXECUTION_PLANNING"
    }
    
    for domain_name, content in sections.items():
        mapped_name = domain_mapping[domain_name]
        domain_node = CapstoneNode(
            label=mapped_name,
            schema_class=mapped_name,
            depth=2
        )
        root.children.append(domain_node)
        
        if not content:
            continue
            
        # Segment content into sentences
        if nlp:
            doc = nlp(content)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback simple split
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
            
        for sent in sentences:
            cleaned = clean_text(sent)
            if not cleaned or len(cleaned.split()) < 3:
                continue
            intent_node = CapstoneNode(
                label=cleaned[:100],  # short label
                schema_class="IntentMatching",
                depth=3,
                raw_text=cleaned,
                normalized_text=cleaned
            )
            domain_node.children.append(intent_node)
            
    return root
