"""
Task 5: Flat Schema-Weighted Embedding Baseline (O(1) Domain-Level Comparison)
Computes per-domain (D1-D4) SBERT embeddings and takes a weighted average
of domain cosine similarities, WITHOUT any tree alignment, APTED, or insert/delete ops.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOMAIN_SECTIONS = {
    'D1_BUSINESS_CONTEXT':       ['context', 'title', 'vietnamese_title'],
    'D2_FUNCTIONAL':             ['functional_requirements', 'proposed_solutions', 'products'],
    'D3_TECHNICAL_REALIZATION':  ['non_functional_requirements', 'theory'],
    'D4_EXECUTION_PLANNING':     ['proposed_tasks'],
}

# Domain weights analogous to schema importance
DOMAIN_WEIGHTS = {
    'D1_BUSINESS_CONTEXT':       0.15,
    'D2_FUNCTIONAL':             0.35,
    'D3_TECHNICAL_REALIZATION':  0.30,
    'D4_EXECUTION_PLANNING':     0.20,
}

def clean_str(text):
    if not text:
        return ""
    t = str(text).strip()
    return "" if t.lower() in ('nan', 'none', 'null') else t

def get_doc_domain_texts(entry):
    domain_texts = {}
    for d_label, sections in DOMAIN_SECTIONS.items():
        texts = [clean_str(entry.get(s, '')) for s in sections if clean_str(entry.get(s, ''))]
        domain_texts[d_label] = " ".join(texts)
    return domain_texts

def compute_flat_schema_similarity(doc_a_dict, doc_b_dict, model):
    texts_a = get_doc_domain_texts(doc_a_dict)
    texts_b = get_doc_domain_texts(doc_b_dict)
    
    domain_sims = {}
    total_weight = 0.0
    weighted_sim = 0.0
    
    for d_label, weight in DOMAIN_WEIGHTS.items():
        txt_a = texts_a.get(d_label, "")
        txt_b = texts_b.get(d_label, "")
        
        if not txt_a and not txt_b:
            sim = 1.0
        elif not txt_a or not txt_b:
            sim = 0.0
        else:
            emb_a = model.encode([txt_a], show_progress_bar=False)[0].reshape(1, -1)
            emb_b = model.encode([txt_b], show_progress_bar=False)[0].reshape(1, -1)
            sim = float(cosine_similarity(emb_a, emb_b)[0][0])
            
        domain_sims[d_label] = sim
        weighted_sim += weight * sim
        total_weight += weight
        
    final_sim = weighted_sim / total_weight if total_weight > 0 else 1.0
    return round(final_sim, 4), domain_sims

if __name__ == '__main__':
    # Load all compiled documents
    su26 = json.load(open('data/processed/topics.json', encoding='utf-8'))
    sp26 = json.load(open('data/processed/topics_sp26.json', encoding='utf-8'))
    t6   = json.load(open('data/dataset/trees_t6_unnormalized.json', encoding='utf-8'))
    regen = json.load(open('data/processed/plag_regen_sections.json', encoding='utf-8'))

    all_docs = {}
    for entry in su26 + sp26:
        doc_id = entry.get('topic_code', entry.get('id', ''))
        if doc_id:
            all_docs[doc_id] = entry
            
    # Extract plag entries from t6 + regen
    DOMAIN_TO_SECTION = {
        'D1_BUSINESS_CONTEXT':       'context',
        'D2_FUNCTIONAL':             'functional_requirements',
        'D3_TECHNICAL_REALIZATION':  'theory',
        'D4_EXECUTION_PLANNING':     'proposed_tasks',
    }
    DOMAIN_TO_REGEN_KEY = {
        'D3_TECHNICAL_REALIZATION': 'theory',
        'D4_EXECUTION_PLANNING':    'proposed_tasks',
    }
    
    for key in t6:
        if key not in all_docs:
            secs = {}
            for child in t6[key].get('children', []):
                sc = child.get('schema_class', '') or child.get('schema', '')
                label = child.get('label', '')
                for d_label, sec_name in DOMAIN_TO_SECTION.items():
                    if label == d_label or sc == d_label:
                        texts = []
                        def collect(n, texts=texts):
                            if n.get('schema_class') == 'AtomicReq' or n.get('schema') == 'AtomicReq':
                                t = n.get('normalized_text', '') or n.get('raw_text', '')
                                if clean_str(t):
                                    texts.append(clean_str(t))
                            for c in n.get('children', []):
                                collect(c, texts)
                        collect(child)
                        if texts:
                            secs[sec_name] = ' '.join(texts)
            if key in regen:
                for d_label, regen_text in regen[key].items():
                    sec_name = DOMAIN_TO_REGEN_KEY.get(d_label)
                    if sec_name and sec_name not in secs and clean_str(regen_text):
                        secs[sec_name] = clean_str(regen_text)
            entry = {'topic_code': key}
            for field in ['context', 'title', 'vietnamese_title',
                          'functional_requirements', 'proposed_solutions', 'products',
                          'non_functional_requirements', 'theory', 'proposed_tasks']:
                entry[field] = secs.get(field, '')
            all_docs[key] = entry

    pairs = pd.read_csv('data/dataset/pairs.csv')
    print("Loading SBERT model for flat baseline...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Evaluating Flat Schema-Weighted Baseline on {len(pairs)} pairs...")
    flat_sims = []
    for idx, row in pairs.iterrows():
        doc_a = all_docs.get(row.doc_a, {})
        doc_b = all_docs.get(row.doc_b, {})
        sim, _ = compute_flat_schema_similarity(doc_a, doc_b, model)
        flat_sims.append(sim)
        
    flat_sims = np.array(flat_sims)
    labels = (pairs['type'] == 'Type_A').astype(int).values

    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(pairs, labels)):
        train_y, test_y = labels[train_idx], labels[test_idx]
        train_sims, test_sims = flat_sims[train_idx], flat_sims[test_idx]
        
        best_t = 0.5
        best_train_f1 = -1
        for t in np.arange(0.0, 1.01, 0.01):
            preds = (train_sims >= t).astype(int)
            f1 = f1_score(train_y, preds, zero_division=0)
            if f1 > best_train_f1:
                best_train_f1 = f1
                best_t = t
                
        test_preds = (test_sims >= best_t).astype(int)
        f1 = f1_score(test_y, test_preds, zero_division=0)
        fold_f1s.append(f1)

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    
    print(f"\nFlat Schema-Weighted Baseline F1: {mean_f1:.4f} (±{std_f1:.4f})")

    # Save outputs
    os.makedirs('results/novelty_test', exist_ok=True)
    res_df = pd.DataFrame([{
        'method': 'Flat Schema-Weighted Baseline (O(1))',
        'f1_mean': round(mean_f1, 4),
        'f1_std': round(std_f1, 4),
        'sw_bted_f1_mean': 0.9697,
        'sw_bted_f1_std': 0.0272
    }])
    res_df.to_csv('results/novelty_test/flat_baseline_comparison.csv', index=False)
    print("Saved to results/novelty_test/flat_baseline_comparison.csv")
