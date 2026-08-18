"""Fold-safe alpha ablation using cached full-document MiniLM embeddings."""
import json
import sys
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "repro_candidate_138" / "data"
sw = importlib.import_module("src.05_sw_bted")
MODEL = r"C:\Users\DuyTuanPC\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETA = {"T2": 0.0, "T3": 0.9, "T4": 0.8}

def main():
    raw = json.loads((DATA / "dataset" / "trees_section.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "dataset" / "full_texts.json").read_text(encoding="utf-8"))
    regen = set(json.loads((DATA / "processed" / "plag_regen_sections.json").read_text(encoding="utf-8")).keys())
    pairs = pd.read_csv(DATA / "dataset" / "pairs.csv")
    pairs = pairs[~(pairs.doc_a.isin(regen) | pairs.doc_b.isin(regen))].reset_index(drop=True)
    labels = (pairs.type == "Type_A").astype(int).to_numpy()
    keys = sorted(set(pairs.doc_a) | set(pairs.doc_b))
    texts = [raw[k].get("label", k) + " " + " ".join(str(v) for v in full.get(k, {}).values() if v) for k in keys]
    model_embed = SentenceTransformer(MODEL)
    embs = model_embed.encode(texts, show_progress_bar=False)
    emb_map = {k: e.tolist() for k, e in zip(keys, embs)}
    trees = {k: sw.CapstoneNode.from_dict(v) for k, v in raw.items()}
    for k in keys: trees[k].embedding = emb_map[k]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    output = {"n_pairs": len(pairs), "positive": int(labels.sum()), "alphas": {}}
    for alpha in ALPHAS:
        cost = sw.SWCostModel(alpha=alpha, beta=BETA, cso_graph=None, max_depth=19)
        scores = np.array([sw.normalize_similarity(trees[r.doc_a], trees[r.doc_b], cost) for r in pairs.itertuples()])
        f1s=[]; ps=[]; rs=[]; ts=[]
        for train,test in skf.split(scores,labels):
            best_t=0.5; best=-1
            for t in np.arange(0,1.001,0.01):
                val=f1_score(labels[train],(scores[train]>=t).astype(int),zero_division=0)
                if val>best: best,val_t=val,float(t); best_t=val_t
            pred=(scores[test]>=best_t).astype(int)
            f1s.append(f1_score(labels[test],pred,zero_division=0)); ps.append(precision_score(labels[test],pred,zero_division=0)); rs.append(recall_score(labels[test],pred,zero_division=0)); ts.append(best_t)
        output["alphas"][str(alpha)]={"f1_mean":float(np.mean(f1s)),"f1_std":float(np.std(f1s)),"precision_mean":float(np.mean(ps)),"recall_mean":float(np.mean(rs)),"fold_thresholds":ts}
        print(f"alpha={alpha:.1f}: F1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}; P={np.mean(ps):.4f}; R={np.mean(rs):.4f}; thresholds={ts}")
    out=ROOT/"reports"/"ablation_alpha_138.json"; out.write_text(json.dumps(output,indent=2),encoding="utf-8"); print(f"Wrote {out}")
if __name__ == "__main__": main()
