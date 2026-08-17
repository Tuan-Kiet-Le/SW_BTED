"""Reproduce and diff the legacy tree-label lexical baselines vs clean full-text baselines."""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.node import CapstoneNode
import src.baselines as clean_base
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"

def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_baselines", r"D:\FPT\Semester_8\RAG_Research\src\baselines.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod

def evaluate(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels); pred = np.zeros(len(labels), int); folds=[]
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels), 1):
        best=(-1.0, 0.0)
        for t in np.arange(0, 1.0001, .005):
            v=f1_score(labels[train], scores[train]>=t, zero_division=0)
            if v > best[0]: best=(v, round(float(t),3))
        pred[test]=(scores[test]>=best[1]).astype(int)
        folds.append(float(f1_score(labels[test], pred[test], zero_division=0)))
    cm=confusion_matrix(labels,pred,labels=[1,0])
    return {"mean_f1":float(np.mean(folds)),"std_f1":float(np.std(folds)),"tp":int(cm[0,0]),"fp":int(cm[1,0]),"tn":int(cm[1,1]),"fn":int(cm[0,1]),"predictions":pred.tolist()}

def main():
    rows=list(csv.DictReader((DATA/"pairs.csv").open(encoding="utf-8-sig",newline=""))); frame=pd.DataFrame(rows); labels=np.array([int(r["label"]) for r in rows])
    raw=json.loads((DATA/"trees_section.json").read_text(encoding="utf-8")); trees={k:CapstoneNode.from_dict(v) for k,v in raw.items()}; full=json.loads((DATA/"full_texts.json").read_text(encoding="utf-8")); legacy=load_legacy()
    score_sets={
        "legacy_tree_label_tfidf":np.array(legacy.get_cosine_tfidf_similarity(trees,frame)),
        "clean_full_text_tfidf":np.array(clean_base.get_cosine_tfidf_similarity(trees,frame,full)),
        "legacy_tree_label_section":np.array(legacy.get_section_cosine_similarity(trees,frame)),
        "clean_full_text_section":np.array(clean_base.get_section_cosine_similarity(trees,frame,full)),
    }
    metrics={name:evaluate(scores,labels) for name,scores in score_sets.items()}
    out_rows=[]
    for i,r in enumerate(rows):
        out_rows.append({"index":i,"doc_a":r["doc_a"],"doc_b":r["doc_b"],"label":r["label"],**{name:float(scores[i]) for name,scores in score_sets.items()}})
    pd.DataFrame(out_rows).to_csv(OUT/"legacy_vs_clean_lexical_scores_138.csv",index=False)
    summary={"scope":"canonical 138 name-matched pairs","source_diff":{"legacy":"RAG_Research/src/baselines.py called without full_texts; tree_to_full_text() uses tree labels/leaves","clean":"SW_BTED_v2/src/baselines.py called with canonical full_texts.json; full document fields used"},"metrics":{name:{k:v for k,v in value.items() if k!="predictions"} for name,value in metrics.items()},"score_distributions":{name:{"positive_mean":float(scores[labels==1].mean()),"negative_mean":float(scores[labels==0].mean()),"min":float(scores.min()),"max":float(scores.max())} for name,scores in score_sets.items()},"mean_abs_diff":{"tfidf":float(np.mean(np.abs(score_sets["legacy_tree_label_tfidf"]-score_sets["clean_full_text_tfidf"]))),"section":float(np.mean(np.abs(score_sets["legacy_tree_label_section"]-score_sets["clean_full_text_section"])))} }
    (OUT/"LEGACY_VS_CLEAN_LEXICAL_RECONCILIATION_138.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
