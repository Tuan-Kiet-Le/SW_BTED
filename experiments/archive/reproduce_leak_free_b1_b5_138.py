"""Reproduce the manuscript's inner-train-only B1/B5 protocol."""
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
DATA = STAGE / "data" / "dataset"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from src.node import CapstoneNode
import baselines

WEIGHTS = {"Context": .10, "Problem": .15, "Solution": .25, "Theory": .15, "Deliverables": .10, "Methodology": .15, "Timeline": .05, "References": .05}

def text(doc, full, tree): return baselines.get_document_full_text(doc, full, tree)
def sec(doc, name, full, tree): return baselines.get_document_section_text(doc, name, full, tree)

def fold_tfidf(sub, train_docs, full, trees):
    docs = {k: text(k, full, trees[k]) for k in set(sub.doc_a) | set(sub.doc_b) | set(train_docs)}
    vec = TfidfVectorizer(); vec.fit([docs[k] for k in train_docs if k in docs])
    out=[]
    for r in sub.itertuples():
        a,b=vec.transform([docs[r.doc_a],docs[r.doc_b]])
        out.append(float(cosine_similarity(a,b)[0,0]))
    return np.array(out)

def fold_section(sub, train_docs, full, trees):
    vs={}
    for name in WEIGHTS:
        v=TfidfVectorizer(); train=[sec(k,name,full,trees[k]) for k in train_docs]
        if any(x.strip() for x in train): v.fit(train); vs[name]=v
    out=[]
    for r in sub.itertuples():
        total=weight=0.0
        for name,w in WEIGHTS.items():
            a=sec(r.doc_a,name,full,trees[r.doc_a]).strip(); b=sec(r.doc_b,name,full,trees[r.doc_b]).strip()
            if name not in vs: continue
            sim=1.0 if not a and not b else 0.0 if not a or not b else float(cosine_similarity(vs[name].transform([a]),vs[name].transform([b]))[0,0])
            total += w*sim; weight += w
        out.append(total/weight if weight else 1.0)
    return np.array(out)

def main():
    pairs=pd.read_csv(DATA/'pairs.csv'); trees_raw=json.loads((DATA/'trees_section.json').read_text(encoding='utf-8')); full=json.loads((DATA/'full_texts.json').read_text(encoding='utf-8'))
    trees={k:CapstoneNode.from_dict(v) for k,v in trees_raw.items()}; y=(pairs.type=='Type_A').astype(int).to_numpy(); results={}; preds={}
    for name, scorer in [('TF-IDF',fold_tfidf),('Section Cosine',fold_section)]:
        oof=np.zeros(len(pairs),int); folds=[]; thresholds=[]
        outer=StratifiedKFold(5,shuffle=True,random_state=42)
        for fold,(tr,te) in enumerate(outer.split(pairs,y),1):
            inner=StratifiedKFold(4,shuffle=True,random_state=42); inner_tr, val_rel=next(inner.split(pairs.iloc[tr], pairs.iloc[tr].type)); actual=tr[inner_tr]; val=tr[val_rel]; train_docs=set(pairs.iloc[actual].doc_a)|set(pairs.iloc[actual].doc_b)
            val_s=scorer(pairs.iloc[val],train_docs,full,trees); test_s=scorer(pairs.iloc[te],train_docs,full,trees)
            best=-1; bt=0
            for t in np.arange(0,1.0001,.01):
                f=f1_score(y[val],val_s>=t,zero_division=0)
                if f>best: best=f;bt=round(float(t),3)
            oof[te]=(test_s>=bt).astype(int); thresholds.append(bt); folds.append({'fold':fold,'threshold':bt,'f1':float(f1_score(y[te],oof[te],zero_division=0)),'precision':float(precision_score(y[te],oof[te],zero_division=0)),'recall':float(recall_score(y[te],oof[te],zero_division=0))})
        cm=confusion_matrix(y,oof,labels=[1,0]); results[name]={'mean_f1':float(np.mean([x['f1'] for x in folds])),'std_f1':float(np.std([x['f1'] for x in folds])),'mean_precision':float(np.mean([x['precision'] for x in folds])),'mean_recall':float(np.mean([x['recall'] for x in folds])),'tp_fp_tn_fn':[int(cm[0,0]),int(cm[1,0]),int(cm[1,1]),int(cm[0,1])],'thresholds':thresholds,'folds':folds}; preds[name]=oof.tolist()
    out={'protocol':{'outer_seed':42,'inner_seed':42,'inner_train_fraction':'3/4 of outer train','threshold_grid':'.01','fit_corpus':'actual inner-train documents only'},'results':results,'predictions':preds}; (ROOT/'reports'/'audit'/'leak_free_b1_b5_138.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
