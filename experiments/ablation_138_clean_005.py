"""Protocol-aligned beta ablation using the frozen 0.005 threshold grid."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import importlib
base=importlib.import_module("experiments.ablation_138_clean")
sw=importlib.import_module("src.05_sw_bted")

def evaluate(scores, labels):
    f1s=[]; ps=[]; rs=[]; thresholds=[]
    for train,test in StratifiedKFold(5,shuffle=True,random_state=42).split(scores,labels):
        best=(-1.0,0.0)
        for t in np.arange(0,1.0001,.005):
            v=f1_score(labels[train],scores[train]>=t,zero_division=0)
            if v>best[0]: best=(v,round(float(t),3))
        pred=(scores[test]>=best[1]).astype(int); thresholds.append(best[1])
        f1s.append(f1_score(labels[test],pred,zero_division=0)); ps.append(precision_score(labels[test],pred,zero_division=0)); rs.append(recall_score(labels[test],pred,zero_division=0))
    return {"f1_mean":float(np.mean(f1s)),"f1_std":float(np.std(f1s)),"precision_mean":float(np.mean(ps)),"recall_mean":float(np.mean(rs)),"fold_thresholds":thresholds}

def main():
    pairs,labels,trees=base.load_data(); out={"n_pairs":len(pairs),"positive":int(labels.sum()),"negative":int(len(labels)-labels.sum()),"threshold_grid_step":.005,"schedules":{}}
    for name,beta in base.SCHEDULES.items():
        model=sw.SWCostModel(alpha=1.0,beta=beta,cso_graph=None,max_depth=19)
        scores=np.array([sw.normalize_similarity(trees[r.doc_a],trees[r.doc_b],model) for r in pairs.itertuples()])
        out["schedules"][name]={"beta":beta,**evaluate(scores,labels),"score_mean":float(scores.mean())}
    path=ROOT/"reports"/"ablation_138_clean_005.json"; path.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
