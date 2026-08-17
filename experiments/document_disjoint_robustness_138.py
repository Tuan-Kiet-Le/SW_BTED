"""Document-disjoint robustness audit for the canonical 138-pair benchmark."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
AUDIT = ROOT / "reports" / "audit"
sys.path.insert(0, str(ROOT))

def load_sw():
    spec=importlib.util.spec_from_file_location("sw_disjoint", ROOT/"repro_candidate_138"/"src"/"05_sw_bted.py")
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod

def main():
    rows=list(csv.DictReader((DATA/"pairs.csv").open(encoding="utf-8-sig",newline="")))
    labels=np.array([int(r["label"]) for r in rows])
    score_rows=list(csv.DictReader((AUDIT/"clean_baseline_suite_pair_scores_138.csv").open(encoding="utf-8-sig",newline="")))
    assert [(r["doc_a"],r["doc_b"],r["label"],r["type"]) for r in rows] == [(r["doc_a"],r["doc_b"],r["label"],r["type"]) for r in score_rows]
    sw=load_sw(); trees_raw=json.loads((DATA/"trees_section.json").read_text(encoding="utf-8")); nodes={k:sw.CapstoneNode.from_dict(v) for k,v in trees_raw.items()}
    model=sw.SWCostModel(alpha=.8,beta={"T2":0.0,"T3":.9,"T4":.8},cso_graph=None,max_depth=19)
    scores={"SW-BTED":np.array([sw.normalize_similarity(nodes[r["doc_a"]],nodes[r["doc_b"]],model) for r in rows])}
    for name in ["TF-IDF","Standard TED","pq-Gram","Section Cosine","Genuine Flat Domain SBERT"]: scores[name]=np.array([float(r[name]) for r in score_rows])
    graph=nx.Graph(); graph.add_edges_from((r["doc_a"],r["doc_b"]) for r in rows)
    component={doc:i for i,comp in enumerate(nx.connected_components(graph)) for doc in comp}
    groups=np.array([component[r["doc_a"]] for r in rows])
    splitter=GroupKFold(n_splits=5); results={}
    for name,values in scores.items():
        pred=np.zeros(len(labels),int); folds=[]
        for fold,(train,test) in enumerate(splitter.split(np.zeros(len(labels)),labels,groups),1):
            best=(-1.0,0.0)
            for t in np.arange(0,1.0001,.005):
                v=f1_score(labels[train],values[train]>=t,zero_division=0)
                if v>best[0]: best=(v,round(float(t),3))
            pred[test]=(values[test]>=best[1]).astype(int)
            folds.append({"fold":fold,"train_pairs":len(train),"test_pairs":len(test),"train_components":len(set(groups[train])),"test_components":len(set(groups[test])),"threshold":best[1],"f1":float(f1_score(labels[test],pred[test],zero_division=0)),"precision":float(precision_score(labels[test],pred[test],zero_division=0)),"recall":float(recall_score(labels[test],pred[test],zero_division=0))})
        cm=confusion_matrix(labels,pred,labels=[1,0])
        results[name]={"mean_fold_f1":float(np.mean([x["f1"] for x in folds])),"std_fold_f1":float(np.std([x["f1"] for x in folds])),"pooled_f1":float(f1_score(labels,pred,zero_division=0)),"tp":int(cm[0,0]),"fp":int(cm[1,0]),"tn":int(cm[1,1]),"fn":int(cm[0,1]),"folds":folds}
    out={"protocol":{"pairs":138,"pair_file_sha256":hashlib.sha256((DATA/"pairs.csv").read_bytes()).hexdigest(),"split":"GroupKFold(5) by connected components of pair-document graph","threshold_grid_step":.005,"threshold_train_only":True,"n_document_components":len(set(groups)),"n_documents":graph.number_of_nodes()},"results":results}
    (AUDIT/"document_disjoint_robustness_138.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    lines=["# Document-disjoint robustness audit — canonical 138 pairs","",f"Protocol: GroupKFold(5) by connected components of the pair-document graph; {graph.number_of_nodes()} documents, {len(set(groups))} components. Thresholds use a 0.005 grid selected on training groups only.","","| Method | Mean F1 | Std | Pooled F1 | TP | FP | TN | FN |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items(): lines.append(f"| {name} | {r['mean_fold_f1']:.4f} | {r['std_fold_f1']:.4f} | {r['pooled_f1']:.4f} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |")
    lines += ["","This is a robustness audit, not a replacement for the primary stratified pair-level result. Connected-component grouping avoids placing the same document in both train and test groups."]
    (ROOT/"reports"/"DOCUMENT_DISJOINT_ROBUSTNESS_138.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"protocol":out["protocol"],"results":{k:{x:v[x] for x in ("mean_fold_f1","std_fold_f1","pooled_f1","tp","fp","tn","fn")} for k,v in results.items()}},indent=2))

if __name__=="__main__": main()
