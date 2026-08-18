"""Read-only scope-matched reproduction of the historical RAG main evaluation."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; RAG=Path(r"D:\FPT\Semester_8\RAG_Research")
DATA=RAG/'Data'/'dataset'; REGEN=RAG/'Data'/'processed'/'plag_regen_sections.json'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from src.node import CapstoneNode
import baselines
WEIGHTS={"Context":.10,"Problem":.15,"Solution":.25,"Theory":.15,"Deliverables":.10,"Methodology":.15,"Timeline":.05,"References":.05}

def doc_text(k,full,trees): return baselines.get_document_full_text(k,full,trees[k])
def sec(k,s,full,trees): return baselines.get_document_section_text(k,s,full,trees[k])
def score_tfidf(sub,train_docs,full,trees):
 docs={k:doc_text(k,full,trees) for k in set(sub.doc_a)|set(sub.doc_b)|set(train_docs)}; v=TfidfVectorizer(); v.fit([docs[k] for k in train_docs if k in docs]); out=[]
 for r in sub.itertuples(): out.append(float(cosine_similarity(v.transform([docs[r.doc_a]]),v.transform([docs[r.doc_b]]))[0,0]))
 return np.array(out)
def score_section(sub,train_docs,full,trees):
 vs={}
 for s in WEIGHTS:
  texts=[sec(k,s,full,trees) for k in train_docs]; v=TfidfVectorizer()
  if any(x.strip() for x in texts): v.fit(texts); vs[s]=v
 out=[]
 for r in sub.itertuples():
  total=weight=0
  for s,w in WEIGHTS.items():
   if s not in vs: continue
   a=sec(r.doc_a,s,full,trees).strip(); b=sec(r.doc_b,s,full,trees).strip(); sim=1 if not a and not b else 0 if not a or not b else float(cosine_similarity(vs[s].transform([a]),vs[s].transform([b]))[0,0]); total+=w*sim; weight+=w
  out.append(total/weight if weight else 1)
 return np.array(out)
def main():
 pairs=pd.read_csv(DATA/'pairs.csv'); raw=json.loads((DATA/'trees_section.json').read_text(encoding='utf-8')); trees={k:CapstoneNode.from_dict(v) for k,v in raw.items()}; full=json.loads((DATA/'full_texts.json').read_text(encoding='utf-8')); y=(pairs.type=='Type_A').astype(int).to_numpy(); regen=set(json.loads(REGEN.read_text(encoding='utf-8')).keys()); real=~(pairs.doc_a.isin(regen)|pairs.doc_b.isin(regen)); print('pairs',len(pairs),'real',int(real.sum()),'regen',len(regen))
 results={}; preds={}
 for name,fn in [('TF-IDF',score_tfidf),('Section Cosine',score_section)]:
  oof=np.zeros(len(pairs),int); scores=np.zeros(len(pairs)); folds=[]
  for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(pairs,pairs.type),1):
   inner=StratifiedKFold(4,shuffle=True,random_state=42); itr,iv=next(inner.split(pairs.iloc[tr],pairs.iloc[tr].type)); actual=tr[itr]; val=tr[iv]; docs=set(pairs.iloc[actual].doc_a)|set(pairs.iloc[actual].doc_b); sv=fn(pairs.iloc[val],docs,full,trees); st=fn(pairs.iloc[te],docs,full,trees); best=-1;bt=0
   for t in np.arange(0,1.0001,.01):
    z=f1_score(y[val],sv>=t,zero_division=0)
    if z>best:best=z;bt=round(float(t),3)
   scores[val]=sv; scores[te]=st; oof[te]=(st>=bt).astype(int); folds.append({'fold':fold,'threshold':bt,'f1':float(f1_score(y[te],oof[te],zero_division=0))})
  def summarize(mask):
   cm=confusion_matrix(y[mask],oof[mask],labels=[1,0]); return {'n':int(mask.sum()),'f1':float(f1_score(y[mask],oof[mask],zero_division=0)),'precision':float(precision_score(y[mask],oof[mask],zero_division=0)),'recall':float(recall_score(y[mask],oof[mask],zero_division=0)),'tp_fp_tn_fn':[int(cm[0,0]),int(cm[1,0]),int(cm[1,1]),int(cm[0,1])]}
  results[name]={'full':summarize(np.ones(len(pairs),bool)),'real_only_by_name':summarize(real),'folds':folds}; preds[name]=oof.tolist()
 out={'protocol':{'source':str(DATA),'pairs':180,'real_filter':'exclude regen keys after OOF predictions','outer_seed':42,'inner_seed':42,'threshold_grid':'.01'},'results':results,'predictions':preds}; (ROOT/'reports'/'audit'/'rag_main_scope_180_b1_b5.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2))
if __name__=='__main__':main()
