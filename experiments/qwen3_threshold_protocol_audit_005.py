"""Re-evaluate downloaded Qwen3 scores with the canonical 0.005 grid."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "reports" / "qwen3_pair_prediction_audit.csv").open(encoding="utf-8-sig")))
y = np.array([int(r["label"]) for r in rows])
scores = np.array([float(r["qwen3_cosine"]) for r in rows])
pred = np.zeros(len(y), dtype=int)
folds = []
splitter = StratifiedKFold(5, shuffle=True, random_state=42)
for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
    best_f1, threshold = -1.0, 0.0
    for t in np.arange(0.0, 1.0001, 0.005):
        value = f1_score(y[train], scores[train] >= t, zero_division=0)
        if value > best_f1:
            best_f1, threshold = value, round(float(t), 3)
    pred[test] = (scores[test] >= threshold).astype(int)
    folds.append({"fold": fold, "threshold": threshold,
                  "f1": float(f1_score(y[test], pred[test], zero_division=0)),
                  "precision": float(precision_score(y[test], pred[test], zero_division=0)),
                  "recall": float(recall_score(y[test], pred[test], zero_division=0))})
cm = confusion_matrix(y, pred, labels=[1, 0])
out = {
    "protocol": {"n_pairs": len(y), "cv_seed": 42, "threshold_train_only": True,
                 "threshold_grid_step": 0.005, "pair_key": "doc_a,doc_b,label,type"},
    "pooled": {"f1": float(f1_score(y, pred)), "precision": float(precision_score(y, pred)),
               "recall": float(recall_score(y, pred)), "tp": int(cm[0, 0]),
               "fp": int(cm[1, 0]), "tn": int(cm[1, 1]), "fn": int(cm[0, 1])},
    "mean_fold_f1": float(np.mean([x["f1"] for x in folds])),
    "std_fold_f1": float(np.std([x["f1"] for x in folds])), "folds": folds,
}
(ROOT / "reports" / "qwen3_threshold_protocol_audit_005.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
