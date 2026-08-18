"""Evaluate clean 138-pair embedding vectors with the historical CV protocol."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "audit" / "clean_raw_embedding_vectors_138.json"
OUT_JSON = ROOT / "reports" / "audit" / "clean_embedding_evaluation_138.json"
OUT_MD = ROOT / "reports" / "CLEAN_EMBEDDING_EVALUATION_138.md"


def evaluate(rows, column):
    y = np.array([int(row["label"]) for row in rows])
    scores = np.array([float(row[column]) for row in rows])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    oof = np.zeros(len(rows), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(rows)), y), 1):
        best_t = 0.0
        best_f1 = -1.0
        for t in np.arange(0.0, 1.0001, 0.005):
            value = f1_score(y[train], scores[train] >= t, zero_division=0)
            if value > best_f1:
                best_f1 = value
                best_t = round(float(t), 3)
        pred = (scores[test] >= best_t).astype(int)
        oof[test] = pred
        folds.append({
            "fold": fold,
            "threshold": best_t,
            "f1": float(f1_score(y[test], pred, zero_division=0)),
            "precision": float(precision_score(y[test], pred, zero_division=0)),
            "recall": float(recall_score(y[test], pred, zero_division=0)),
            "n_test": int(len(test)),
        })
    cm = confusion_matrix(y, oof, labels=[1, 0])
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = None
    return {
        "model": column,
        "n": len(rows),
        "folds": folds,
        "mean_f1": float(np.mean([x["f1"] for x in folds])),
        "std_f1": float(np.std([x["f1"] for x in folds])),
        "mean_precision": float(np.mean([x["precision"] for x in folds])),
        "mean_recall": float(np.mean([x["recall"] for x in folds])),
        "pooled_oof_f1": float(f1_score(y, oof, zero_division=0)),
        "pooled_oof_precision": float(precision_score(y, oof, zero_division=0)),
        "pooled_oof_recall": float(recall_score(y, oof, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix_order_TP_FP_TN_FN": [
            int(cm[0, 0]), int(cm[1, 0]), int(cm[1, 1]), int(cm[0, 1])
        ],
        "errors": [
            {"index": int(i), "doc_a": rows[i]["doc_a"], "doc_b": rows[i]["doc_b"],
             "label": int(y[i]), "score": float(scores[i]), "prediction": int(oof[i])}
            for i in range(len(rows)) if int(y[i]) != int(oof[i])
        ],
    }


def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = payload["pairs"]
    results = [evaluate(rows, name) for name in ("SBERT_MiniLM", "BGE_Small_v1.5", "MPNet_Base_v2")]
    output = {
        "input": str(INPUT),
        "protocol": {"folds": 5, "shuffle": True, "random_state": 42, "threshold_grid_step": 0.005,
                     "threshold_selection": "train fold only"},
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# Clean embedding evaluation — canonical 138 pairs", "",
        "Protocol: 5-fold StratifiedKFold, shuffle=True, random_state=42; threshold grid 0.005; train-fold-only threshold selection.", "",
        "| Model | Mean F1 | Std | Precision | Recall | ROC-AUC | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        tp, fp, tn, fn = r["confusion_matrix_order_TP_FP_TN_FN"]
        lines.append(f"| {r['model']} | {r['mean_f1']:.4f} | {r['std_f1']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {r['roc_auc']:.4f} | {tp} | {fp} | {tn} | {fn} |")
        lines.append("")
        lines.append(f"**{r['model']} fold thresholds:** " + ", ".join(str(x["threshold"]) for x in r["folds"]))
        if r["errors"]:
            lines.append("Errors: " + "; ".join(f"{x['index']} {x['doc_a']}–{x['doc_b']} (score={x['score']:.6f}, label={x['label']}, pred={x['prediction']})" for x in r["errors"]))
        else:
            lines.append("Errors: none")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    for r in results:
        print(r["model"], f"F1={r['mean_f1']:.6f} +/- {r['std_f1']:.6f}",
              f"P={r['mean_precision']:.6f}", f"R={r['mean_recall']:.6f}",
              "CM=" + str(r["confusion_matrix_order_TP_FP_TN_FN"]),
              "thresholds=" + str([x["threshold"] for x in r["folds"]]))


if __name__ == "__main__":
    main()
