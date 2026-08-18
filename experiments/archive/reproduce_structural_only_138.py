"""Reproduce the manuscript's fixed structural-only 138-pair path."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
sys.path.insert(0, str(STAGE))

import importlib

node_mod = importlib.import_module("src.node")
sw_mod = importlib.import_module("src.05_sw_bted")


def main():
    pairs = pd.read_csv(STAGE / "data" / "dataset" / "pairs.csv")
    with (STAGE / "data" / "dataset" / "trees_section.json").open(encoding="utf-8") as handle:
        raw_trees = json.load(handle)
    with (STAGE / "data" / "processed" / "cso_graph.pkl").open("rb") as handle:
        cso = pickle.load(handle)

    # alpha=1.0 removes the global embedding component. beta=None makes the
    # cost model use the configured per-layer beta values T2/T3/T4.
    model = sw_mod.SWCostModel(alpha=1.0, beta=None, cso_graph=cso["graph"], max_depth=cso.get("max_depth", 19))
    scores = []
    for row in pairs.itertuples(index=False):
        a = node_mod.CapstoneNode.from_dict(raw_trees[row.doc_a])
        b = node_mod.CapstoneNode.from_dict(raw_trees[row.doc_b])
        scores.append(sw_mod.normalize_similarity(a, b, model))

    scores = np.asarray(scores, dtype=float)
    labels = (pairs["type"].to_numpy() == "Type_A").astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    predictions = np.zeros(len(labels), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(skf.split(scores, labels), 1):
        best_t, best_f1, best_p = 0.0, -1.0, -1.0
        for threshold in np.arange(0.0, 1.001, 0.01):
            pred = (scores[train_idx] >= threshold).astype(int)
            f1 = f1_score(labels[train_idx], pred, zero_division=0)
            precision = precision_score(labels[train_idx], pred, zero_division=0)
            if (f1, precision, -abs(threshold - 0.34)) > (best_f1, best_p, -abs(best_t - 0.34)):
                best_t, best_f1, best_p = threshold, f1, precision
        test_pred = (scores[test_idx] >= best_t).astype(int)
        predictions[test_idx] = test_pred
        fold_rows.append({
            "fold": fold,
            "threshold": best_t,
            "f1": f1_score(labels[test_idx], test_pred, zero_division=0),
            "precision": precision_score(labels[test_idx], test_pred, zero_division=0),
            "recall": recall_score(labels[test_idx], test_pred, zero_division=0),
        })

    result = {
        "n_pairs": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "cv_f1_mean": float(np.mean([r["f1"] for r in fold_rows])),
        "cv_f1_std": float(np.std([r["f1"] for r in fold_rows])),
        "cv_precision_mean": float(np.mean([r["precision"] for r in fold_rows])),
        "cv_recall_mean": float(np.mean([r["recall"] for r in fold_rows])),
        "global_f1": float(f1_score(labels, predictions, zero_division=0)),
        "global_precision": float(precision_score(labels, predictions, zero_division=0)),
        "global_recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "folds": fold_rows,
    }

    out = STAGE / "results"
    out.mkdir(exist_ok=True)
    with (out / "structural_only_pair_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_a", "doc_b", "label", "type", "score", "prediction"])
        for row, score, prediction in zip(pairs.itertuples(index=False), scores, predictions):
            writer.writerow([row.doc_a, row.doc_b, int(row.type == "Type_A"), row.type, f"{score:.8f}", int(prediction)])
    (out / "structural_only_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
