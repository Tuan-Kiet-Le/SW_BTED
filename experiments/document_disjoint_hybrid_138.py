"""Document-disjoint audit for the canonical four-layer hybrid configuration."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
AUDIT = ROOT / "reports" / "audit"
sys.path.insert(0, str(ROOT))


def load_sw():
    spec = importlib.util.spec_from_file_location("sw_hybrid_disjoint", ROOT / "repro_candidate_138" / "src" / "05_sw_bted.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def evaluate(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    splitter = GroupKFold(n_splits=5)
    pred = np.zeros(len(labels), dtype=int)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), 1):
        best_f1, best_t = -1.0, 0.0
        for threshold in np.arange(0.0, 1.0001, 0.005):
            value = f1_score(labels[train], scores[train] >= threshold, zero_division=0)
            if value > best_f1:
                best_f1, best_t = value, round(float(threshold), 3)
        pred[test] = (scores[test] >= best_t).astype(int)
        folds.append({
            "fold": fold,
            "train_pairs": int(len(train)),
            "test_pairs": int(len(test)),
            "threshold": best_t,
            "f1": float(f1_score(labels[test], pred[test], zero_division=0)),
            "precision": float(precision_score(labels[test], pred[test], zero_division=0)),
            "recall": float(recall_score(labels[test], pred[test], zero_division=0)),
        })
    cm = confusion_matrix(labels, pred, labels=[1, 0])
    return {
        "mean_fold_f1": float(np.mean([x["f1"] for x in folds])),
        "std_fold_f1": float(np.std([x["f1"] for x in folds])),
        "pooled_f1": float(f1_score(labels, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "tp": int(cm[0, 0]), "fp": int(cm[1, 0]),
        "tn": int(cm[1, 1]), "fn": int(cm[0, 1]),
        "folds": folds,
    }


def main():
    rows = list(csv.DictReader((DATA / "pairs.csv").open(encoding="utf-8-sig", newline="")))
    raw = list(csv.DictReader((AUDIT / "clean_raw_embedding_vectors_138.csv").open(encoding="utf-8-sig", newline="")))
    key = lambda r: (r["doc_a"], r["doc_b"], r["label"], r["type"])
    if [key(r) for r in rows] != [key(r) for r in raw]:
        raise ValueError("Pair order/provenance mismatch between canonical pairs.csv and raw embedding CSV")
    labels = np.array([int(r["label"]) for r in rows])
    global_scores = np.array([float(r["SBERT_MiniLM"]) for r in raw])

    sw = load_sw()
    trees_raw = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    nodes = {k: sw.CapstoneNode.from_dict(v) for k, v in trees_raw.items()}
    # Historical canonical hybrid formula: alpha * structural + (1-alpha) * full-document cosine.
    structural_model = sw.SWCostModel(alpha=1.0, beta={"T2": 0.0, "T3": 0.9, "T4": 0.8}, cso_graph=None, max_depth=19)
    structural = np.array([sw.normalize_similarity(nodes[r["doc_a"]], nodes[r["doc_b"]], structural_model) for r in rows])
    alpha = 0.6
    hybrid = alpha * structural + (1.0 - alpha) * global_scores

    graph = nx.Graph()
    graph.add_edges_from((r["doc_a"], r["doc_b"]) for r in rows)
    component = {doc: i for i, comp in enumerate(nx.connected_components(graph)) for doc in comp}
    groups = np.array([component[r["doc_a"]] for r in rows])
    result = evaluate(hybrid, labels, groups)
    output = {
        "protocol": {
            "pairs": len(rows),
            "pair_file_sha256": hashlib.sha256((DATA / "pairs.csv").read_bytes()).hexdigest(),
            "split": "GroupKFold(5) by connected components of pair-document graph",
            "threshold_grid_step": 0.005,
            "threshold_train_only": True,
            "alpha": alpha,
            "formula": "alpha * structural_similarity + (1-alpha) * full_document_SBERT_cosine",
            "structural_beta": {"T2": 0.0, "T3": 0.9, "T4": 0.8},
            "n_documents": graph.number_of_nodes(),
            "n_components": len(set(groups)),
            "input_order_verified_by": "doc_a,doc_b,label,type",
        },
        "result": result,
    }
    (AUDIT / "document_disjoint_hybrid_138.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# Document-disjoint Hybrid audit — canonical 138 pairs", "",
        "The audit uses the canonical pair order, full-document MiniLM cosine scores, alpha = 0.6, and structural beta = (0.0, 0.9, 0.8). Connected components keep documents out of both train and test groups. Thresholds are selected on training groups only using a 0.005 grid.", "",
        "| Mean F1 | SD | Pooled F1 | MCC | Precision | Recall | TP | FP | TN | FN |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {result['mean_fold_f1']:.4f} | {result['std_fold_f1']:.4f} | {result['pooled_f1']:.4f} | {result['mcc']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['tp']} | {result['fp']} | {result['tn']} | {result['fn']} |",
        "", "This is a robustness audit. It does not establish that alpha = 0.6 was selected without access to the benchmark labels; the historical provenance audit describes alpha as a fixed configuration.",
    ]
    (ROOT / "reports" / "DOCUMENT_DISJOINT_HYBRID_138.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol": output["protocol"], "result": {k: result[k] for k in ("mean_fold_f1", "std_fold_f1", "pooled_f1", "mcc", "tp", "fp", "tn", "fn")}}, indent=2))


if __name__ == "__main__":
    main()
