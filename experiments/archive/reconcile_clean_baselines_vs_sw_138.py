"""Compare clean embedding OOF predictions with canonical SW-BTED predictions."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
PAIR_JSON = ROOT / "reports" / "audit" / "clean_raw_embedding_vectors_138.json"
OUT = ROOT / "reports" / "CLEAN_BASELINE_RECONCILIATION_138.md"
OUT_JSON = ROOT / "reports" / "audit" / "clean_baseline_reconciliation_138.json"


def load_module(path):
    spec = importlib.util.spec_from_file_location("sw_clean_reconcile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cv_predictions(scores, labels, step=0.005):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = np.zeros(len(labels), dtype=int)
    folds = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        best_f1, best_t = -1.0, 0.0
        for t in np.arange(0, 1.0001, step):
            value = f1_score(labels[train], scores[train] >= t, zero_division=0)
            if value > best_f1:
                best_f1, best_t = value, round(float(t), 3)
        pred[test] = (scores[test] >= best_t).astype(int)
        folds.append({"fold": fold, "threshold": best_t,
                      "f1": float(f1_score(labels[test], pred[test], zero_division=0)),
                      "precision": float(precision_score(labels[test], pred[test], zero_division=0)),
                      "recall": float(recall_score(labels[test], pred[test], zero_division=0))})
    return pred, folds


def metrics(labels, pred, folds):
    cm = confusion_matrix(labels, pred, labels=[1, 0])
    return {"mean_f1": float(np.mean([x["f1"] for x in folds])),
            "std_f1": float(np.std([x["f1"] for x in folds])),
            "mean_precision": float(np.mean([x["precision"] for x in folds])),
            "mean_recall": float(np.mean([x["recall"] for x in folds])),
            "tp_fp_tn_fn": [int(cm[0, 0]), int(cm[1, 0]), int(cm[1, 1]), int(cm[0, 1])],
            "folds": folds}


def main():
    payload = json.loads(PAIR_JSON.read_text(encoding="utf-8"))
    rows = payload["pairs"]
    labels = np.array([int(x["label"]) for x in rows])
    vectors = {name: np.array([float(x[name]) for x in rows]) for name in ("SBERT_MiniLM", "BGE_Small_v1.5", "MPNet_Base_v2")}

    sw = load_module(STAGE / "src" / "05_sw_bted.py")
    trees = json.loads((STAGE / "data" / "dataset" / "trees_section.json").read_text(encoding="utf-8"))
    nodes = {key: sw.CapstoneNode.from_dict(value) for key, value in trees.items()}
    model = sw.SWCostModel(alpha=0.8, beta={"T2": 0.0, "T3": 0.9, "T4": 0.8}, cso_graph=None, max_depth=19)
    sw_scores = np.array([sw.normalize_similarity(nodes[x["doc_a"]], nodes[x["doc_b"]], model) for x in rows])

    scores = {"SW_BTED": sw_scores, **vectors}
    predictions = {}
    result = {}
    for name, vals in scores.items():
        pred, folds = cv_predictions(vals, labels)
        predictions[name] = pred
        result[name] = metrics(labels, pred, folds)

    suite_path = ROOT / "reports" / "audit" / "clean_baseline_suite_138.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8")) if suite_path.exists() else None
    if suite:
        for name, suite_result in suite["results"].items():
            result[name] = suite_result
            predictions[name] = np.array(suite["predictions"][name])

    comparisons = []
    for name in ("SBERT_MiniLM", "BGE_Small_v1.5", "MPNet_Base_v2"):
        a, b = predictions["SW_BTED"], predictions[name]
        a_ok, b_ok = a == labels, b == labels
        b_count = int(np.sum(a_ok & ~b_ok))
        c_count = int(np.sum(~a_ok & b_ok))
        n = b_count + c_count
        p = float(binomtest(min(b_count, c_count), n, p=0.5).pvalue) if n else 1.0
        comparisons.append({"baseline": name, "sw_only_correct": b_count, "baseline_only_correct": c_count, "discordant": n, "mcnemar_exact_p": p})

    if suite_path.exists():
        for name, values in suite["predictions"].items():
            a, b = predictions["SW_BTED"], np.array(values)
            a_ok, b_ok = a == labels, b == labels
            b_count = int(np.sum(a_ok & ~b_ok)); c_count = int(np.sum(~a_ok & b_ok)); n = b_count + c_count
            p = float(binomtest(min(b_count, c_count), n, p=0.5).pvalue) if n else 1.0
            comparisons.append({"baseline": name, "sw_only_correct": b_count, "baseline_only_correct": c_count, "discordant": n, "mcnemar_exact_p": p})

    output = {"protocol": {"folds": 5, "seed": 42, "threshold_step": 0.005, "threshold_train_only": True}, "metrics": result, "mcnemar_vs_sw": comparisons}
    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# Clean baseline reconciliation — canonical 138 pairs", "", "Protocol: same 138-pair input, five-fold StratifiedKFold (seed 42), 0.005 threshold grid, train-fold-only threshold selection.", "", "| Method | Mean F1 | Std | Precision | Recall | TP | FP | TN | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in result.items():
        tp, fp, tn, fn = r["tp_fp_tn_fn"]
        lines.append(f"| {name} | {r['mean_f1']:.4f} | {r['std_f1']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {tp} | {fp} | {tn} | {fn} |")
    lines += ["", "## McNemar exact tests vs SW-BTED", "", "| Baseline | SW-only correct | Baseline-only correct | Discordant | Exact p-value |", "|---|---:|---:|---:|---:|"]
    for x in comparisons:
        lines.append(f"| {x['baseline']} | {x['sw_only_correct']} | {x['baseline_only_correct']} | {x['discordant']} | {x['mcnemar_exact_p']:.6g} |")
    lines += ["", "The old manually anchored raw-vector artifact was not used."]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    for name, r in result.items(): print(name, r["mean_f1"], r["tp_fp_tn_fn"])
    print("McNemar", comparisons)


if __name__ == "__main__":
    main()
