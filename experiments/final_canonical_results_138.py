import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
DATA = STAGE / "data" / "dataset"
AUDIT = ROOT / "reports" / "audit"
BOOT_SEED = 20260814


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues), float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted, adjusted <= 0.05


def load_sw():
    spec = importlib.util.spec_from_file_location("sw_final", STAGE / "src" / "05_sw_bted.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cv(scores, labels, step=0.005):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pred = np.zeros(len(labels), int)
    folds = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels), 1):
        best, threshold = -1.0, 0.0
        for t in np.arange(0, 1.0001, step):
            value = f1_score(labels[train], scores[train] >= t, zero_division=0)
            if value > best:
                best, threshold = value, round(float(t), 3)
        pred[test] = (scores[test] >= threshold).astype(int)
        folds.append({"fold": fold, "threshold": threshold,
                      "f1": float(f1_score(labels[test], pred[test], zero_division=0)),
                      "precision": float(precision_score(labels[test], pred[test], zero_division=0)),
                      "recall": float(recall_score(labels[test], pred[test], zero_division=0))})
    return pred, folds


def bootstrap(pred, labels, n=2000):
    rng = np.random.default_rng(BOOT_SEED)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        values.append([f1_score(labels[idx], pred[idx], zero_division=0),
                       precision_score(labels[idx], pred[idx], zero_division=0),
                       recall_score(labels[idx], pred[idx], zero_division=0)])
    values = np.asarray(values)
    return {name: [float(np.quantile(values[:, i], .025)), float(np.quantile(values[:, i], .975))]
            for i, name in enumerate(("f1", "precision", "recall"))}


def main():
    rows = list(csv.DictReader((DATA / "pairs.csv").open(encoding="utf-8-sig", newline="")))
    labels = np.array([int(row["label"]) for row in rows])
    keys = [(row["doc_a"], row["doc_b"], row["label"], row["type"]) for row in rows]
    clean = json.loads((AUDIT / "clean_raw_embedding_vectors_138.json").read_text(encoding="utf-8"))["pairs"]
    assert keys == [(row["doc_a"], row["doc_b"], row["label"], row["type"]) for row in clean]

    sw = load_sw()
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    nodes = {key: sw.CapstoneNode.from_dict(value) for key, value in trees.items()}
    model = sw.SWCostModel(alpha=.8, beta={"T2": 0.0, "T3": .9, "T4": .8}, cso_graph=None, max_depth=19)
    scores = {"SW-BTED": np.array([sw.normalize_similarity(nodes[r["doc_a"]], nodes[r["doc_b"]], model) for r in rows])}
    for name in ("SBERT_MiniLM", "BGE_Small_v1.5", "MPNet_Base_v2"):
        scores[name] = np.array([float(row[name]) for row in clean])

    suite = json.loads((AUDIT / "clean_baseline_suite_138.json").read_text(encoding="utf-8"))
    predictions = {}
    fold_data = {}
    for name, values in scores.items():
        predictions[name], fold_data[name] = cv(values, labels)
    for name in ("TF-IDF", "Standard TED", "pq-Gram", "Section Cosine", "Genuine Flat Domain SBERT"):
        predictions[name] = np.array(suite["predictions"][name])
        fold_data[name] = suite["results"][name]["folds"]

    qwen = list(csv.DictReader((ROOT / "reports" / "qwen3_pair_prediction_audit.csv").open(encoding="utf-8-sig", newline="")))
    assert keys == [(row["doc_a"], row["doc_b"], row["label"], row["type"]) for row in qwen]
    predictions["Qwen3-Embedding-4B"] = np.array([int(row["oof_prediction"]) for row in qwen])
    fold_data["Qwen3-Embedding-4B"] = []

    methods = {}
    for name, pred in predictions.items():
        methods[name] = {
            "mean_fold_f1": float(np.mean([x["f1"] for x in fold_data[name]])) if fold_data[name] else None,
            "std_fold_f1": float(np.std([x["f1"] for x in fold_data[name]])) if fold_data[name] else None,
            "pooled_f1": float(f1_score(labels, pred, zero_division=0)),
            "pooled_precision": float(precision_score(labels, pred, zero_division=0)),
            "pooled_recall": float(recall_score(labels, pred, zero_division=0)),
            "bootstrap_95_ci": bootstrap(pred, labels),
            "predictions": pred.tolist(),
        }

    tests = []
    sw_pred = predictions["SW-BTED"]
    for name, pred in predictions.items():
        if name == "SW-BTED":
            continue
        sw_ok, base_ok = sw_pred == labels, pred == labels
        b = int(np.sum(sw_ok & ~base_ok))
        c = int(np.sum(~sw_ok & base_ok))
        raw = float(binomtest(min(b, c), b + c, p=.5).pvalue) if b + c else 1.0
        tests.append({"baseline": name, "sw_only_correct": b, "baseline_only_correct": c, "raw_p": raw})
    adjusted, reject = holm([x["raw_p"] for x in tests])
    for item, is_rejected, adjusted_p in zip(tests, reject, adjusted):
        item["holm_p"] = float(adjusted_p)
        item["significant_holm_0.05"] = bool(is_rejected)

    result = {"protocol": {"pairs_sha256": hashlib.sha256((DATA / "pairs.csv").read_bytes()).hexdigest(), "cv_seed": 42, "threshold_grid_step": .005, "bootstrap_resamples": 2000, "bootstrap_seed": BOOT_SEED, "prediction_matching": "doc_a,doc_b,label,type"}, "methods": methods, "mcnemar_vs_sw_bted": tests}
    (AUDIT / "final_canonical_results_138.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (AUDIT / "final_canonical_predictions_138.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "doc_a", "doc_b", "label", "type"] + list(predictions)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({**row, "index": i, **{name: int(pred[i]) for name, pred in predictions.items()}})
    print(json.dumps({"methods": {k: {x: v[x] for x in ("mean_fold_f1", "std_fold_f1", "pooled_f1", "pooled_precision", "pooled_recall")} for k, v in methods.items()}, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
