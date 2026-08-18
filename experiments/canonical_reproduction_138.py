"""Canonical side-by-side reproduction on the fixed 138-pair real-only slice.

This harness fixes the dataset, beta schedule, alpha, fold seed and threshold
grid. It compares the historical candidate source preserved in
``repro_candidate_138`` with the current workspace source.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
DATA = STAGE / "data"
REGEN = DATA / "processed" / "plag_regen_sections.json"
TREES = DATA / "dataset" / "trees_section.json"
PAIRS = DATA / "dataset" / "pairs.csv"
BETA = {"T2": 0.0, "T3": 0.9, "T4": 0.8}
ALPHA = 0.8
SEED = 42


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_slice(sw_module):
    trees = json.loads(TREES.read_text(encoding="utf-8"))
    pairs = pd.read_csv(PAIRS)
    regen = set(json.loads(REGEN.read_text(encoding="utf-8")).keys())
    real = pairs[~(pairs.doc_a.isin(regen) | pairs.doc_b.isin(regen))].reset_index(drop=True)
    labels = (real.type == "Type_A").astype(int).to_numpy()
    nodes = {key: sw_module.CapstoneNode.from_dict(value) for key, value in trees.items()}
    return real, labels, nodes


def evaluate(sw_module, beta_override=BETA):
    pairs, labels, nodes = load_slice(sw_module)
    model = sw_module.SWCostModel(alpha=ALPHA, beta=beta_override, cso_graph=None, max_depth=19)
    scores = np.array([
        sw_module.normalize_similarity(nodes[row.doc_a], nodes[row.doc_b], model)
        for row in pairs.itertuples()
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f1s, ps, rs, thresholds = [], [], [], []
    for train_idx, test_idx in skf.split(scores, labels):
        best_t, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.0, 1.01, 0.01):
            train_pred = (scores[train_idx] >= threshold).astype(int)
            value = f1_score(labels[train_idx], train_pred, zero_division=0)
            if value > best_f1:
                best_f1, best_t = value, float(threshold)
        test_pred = (scores[test_idx] >= best_t).astype(int)
        f1s.append(f1_score(labels[test_idx], test_pred, zero_division=0))
        ps.append(precision_score(labels[test_idx], test_pred, zero_division=0))
        rs.append(recall_score(labels[test_idx], test_pred, zero_division=0))
        thresholds.append(best_t)

    return {
        "n_pairs": int(len(pairs)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "alpha": ALPHA,
        "beta": beta_override if beta_override is not None else "module_config",
        "seed": SEED,
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_precision_mean": float(np.mean(ps)),
        "cv_recall_mean": float(np.mean(rs)),
        "fold_thresholds": thresholds,
        "score_mean": float(np.mean(scores)),
    }


def main():
    historical = load_module("historical_sw_bted", STAGE / "src" / "05_sw_bted.py")
    current = load_module("current_sw_bted", ROOT / "src" / "05_sw_bted.py")
    result = {
        "protocol": {
            "dataset": str(DATA),
            "tree_file": str(TREES),
            "pair_filter": "exclude every pair touching plag_regen_sections keys",
            "cv": "5-fold StratifiedKFold shuffle=True random_state=42",
            "threshold_grid": "0.00..1.00 step 0.01; selected on train fold",
        },
        "historical_source": evaluate(historical, BETA),
        "current_source": evaluate(current, BETA),
    }
    out = ROOT / "reports" / "canonical_reproduction_138.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ("historical_source", "current_source"):
        row = result[name]
        print(f"{name}: F1={row['cv_f1_mean']:.4f} +/- {row['cv_f1_std']:.4f}; "
              f"P={row['cv_precision_mean']:.4f}; R={row['cv_recall_mean']:.4f}; "
              f"thresholds={row['fold_thresholds']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
