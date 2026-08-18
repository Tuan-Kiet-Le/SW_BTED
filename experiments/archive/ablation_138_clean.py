"""Fold-safe, reproducible beta-schedule ablation on the canonical 138 pairs."""
import json
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "repro_candidate_138" / "data"
sw = importlib.import_module("src.05_sw_bted")

SCHEDULES = {
    "documented_beta_080": {"T2": 0.0, "T3": 0.9, "T4": 0.8},
    "current_beta_100": {"T2": 0.0, "T3": 0.9, "T4": 1.0},
    "uniform_beta_050": {"T2": 0.0, "T3": 0.5, "T4": 0.5},
    "content_heavy_beta_100": {"T2": 0.0, "T3": 1.0, "T4": 1.0},
    "schema_heavy_beta_000": {"T2": 0.0, "T3": 0.0, "T4": 0.0},
}


def load_data():
    trees_raw = json.loads((DATA / "dataset" / "trees_section.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(DATA / "dataset" / "pairs.csv")
    regen = set(json.loads((DATA / "processed" / "plag_regen_sections.json").read_text(encoding="utf-8")).keys())
    pairs = pairs[~(pairs.doc_a.isin(regen) | pairs.doc_b.isin(regen))].reset_index(drop=True)
    labels = (pairs.type == "Type_A").astype(int).to_numpy()
    trees = {key: sw.CapstoneNode.from_dict(value) for key, value in trees_raw.items()}
    return pairs, labels, trees


def evaluate(scores, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, ps, rs, thresholds = [], [], [], []
    for train, test in skf.split(scores, labels):
        best_t, best_f1 = 0.5, -1.0
        for threshold in np.arange(0.0, 1.001, 0.01):
            value = f1_score(labels[train], (scores[train] >= threshold).astype(int), zero_division=0)
            if value > best_f1:
                best_f1, best_t = value, float(threshold)
        pred = (scores[test] >= best_t).astype(int)
        f1s.append(f1_score(labels[test], pred, zero_division=0))
        ps.append(precision_score(labels[test], pred, zero_division=0))
        rs.append(recall_score(labels[test], pred, zero_division=0))
        thresholds.append(best_t)
    return {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(ps)), "recall_mean": float(np.mean(rs)),
        "fold_thresholds": thresholds,
    }


def main():
    pairs, labels, trees = load_data()
    output = {"n_pairs": len(pairs), "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum()), "schedules": {}}
    for name, beta in SCHEDULES.items():
        model = sw.SWCostModel(alpha=1.0, beta=beta, cso_graph=None, max_depth=19)
        scores = np.array([sw.normalize_similarity(trees[r.doc_a], trees[r.doc_b], model) for r in pairs.itertuples()])
        output["schedules"][name] = {"beta": beta, **evaluate(scores, labels), "score_mean": float(scores.mean())}
        row = output["schedules"][name]
        print(f"{name}: F1={row['f1_mean']:.4f} +/- {row['f1_std']:.4f}; P={row['precision_mean']:.4f}; R={row['recall_mean']:.4f}; thresholds={row['fold_thresholds']}")
    out = ROOT / "reports" / "ablation_138_clean.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
