"""Independent audit of the downloaded Qwen3 pair scores and MiniLM comparison."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data"
QWEN = ROOT / "kaggle" / "qwen3_results" / "qwen3_results"
MODEL = r"C:\Users\DuyTuanPC\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def fold_predictions(scores, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(len(labels), dtype=int)
    thresholds = np.zeros(len(labels), dtype=float)
    fold_rows = []
    for fold, (train, test) in enumerate(skf.split(scores, labels), start=1):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.0, 1.001, 0.01):
            value = f1_score(labels[train], (scores[train] >= t).astype(int), zero_division=0)
            if value > best_f1:
                best_f1, best_t = value, float(t)
        predictions[test] = (scores[test] >= best_t).astype(int)
        thresholds[test] = best_t
        fold_rows.append({
            "fold": fold, "n_test": int(len(test)), "threshold_selected_on": "train_fold",
            "threshold": best_t,
            "f1": float(f1_score(labels[test], predictions[test], zero_division=0)),
            "precision": float(precision_score(labels[test], predictions[test], zero_division=0)),
            "recall": float(recall_score(labels[test], predictions[test], zero_division=0)),
        })
    return predictions, thresholds, fold_rows


def score_summary(labels, scores, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "f1_pooled_oof": float(f1_score(labels, predictions, zero_division=0)),
        "precision_pooled_oof": float(precision_score(labels, predictions, zero_division=0)),
        "recall_pooled_oof": float(recall_score(labels, predictions, zero_division=0)),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
    }


def main():
    qwen_df = pd.read_csv(QWEN / "qwen3_pair_scores.csv")
    labels = qwen_df.label.to_numpy(dtype=int)
    qwen_scores = qwen_df.qwen3_cosine.to_numpy(dtype=float)
    qwen_pred, qwen_thresholds, qwen_folds = fold_predictions(qwen_scores, labels)
    qwen_df["fold_threshold"] = qwen_thresholds
    qwen_df["oof_prediction"] = qwen_pred
    qwen_df["correct"] = qwen_pred == labels

    # Independent MiniLM encoding of the same 178 documents and same pair order.
    raw_trees = json.loads((DATA / "dataset" / "trees_section.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "dataset" / "full_texts.json").read_text(encoding="utf-8"))
    keys = sorted(set(qwen_df.doc_a) | set(qwen_df.doc_b))
    texts = [raw_trees[k].get("label", k) + " " + " ".join(str(v) for v in full.get(k, {}).values() if v) for k in keys]
    encoder = SentenceTransformer(MODEL)
    embeddings = encoder.encode(texts, show_progress_bar=False)
    idx = {key: i for i, key in enumerate(keys)}
    mini_scores = np.array([float(np.dot(embeddings[idx[r.doc_a]], embeddings[idx[r.doc_b]]) / (np.linalg.norm(embeddings[idx[r.doc_a]]) * np.linalg.norm(embeddings[idx[r.doc_b]]))) for r in qwen_df.itertuples()])
    mini_pred, mini_thresholds, mini_folds = fold_predictions(mini_scores, labels)
    qwen_df["minilm_cosine_independent"] = mini_scores
    qwen_df["minilm_fold_threshold"] = mini_thresholds
    qwen_df["minilm_oof_prediction"] = mini_pred

    pair84 = qwen_df.iloc[84].to_dict()
    result = {
        "source": {
            "qwen_pair_scores": str(QWEN / "qwen3_pair_scores.csv"),
            "qwen_provenance": str(QWEN / "provenance_manifest.json"),
            "pair_order": "preserved from downloaded Kaggle CSV",
            "qwen_model": "Qwen/Qwen3-Embedding-4B",
            "minilm_model_path": MODEL,
        },
        "protocol": {"n_pairs": int(len(labels)), "positive": int(labels.sum()), "negative": int((labels == 0).sum()), "cv": "5-fold StratifiedKFold shuffle=True random_state=42", "threshold_selection": "training fold only", "threshold_grid": "0.00..1.00 step 0.01"},
        "qwen3": {**score_summary(labels, qwen_scores, qwen_pred), "folds": qwen_folds},
        "minilm_independent": {**score_summary(labels, mini_scores, mini_pred), "folds": mini_folds},
        "pair_index_84_zero_based": pair84,
        "same_oof_predictions": int(np.sum(qwen_pred == mini_pred)),
        "different_oof_predictions": int(np.sum(qwen_pred != mini_pred)),
    }
    out = ROOT / "reports" / "qwen3_prediction_audit.json"
    out.write_text(json.dumps(result, indent=2, default=lambda x: float(x)), encoding="utf-8")
    qwen_df.to_csv(ROOT / "reports" / "qwen3_pair_prediction_audit.csv", index=False)
    print(json.dumps(result, indent=2, default=lambda x: float(x)))


if __name__ == "__main__":
    main()
