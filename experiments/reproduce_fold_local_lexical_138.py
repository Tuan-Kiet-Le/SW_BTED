"""Reproduce TF-IDF and Section Cosine with fold-local fitting on 138 pairs."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
DATA = STAGE / "data" / "dataset"
OUT = ROOT / "reports" / "audit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.node import CapstoneNode
import baselines


SECTION_WEIGHTS = {
    "Context": 0.10,
    "Problem": 0.15,
    "Solution": 0.25,
    "Theory": 0.15,
    "Deliverables": 0.10,
    "Methodology": 0.15,
    "Timeline": 0.05,
    "References": 0.05,
}


def document_text(doc_id, full_texts, trees):
    return baselines.get_document_full_text(doc_id, full_texts, trees[doc_id])


def section_text(doc_id, section, full_texts, trees):
    return baselines.get_document_section_text(doc_id, section, full_texts, trees[doc_id])


def score_tfidf(subset, train_docs, full_texts, trees):
    docs = {doc_id: document_text(doc_id, full_texts, trees) for doc_id in train_docs}
    vectorizer = TfidfVectorizer()
    vectorizer.fit([docs[doc_id] for doc_id in sorted(docs)])
    output = []
    for row in subset.itertuples(index=False):
        vectors = vectorizer.transform([
            document_text(row.doc_a, full_texts, trees),
            document_text(row.doc_b, full_texts, trees),
        ])
        output.append(float(cosine_similarity(vectors[0], vectors[1])[0, 0]))
    return np.asarray(output, dtype=float)


def score_section_cosine(subset, train_docs, full_texts, trees):
    vectorizers = {}
    for section in SECTION_WEIGHTS:
        texts = [section_text(doc_id, section, full_texts, trees) for doc_id in sorted(train_docs)]
        vectorizer = TfidfVectorizer()
        if any(text.strip() for text in texts):
            vectorizer.fit(texts)
            vectorizers[section] = vectorizer

    output = []
    for row in subset.itertuples(index=False):
        total = 0.0
        weight = 0.0
        for section, section_weight in SECTION_WEIGHTS.items():
            if section not in vectorizers:
                continue
            text_a = section_text(row.doc_a, section, full_texts, trees).strip()
            text_b = section_text(row.doc_b, section, full_texts, trees).strip()
            if not text_a and not text_b:
                similarity = 1.0
            elif not text_a or not text_b:
                similarity = 0.0
            else:
                vectors = vectorizers[section].transform([text_a, text_b])
                similarity = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
            total += section_weight * similarity
            weight += section_weight
        output.append(total / weight if weight else 1.0)
    return np.asarray(output, dtype=float)


def evaluate(scores_by_fold, labels, folds):
    predictions = np.zeros(len(labels), dtype=int)
    fold_metrics = []
    thresholds = []
    for fold_number, train_idx, test_idx, train_scores, test_scores in scores_by_fold:
        best_f1 = -1.0
        best_threshold = 0.0
        for threshold in np.arange(0.0, 1.0001, 0.005):
            candidate = (train_scores >= threshold).astype(int)
            value = f1_score(labels[train_idx], candidate, zero_division=0)
            if value > best_f1:
                best_f1 = value
                best_threshold = round(float(threshold), 3)
        test_prediction = (test_scores >= best_threshold).astype(int)
        predictions[test_idx] = test_prediction
        thresholds.append(best_threshold)
        fold_metrics.append({
            "fold": fold_number,
            "threshold": best_threshold,
            "f1": float(f1_score(labels[test_idx], test_prediction, zero_division=0)),
            "precision": float(precision_score(labels[test_idx], test_prediction, zero_division=0)),
            "recall": float(recall_score(labels[test_idx], test_prediction, zero_division=0)),
        })
    cm = confusion_matrix(labels, predictions, labels=[1, 0])
    return predictions, {
        "mean_f1": float(np.mean([item["f1"] for item in fold_metrics])),
        "std_f1": float(np.std([item["f1"] for item in fold_metrics])),
        "mean_precision": float(np.mean([item["precision"] for item in fold_metrics])),
        "mean_recall": float(np.mean([item["recall"] for item in fold_metrics])),
        "pooled_f1": float(f1_score(labels, predictions, zero_division=0)),
        "pooled_precision": float(precision_score(labels, predictions, zero_division=0)),
        "pooled_recall": float(recall_score(labels, predictions, zero_division=0)),
        "tp_fp_tn_fn": [int(cm[0, 0]), int(cm[1, 0]), int(cm[1, 1]), int(cm[0, 1])],
        "thresholds": thresholds,
        "folds": fold_metrics,
    }


def main():
    pairs = pd.read_csv(DATA / "pairs.csv")
    labels = (pairs["type"] == "Type_A").astype(int).to_numpy()
    trees_raw = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    full_texts = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    trees = {key: CapstoneNode.from_dict(value) for key, value in trees_raw.items()}

    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_indices = list(splitter.split(pairs, labels))
    all_results = {}
    all_predictions = {}
    all_scores = {}

    for name, scorer in (("TF-IDF", score_tfidf), ("Section Cosine", score_section_cosine)):
        fold_scores = []
        for fold_number, (train_idx, test_idx) in enumerate(fold_indices, 1):
            train_rows = pairs.iloc[train_idx]
            test_rows = pairs.iloc[test_idx]
            train_docs = set(train_rows["doc_a"]) | set(train_rows["doc_b"])
            train_scores = scorer(train_rows, train_docs, full_texts, trees)
            test_scores = scorer(test_rows, train_docs, full_texts, trees)
            fold_scores.append((fold_number, train_idx, test_idx, train_scores, test_scores))
        predictions, metrics = evaluate(fold_scores, labels, fold_indices)
        all_results[name] = metrics
        all_predictions[name] = predictions.tolist()
        score_vector = np.zeros(len(pairs), dtype=float)
        for _, _, test_idx, _, test_scores in fold_scores:
            score_vector[test_idx] = test_scores
        all_scores[name] = score_vector.tolist()

    output = {
        "protocol": {
            "pairs": 138,
            "outer_cv": "5-fold StratifiedKFold",
            "seed": 42,
            "tfidf_fit_scope": "documents participating in the training portion of each outer fold only",
            "threshold_selection": "training-fold scores only",
            "threshold_grid_step": 0.005,
            "prediction_order": "canonical pairs.csv order",
        },
        "results": all_results,
        "predictions": all_predictions,
        "scores": all_scores,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fold_local_lexical_suite_138.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (OUT / "fold_local_lexical_pair_scores_138.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "doc_a", "doc_b", "label", "type", "TF-IDF", "Section Cosine"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in pairs.iterrows():
            writer.writerow({**row.to_dict(), "index": index, "TF-IDF": all_scores["TF-IDF"][index], "Section Cosine": all_scores["Section Cosine"][index]})
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
