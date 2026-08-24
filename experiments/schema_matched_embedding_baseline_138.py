"""Schema-matched flat embedding baseline on the canonical 138 pairs.

This baseline embeds the same D1-D4 text blocks used by the four-layer
representation, averages the four domain cosine similarities, and performs
the canonical train-fold-only threshold selection. It contains no tree edit
distance or structural alignment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"
MODEL = Path(os.environ.get("SCHEMA_MODEL", str(Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf")))
DOMAIN_MAP = {
    "D1": ["context", "english_title", "vietnamese_title"],
    "D2": ["functional_requirement", "proposed_solutions", "products"],
    "D3": ["nonfunctional_requirement", "applied_theory"],
    "D4": ["proposed_tasks"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cv(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    pred = np.zeros(len(labels), dtype=int)
    folds = []
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(split.split(np.zeros(len(labels)), labels), 1):
        best_f1, threshold = -1.0, 0.0
        for t in np.arange(0.0, 1.0001, 0.005):
            value = f1_score(labels[train], scores[train] >= t, zero_division=0)
            if value > best_f1:
                best_f1, threshold = float(value), round(float(t), 3)
        pred[test] = (scores[test] >= threshold).astype(int)
        folds.append({
            "fold": fold,
            "threshold": threshold,
            "f1": float(f1_score(labels[test], pred[test], zero_division=0)),
            "precision": float(precision_score(labels[test], pred[test], zero_division=0)),
            "recall": float(recall_score(labels[test], pred[test], zero_division=0)),
        })
    cm = confusion_matrix(labels, pred, labels=[1, 0])
    return pred, {
        "mean_f1": float(np.mean([x["f1"] for x in folds])),
        "std_f1": float(np.std([x["f1"] for x in folds])),
        "mean_precision": float(np.mean([x["precision"] for x in folds])),
        "mean_recall": float(np.mean([x["recall"] for x in folds])),
        "tp_fp_tn_fn": [int(cm[0, 0]), int(cm[1, 0]), int(cm[1, 1]), int(cm[0, 1])],
        "folds": folds,
    }


def main() -> None:
    pairs = list(csv.DictReader((DATA / "pairs.csv").open(encoding="utf-8-sig", newline="")))
    full = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    labels = np.asarray([int(row["label"]) for row in pairs])
    docs = sorted({row["doc_a"] for row in pairs} | {row["doc_b"] for row in pairs})
    model = SentenceTransformer(str(MODEL))
    domain_embeddings: dict[str, dict[str, np.ndarray]] = {d: {} for d in DOMAIN_MAP}
    for domain, fields in DOMAIN_MAP.items():
        texts = [" ".join(full.get(doc, {}).get(field, "") for field in fields) for doc in docs]
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=False)
        for doc, embedding in zip(docs, embeddings):
            domain_embeddings[domain][doc] = embedding
    scores = []
    per_pair = []
    for index, row in enumerate(pairs):
        domain_scores = {}
        for domain in DOMAIN_MAP:
            a = domain_embeddings[domain][row["doc_a"]]
            b = domain_embeddings[domain][row["doc_b"]]
            domain_scores[domain] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        score = float(np.mean(list(domain_scores.values())))
        scores.append(score)
        per_pair.append({**row, "index": index, "score": score, **domain_scores})
    values = np.asarray(scores)
    predictions, result = cv(values, labels)
    output = {
        "protocol": {
            "dataset": "canonical 138-pair real-only",
            "pairs_sha256": sha256(DATA / "pairs.csv"),
            "model_snapshot": str(MODEL),
            "representation": "D1-D4 domain text blocks from full_texts.json",
            "aggregation": "unweighted arithmetic mean of four domain cosine similarities",
            "pooling": "SentenceTransformer model pooling module",
            "tokenizer_max_length": int(model.max_seq_length),
            "encode_batch_size": 16,
            "threshold_grid_step": 0.005,
            "threshold_train_only": True,
            "cv": "5-fold StratifiedKFold, shuffle=True, random_state=42",
        },
        "result": result,
        "predictions": predictions.tolist(),
        "pair_scores": per_pair,
    }
    tag = "bge_small" if "bge-small" in str(MODEL).lower() else "minilm"
    (OUT / f"schema_matched_embedding_baseline_{tag}_138.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (OUT / f"schema_matched_embedding_pair_scores_{tag}_138.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "doc_a", "doc_b", "label", "type", "score", "D1", "D2", "D3", "D4"])
        writer.writeheader(); writer.writerows(per_pair)
    tp, fp, tn, fn = result["tp_fp_tn_fn"]
    md = [
        "# Schema-matched embedding baseline — canonical 138 pairs", "",
        "This is a flat baseline: it embeds D1–D4 separately, averages their cosine similarities, and uses no tree alignment or edit operations.", "",
        f"Protocol: MiniLM snapshot `{MODEL.name}`, model max length `{model.max_seq_length}`, SentenceTransformer pooling, 5-fold stratified CV (seed 42), train-fold-only threshold grid 0.005.", "",
        "| F1 mean | F1 std | Precision | Recall | TP | FP | TN | FN |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {result['mean_f1']:.4f} | {result['std_f1']:.4f} | {result['mean_precision']:.4f} | {result['mean_recall']:.4f} | {tp} | {fp} | {tn} | {fn} |", "",
        "Interpretation: this isolates the contribution of schema decomposition without SW-BTED's structural edit distance. It is a comparison baseline, not evidence that the embedding model is universally weak or strong.",
    ]
    (ROOT / "reports" / f"SCHEMA_MATCHED_EMBEDDING_BASELINE_{tag.upper()}_138.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"protocol": output["protocol"], "result": result}, indent=2))


if __name__ == "__main__":
    main()
