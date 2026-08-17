"""Clean B1/B3/B4/B5 plus genuine flat-domain SBERT suite on 138 pairs."""
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import apted
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "repro_candidate_138"
DATA = STAGE / "data" / "dataset"
OUT = ROOT / "reports" / "audit"


def cv(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred = np.zeros(len(labels), dtype=int)
    folds = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(labels)), labels), 1):
        best, threshold = -1.0, 0.0
        for t in np.arange(0, 1.0001, 0.005):
            value = f1_score(labels[tr], scores[tr] >= t, zero_division=0)
            if value > best:
                best, threshold = value, round(float(t), 3)
        pred[te] = (scores[te] >= threshold).astype(int)
        folds.append({"fold": fold, "threshold": threshold,
                      "f1": float(f1_score(labels[te], pred[te], zero_division=0)),
                      "precision": float(precision_score(labels[te], pred[te], zero_division=0)),
                      "recall": float(recall_score(labels[te], pred[te], zero_division=0))})
    cm = confusion_matrix(labels, pred, labels=[1, 0])
    return pred, {"mean_f1": float(np.mean([x["f1"] for x in folds])), "std_f1": float(np.std([x["f1"] for x in folds])),
                  "mean_precision": float(np.mean([x["precision"] for x in folds])), "mean_recall": float(np.mean([x["recall"] for x in folds])),
                  "tp_fp_tn_fn": [int(cm[0, 0]), int(cm[1, 0]), int(cm[1, 1]), int(cm[0, 1])], "folds": folds}


def standard_ted_sequential(trees, pairs):
    """Same Standard TED definition as src.baselines, without Windows workers."""
    def walk(node):
        yield node
        for child in node.children:
            yield from walk(child)
    values = []
    for row in pairs:
        a, b = trees[row["doc_a"]], trees[row["doc_b"]]
        config = apted.Config()
        config.rename = lambda u, v: 0.0 if u.label == v.label else 1.0
        config.delete = lambda u: 1.0
        config.insert = lambda v: 1.0
        distance = apted.APTED(a, b, config).compute_edit_distance()
        denom = sum(1 for _ in walk(a)) + sum(1 for _ in walk(b))
        values.append(1.0 - distance / denom if denom else 1.0)
    return values


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import baselines
    from src.node import CapstoneNode
    pairs = list(csv.DictReader((DATA / "pairs.csv").open(encoding="utf-8-sig", newline="")))
    labels = np.array([int(x["label"]) for x in pairs])
    trees_raw = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    trees = {k: CapstoneNode.from_dict(v) for k, v in trees_raw.items()}
    import pandas as pd
    frame = pd.DataFrame(pairs)
    # Fit corpus-based baselines only on documents participating in the
    # canonical 138-pair slice; excluded regen documents must not affect IDF.
    pair_doc_ids = sorted(set(x["doc_a"] for x in pairs) | set(x["doc_b"] for x in pairs))
    corpus_trees = {k: trees[k] for k in pair_doc_ids}
    corpus_full = {k: full[k] for k in pair_doc_ids if k in full}

    print("Computing TF-IDF, Standard TED, pq-Gram, Section Cosine", flush=True)
    scores = {
        "TF-IDF": np.array(baselines.get_cosine_tfidf_similarity(corpus_trees, frame, corpus_full)),
        "Standard TED": np.array(standard_ted_sequential(trees, pairs)),
        "pq-Gram": np.array(baselines.get_pqgram_similarity(trees, frame)),
        "Section Cosine": np.array(baselines.get_section_cosine_similarity(corpus_trees, frame, corpus_full)),
    }

    domain_map = {"D1": ["context", "english_title", "vietnamese_title"], "D2": ["functional_requirement", "proposed_solutions", "products"], "D3": ["nonfunctional_requirement", "applied_theory"], "D4": ["proposed_tasks"]}
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    model = SentenceTransformer(str(cache))
    domain_emb = {k: {} for k in trees}
    for domain, fields in domain_map.items():
        texts = [" ".join(full.get(k, {}).get(f, "") for f in fields) for k in trees]
        emb = model.encode(texts, show_progress_bar=False)
        for k, value in zip(trees, emb): domain_emb[k][domain] = value
    flat = []
    for row in pairs:
        values = []
        for domain in domain_map:
            a, b = domain_emb[row["doc_a"]][domain], domain_emb[row["doc_b"]][domain]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            values.append(float(np.dot(a, b) / denom) if denom else 0.0)
        flat.append(float(np.mean(values)))
    scores["Genuine Flat Domain SBERT"] = np.array(flat)

    results, predictions = {}, {}
    for name, values in scores.items():
        predictions[name], results[name] = cv(values, labels)

    output = {"protocol": {"n_pairs": len(pairs), "seed": 42, "threshold_grid_step": 0.005, "threshold_train_only": True, "inputs": str(DATA)}, "results": results,
              "predictions": {name: pred.tolist() for name, pred in predictions.items()}}
    (OUT / "clean_baseline_suite_138.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (OUT / "clean_baseline_suite_pair_scores_138.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "doc_a", "doc_b", "label", "type"] + list(scores)
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(pairs): writer.writerow({**row, "index": i, **{k: float(v[i]) for k, v in scores.items()}})
    lines = ["# Clean baseline suite — canonical 138 pairs", "", "Protocol: 5-fold stratified CV, seed 42, threshold grid 0.005, train-fold-only selection.", "", "| Method | F1 | Std | Precision | Recall | TP | FP | TN | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        tp, fp, tn, fn = r["tp_fp_tn_fn"]
        lines.append(f"| {name} | {r['mean_f1']:.4f} | {r['std_f1']:.4f} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {tp} | {fp} | {tn} | {fn} |")
    (ROOT / "reports" / "CLEAN_BASELINE_SUITE_138.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
