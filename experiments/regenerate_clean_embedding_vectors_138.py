"""Regenerate embedding baseline vectors directly from the canonical 138-pair inputs.

This intentionally writes new artifacts and does not modify the historical audit JSON.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_texts():
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    texts = {}
    for key, tree in trees.items():
        sections = full.get(key, {})
        texts[key] = tree.get("label", "") + " " + " ".join(
            value for value in sections.values() if value
        )
    return texts


def main():
    pairs_path = DATA / "pairs.csv"
    with pairs_path.open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    texts = load_texts()
    keys = list(texts)

    # Use the already cached immutable snapshots so this audit is offline and
    # cannot silently resolve a newer model revision.
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_specs = {
        "SBERT_MiniLM": str(cache / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
        "BGE_Small_v1.5": str(cache / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
        "MPNet_Base_v2": str(cache / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
    }
    scores = {}
    for name, model_id in model_specs.items():
        print(f"Loading {name}: {model_id}", flush=True)
        model = SentenceTransformer(model_id)
        embeddings = model.encode(
            [texts[key] for key in keys],
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        index = {key: i for i, key in enumerate(keys)}
        values = []
        for row in pairs:
            a = embeddings[index[row["doc_a"]]]
            b = embeddings[index[row["doc_b"]]]
            values.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        scores[name] = values

    rows = []
    for i, pair in enumerate(pairs):
        row = dict(pair, index=i)
        for name, values in scores.items():
            row[name] = values[i]
        rows.append(row)

    csv_path = OUT / "clean_raw_embedding_vectors_138.csv"
    fields = ["index", "doc_a", "doc_b", "label", "type", *model_specs]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT / "clean_raw_embedding_vectors_138.json"
    payload = {
        "protocol": {
            "dataset": "canonical 138-pair real-only input",
            "pairs_sha256": sha256(pairs_path),
            "text_construction": "tree label + full_texts section values in JSON order",
            "normalize_embeddings": False,
            "cosine": "dot(a,b)/(||a||*||b||)",
            "model_ids": model_specs,
        },
        "pairs": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        if row["index"] == 84:
            print("INDEX_84", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
