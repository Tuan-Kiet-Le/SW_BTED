"""Audit embedding input length and model truncation on canonical 138 pairs.

This is a read-only scientific audit of the existing canonical inputs.  It
does not regenerate embeddings or overwrite any historical result.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_texts() -> dict[str, str]:
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    full = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    return {
        key: tree.get("label", "") + " " + " ".join(v for v in full.get(key, {}).values() if v)
        for key, tree in trees.items()
    }


def stats(lengths: list[int], limit: int) -> dict:
    values = np.asarray(lengths, dtype=int)
    return {
        "n_documents": int(len(values)),
        "min_tokens": int(values.min()) if len(values) else 0,
        "median_tokens": float(np.median(values)) if len(values) else 0.0,
        "p95_tokens": float(np.percentile(values, 95)) if len(values) else 0.0,
        "max_tokens": int(values.max()) if len(values) else 0,
        "n_documents_over_model_limit": int(np.sum(values > limit)),
        "fraction_documents_over_model_limit": float(np.mean(values > limit)) if len(values) else 0.0,
    }


def main() -> None:
    pairs_path = DATA / "pairs.csv"
    pairs = list(csv.DictReader(pairs_path.open(encoding="utf-8-sig", newline="")))
    pair_docs = sorted({r["doc_a"] for r in pairs} | {r["doc_b"] for r in pairs})
    texts = build_texts()
    models = {
        "SBERT_MiniLM": ROOT / "model_snapshots" / "missing",
        "BGE_Small_v1.5": ROOT / "model_snapshots" / "missing",
        "MPNet_Base_v2": ROOT / "model_snapshots" / "missing",
    }
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = {
        "SBERT_MiniLM": hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "BGE_Small_v1.5": hub / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "MPNet_Base_v2": hub / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    }
    models = {k: v for k, v in candidates.items() if v.exists()}
    report = {
        "dataset": {
            "pairs": len(pairs),
            "participating_documents": len(pair_docs),
            "pairs_sha256": sha256(pairs_path),
            "text_source": "tree label + full_texts.json section values in JSON order",
        },
        "models": {},
    }
    for name, path in models.items():
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        tokenizer_limit = int(getattr(tokenizer, "model_max_length", 512))
        # SentenceTransformer may impose a shorter effective limit than the
        # underlying tokenizer (MiniLM's cached pipeline is one such case).
        effective_limit = int(SentenceTransformer(str(path)).max_seq_length)
        # The historical runs did not pass truncation/max_length explicitly;
        # SentenceTransformer therefore uses its model/tokenizer max length.
        lengths = [len(tokenizer(texts[k], add_special_tokens=True, truncation=False)["input_ids"]) for k in pair_docs]
        report["models"][name] = {
            "snapshot": str(path),
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_model_max_length": tokenizer_limit,
            "sentence_transformer_effective_max_seq_length": effective_limit,
            "encode_call_in_historical_script": "model.encode(texts, show_progress_bar=False, normalize_embeddings=False where applicable)",
            "truncation_behavior": "inputs longer than model_max_length are truncated by the SentenceTransformer Transformer module",
            "length_stats": stats(lengths, effective_limit),
        }
    out_json = OUT / "embedding_input_scope_audit_138.json"
    out_md = OUT / "EMBEDDING_INPUT_SCOPE_AUDIT_138.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Embedding input-scope audit — canonical 138 pairs",
        "",
        f"Dataset: `{len(pairs)}` pairs, `{len(pair_docs)}` participating documents.",
        f"Pair hash: `{report['dataset']['pairs_sha256']}`.",
        "",
        "The audit measures untruncated tokenizer lengths for the exact full-document strings used by the clean embedding scripts. Historical `encode()` calls did not pass an explicit max length; the model tokenizer limit therefore governs truncation.",
        "",
        "| Model | Tokenizer max | Effective encode max | Median | P95 | Max observed | Docs over effective limit | Fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["models"].items():
        s = item["length_stats"]
        lines.append(f"| {name} | {item['tokenizer_model_max_length']} | {item['sentence_transformer_effective_max_seq_length']} | {s['median_tokens']:.0f} | {s['p95_tokens']:.0f} | {s['max_tokens']} | {s['n_documents_over_model_limit']} | {s['fraction_documents_over_model_limit']:.3f} |")
    lines += [
        "",
        "Interpretation: truncation is a protocol limitation when the fraction above the model limit is non-zero. It does not invalidate the historical result, but the manuscript should report the limit and affected-document count, and schema-matched experiments should use the same explicit tokenizer/truncation policy.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
