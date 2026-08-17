"""Re-run the controlled 20-pair D2/D3 structural perturbation benchmark.

This is deliberately isolated from the historical script, whose output path
was shared with earlier runs.  The algorithm follows the recovered generator
and evaluator verbatim, while saving a new raw table and machine-readable
summary in the canonical workspace.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
OUT = ROOT / "reports" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

sw_mod = importlib.import_module("src.05_sw_bted")
from src.node import CapstoneNode


def generate_perturbed_pairs() -> list[dict]:
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    texts = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(DATA / "pairs.csv")
    type_a = pairs[pairs["type"] == "Type_A"].iloc[:20]
    records = []
    for _, row in type_a.iterrows():
        doc_id = row["doc_a"]
        original = trees[doc_id]
        perturbed = copy.deepcopy(original)
        children = perturbed.get("children", [])
        d2 = next((i for i, c in enumerate(children)
                   if c.get("schema_class") == "D2_FUNCTIONAL"), -1)
        d3 = next((i for i, c in enumerate(children)
                   if c.get("schema_class") == "D3_TECHNICAL_REALIZATION"), -1)
        if d2 != -1 and d3 != -1:
            children[d2]["schema_class"], children[d3]["schema_class"] = (
                "D3_TECHNICAL_REALIZATION", "D2_FUNCTIONAL"
            )
        records.append({
            "pair_id": f"P1_{doc_id}", "doc_a": doc_id,
            "tree_a": original, "tree_b": perturbed,
            "text_a": texts.get(doc_id, {}), "text_b": texts.get(doc_id, {}),
            "type": "Perturbed_P1_SectionReorder", "label": 0,
            "d2_index": d2, "d3_index": d3,
        })
    return records


def main() -> None:
    records = generate_perturbed_pairs()
    if len(records) != 20:
        raise RuntimeError(f"Expected 20 Type_A perturbations, got {len(records)}")
    if any(r["d2_index"] < 0 or r["d3_index"] < 0 for r in records):
        raise RuntimeError("At least one source tree lacks D2_FUNCTIONAL or D3_TECHNICAL_REALIZATION")

    # Use the pinned snapshot already used by the clean 138-pair baseline;
    # external Hugging Face access is intentionally not required.
    model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if not model_path.exists():
        raise FileNotFoundError(f"Pinned local model snapshot not found: {model_path}")
    model = SentenceTransformer(str(model_path))
    cost_model = sw_mod.SWCostModel(alpha=0.6)
    rows = []
    for rec in records:
        ta, tb = rec["tree_a"], rec["tree_b"]
        text_a = ta.get("label", "") + " " + " ".join(t for t in rec["text_a"].values() if t)
        text_b = tb.get("label", "") + " " + " ".join(t for t in rec["text_b"].values() if t)
        ea, eb = model.encode(text_a, show_progress_bar=False), model.encode(text_b, show_progress_bar=False)
        cosine = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        cosine = round(float(np.clip(cosine, 0.0, 1.0)), 4)
        na, nb = CapstoneNode.from_dict(ta), CapstoneNode.from_dict(tb)
        na.embedding, nb.embedding = ea.tolist(), eb.tolist()
        result = sw_mod.sw_bted(na, nb, cost_model)
        denom = result["max_possible_cost"]
        if result.get("pruned") or result["distance"] == float("inf"):
            structural = 0.0
        elif denom == 0:
            structural = 1.0
        else:
            normalized = result["distance"] / denom
            structural = (0.0 if normalized > cost_model.max_edit_budget_ratio else
                          1.0 - normalized / cost_model.max_edit_budget_ratio)
        structural = round(float(structural), 4)
        hybrid = round(0.6 * structural + 0.4 * cosine, 4)
        rows.append({
            "pair_id": rec["pair_id"], "doc_a": rec["doc_a"], "type": rec["type"],
            "d2_index": rec["d2_index"], "d3_index": rec["d3_index"],
            "sim_struct": structural, "sim_global_sbert": cosine,
            "sim_hybrid": hybrid, "label": rec["label"],
        })

    df = pd.DataFrame(rows)
    cutoff = 0.45
    labels = df.label.to_numpy(dtype=int)
    predictions = {
        "full_doc_sbert": (df.sim_global_sbert.to_numpy() >= cutoff).astype(int),
        "sw_bted_structural": (df.sim_struct.to_numpy() >= cutoff).astype(int),
        "sw_bted_hybrid": (df.sim_hybrid.to_numpy() >= cutoff).astype(int),
    }
    counts = {}
    for name, pred in predictions.items():
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        counts[name] = {"tp": 0, "fp": fp, "tn": tn, "fn": 0,
                        "accuracy": float(np.mean(pred == labels))}
    struct_ok, sbert_ok = predictions["sw_bted_structural"] == labels, predictions["full_doc_sbert"] == labels
    n10, n01 = int(np.sum(struct_ok & ~sbert_ok)), int(np.sum(~struct_ok & sbert_ok))
    p = float(binomtest(min(n10, n01), n=n10 + n01, p=0.5).pvalue) if n10 + n01 else 1.0
    summary = {
        "n_pairs": len(df), "cutoff": cutoff, "generator": str(Path(r"D:\FPT\Semester_8\RAG_Research\scratch\run_structural_perturbation_benchmark.py")),
        "source_dataset": {"trees": str(DATA / "trees_section.json"), "texts": str(DATA / "full_texts.json"), "pairs": str(DATA / "pairs.csv")},
        "score_distributions": {c: {"mean": float(df[c].mean()), "min": float(df[c].min()), "max": float(df[c].max())} for c in ["sim_struct", "sim_global_sbert", "sim_hybrid"]},
        "confusion": counts, "mcnemar": {"n10_struct_correct_sbert_wrong": n10, "n01_sbert_correct_struct_wrong": n01, "p_exact": p},
        "rows": rows,
    }
    csv_path = OUT / "clean_structural_perturbation_results_20.csv"
    json_path = OUT / "clean_structural_perturbation_metrics_20.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
