"""Secondary observable structural perturbation audit.

Unlike the schema-reassignment test, this construction changes both the tree
content assigned to D2/D3 and the serialized section order supplied to the
text embedding. It is a diagnostic benchmark, not part of the canonical 138
pair result.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"
MODEL = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def load_sw():
    path = ROOT / "repro_candidate_138" / "src" / "05_sw_bted.py"
    spec = importlib.util.spec_from_file_location("observable_sw", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module); return module


def tree_text(tree: dict, sections: dict, swap: bool = False) -> str:
    order = ["Context", "Problem", "Solution", "Theory", "Deliverables", "Methodology", "Timeline", "References"]
    if swap:
        order[1], order[3] = order[3], order[1]
        order[2], order[3] = order[3], order[2]
    return tree.get("label", "") + " " + " ".join(sections.get(k, "") for k in order)


def main() -> None:
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    texts = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(DATA / "pairs.csv")
    sources = pairs[pairs["type"] == "Type_A"].iloc[:20]
    sw = load_sw()
    model = SentenceTransformer(str(MODEL))
    cost = sw.SWCostModel(alpha=1.0, beta={"T2": 0.0, "T3": 0.9, "T4": 1.0}, cso_graph=None, max_depth=19)
    rows = []
    for _, pair in sources.iterrows():
        doc = str(pair["doc_a"])
        original = trees[doc]
        perturbed = copy.deepcopy(original)
        d2 = next(c for c in perturbed["children"] if c.get("schema_class") == "D2_FUNCTIONAL")
        d3 = next(c for c in perturbed["children"] if c.get("schema_class") == "D3_TECHNICAL_REALIZATION")
        d2["children"], d3["children"] = d3["children"], d2["children"]
        a = sw.CapstoneNode.from_dict(original); b = sw.CapstoneNode.from_dict(perturbed)
        structural = sw.normalize_similarity(a, b, cost)
        text_a, text_b = tree_text(original, texts[doc]), tree_text(perturbed, texts[doc], swap=True)
        ea, eb = model.encode([text_a, text_b], show_progress_bar=False, normalize_embeddings=False)
        cosine = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
        rows.append({"doc_id": doc, "label": 0, "text_changed": text_a != text_b, "structural_similarity": structural, "embedding_cosine": cosine})
    frame = pd.DataFrame(rows)
    cutoff = 0.45
    result = {
        "n_pairs": len(frame),
        "cutoff": cutoff,
        "model": str(MODEL),
        "construction": "swap D2/D3 child content in the tree and swap serialized Problem/Theory order in embedding input; labels are controlled negatives",
        "structural": {"accuracy": float(np.mean(frame.structural_similarity < cutoff)), "mean": float(frame.structural_similarity.mean()), "min": float(frame.structural_similarity.min()), "max": float(frame.structural_similarity.max())},
        "embedding": {"accuracy": float(np.mean(frame.embedding_cosine < cutoff)), "mean": float(frame.embedding_cosine.mean()), "min": float(frame.embedding_cosine.min()), "max": float(frame.embedding_cosine.max())},
        "all_text_inputs_changed": bool(frame.text_changed.all()),
        "rows": rows,
    }
    frame.to_csv(OUT / "observable_structural_perturbation_20.csv", index=False)
    (OUT / "observable_structural_perturbation_20.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "reports" / "OBSERVABLE_STRUCTURAL_PERTURBATION_20.md").write_text(
        "# Observable structural perturbation audit — 20 pairs\n\n"
        f"Construction: `{result['construction']}`. All text inputs changed: `{result['all_text_inputs_changed']}`.\n\n"
        f"SW-BTED structural-only accuracy at threshold `{cutoff}`: **{result['structural']['accuracy']:.1%}**; mean similarity `{result['structural']['mean']:.4f}`.\n\n"
        f"MiniLM full-text accuracy at the same threshold: **{result['embedding']['accuracy']:.1%}**; mean cosine `{result['embedding']['mean']:.4f}`.\n\n"
        "This is a secondary controlled diagnostic with constructed negative labels, not a replacement for the canonical real-only benchmark.", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
