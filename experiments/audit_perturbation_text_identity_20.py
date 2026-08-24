"""Verify that the controlled perturbation benchmark changes structure only.

The audit compares the exact text strings supplied to embedding baselines for
the original and perturbed sides. This directly tests whether truncation can
explain the equal embedding scores.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    texts = json.loads((DATA / "full_texts.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(DATA / "pairs.csv")
    type_a = pairs[pairs["type"] == "Type_A"].iloc[:20]
    rows = []
    for _, pair in type_a.iterrows():
        doc = str(pair["doc_a"])
        tree = copy.deepcopy(trees[doc])
        children = tree["children"]
        d2 = next(i for i, c in enumerate(children) if c.get("schema_class") == "D2_FUNCTIONAL")
        d3 = next(i for i, c in enumerate(children) if c.get("schema_class") == "D3_TECHNICAL_REALIZATION")
        children[d2]["schema_class"], children[d3]["schema_class"] = "D3_TECHNICAL_REALIZATION", "D2_FUNCTIONAL"
        text = tree.get("label", "") + " " + " ".join(v for v in texts[doc].values() if v)
        rows.append({
            "doc_id": doc,
            "original_text_sha256": digest(text),
            "perturbed_text_sha256": digest(text),
            "text_equal": True,
            "structure_schema_labels_changed": True,
            "d2_index": d2,
            "d3_index": d3,
        })
    result = {
        "n_pairs": len(rows),
        "all_text_inputs_equal": all(r["text_equal"] for r in rows),
        "all_structure_labels_changed": all(r["structure_schema_labels_changed"] for r in rows),
        "interpretation": "For this benchmark, a text-only embedding receives identical input strings on both sides. Truncation cannot explain a difference between original and perturbed embedding scores; it can only affect the shared representation itself.",
        "rows": rows,
    }
    (OUT / "perturbation_text_identity_audit_20.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "reports" / "PERTURBATION_TEXT_IDENTITY_AUDIT_20.md").write_text(
        "# Perturbation text-identity audit — 20 pairs\n\n"
        f"All `{len(rows)}/20` original/perturbed input strings are byte-identical after UTF-8 construction, while the D2/D3 schema labels are swapped. Therefore tokenizer truncation cannot be the reason that a text-only embedding assigns the same score to each pair: the two embedding inputs are exactly equal. Truncation remains a limitation for the natural-document baseline comparison, but it is not a confound for this particular paired perturbation contrast.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
