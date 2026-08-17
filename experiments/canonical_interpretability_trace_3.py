"""Generate provenance-safe per-domain APTED traces for three canonical cases."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import apted

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "interpretability"
OUT.mkdir(parents=True, exist_ok=True)
DOMAINS = ["D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"]
CASES = [
    {"case": "A", "doc_a": "SU26SE102", "doc_b": "SU26SE102_plag", "label": 1,
     "role": "positive plagiarism pair"},
    {"case": "B", "doc_a": "SP26SE068", "doc_b": "SU26SE063", "label": 0,
     "role": "SBERT false positive; SW-BTED correct"},
    {"case": "C", "doc_a": "SP26SE122", "doc_b": "SP26SE055", "label": 0,
     "role": "negative Type_C pair; both correct"},
]


def load_sw():
    spec = importlib.util.spec_from_file_location("sw_trace", ROOT / "repro_candidate_138" / "src" / "05_sw_bted.py")
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); return mod


def nodes(n):
    yield n
    for child in n.children:
        yield from nodes(child)


def text(n):
    return (n.feature_label or n.normalized_text or n.raw_text or n.label or "").strip()


def trace_pair(sw, a, b, model):
    cache_r, cache_d, cache_i = {}, {}, {}
    def rename(u, v):
        key = (id(u), id(v)); cache_r.setdefault(key, model.w_rep(u, v)); return cache_r[key]
    def delete(u):
        key = id(u); cache_d.setdefault(key, model.w_del(u)); return cache_d[key]
    def insert(v):
        key = id(v); cache_i.setdefault(key, model.w_ins(v)); return cache_i[key]
    config = apted.Config(); config.rename, config.delete, config.insert = rename, delete, insert
    by_a, by_b = {c.schema_class: c for c in a.children}, {c.schema_class: c for c in b.children}
    total = 0.0; domains = []
    for domain in DOMAINS:
        ca, cb = by_a.get(domain), by_b.get(domain)
        mapping = []
        if ca and cb:
            runner = apted.APTED(ca, cb, config)
            distance = float(runner.compute_edit_distance())
            mapping = runner.compute_edit_mapping()
        elif ca:
            distance = float(sum(model.w_del(n) for n in nodes(ca)))
            mapping = [(n, None) for n in nodes(ca)]
        elif cb:
            distance = float(sum(model.w_ins(n) for n in nodes(cb)))
            mapping = [(None, n) for n in nodes(cb)]
        else:
            distance = 0.0
        total += distance
        denom = sum(model.w_del(n) for n in nodes(ca)) + sum(model.w_ins(n) for n in nodes(cb)) if ca and cb else distance
        similarity = 1.0 - distance / denom if denom else 1.0
        replacements, deletions, insertions, examples = 0, 0, 0, []
        for u, v in mapping:
            if u is None: insertions += 1
            elif v is None: deletions += 1
            else:
                if u.label != v.label or text(u) != text(v):
                    replacements += 1
                    if len(examples) < 5: examples.append({"a": text(u), "b": text(v), "depth": u.depth})
        domains.append({"domain": domain, "distance": distance, "max_cost": denom,
                        "normalized_similarity": similarity, "mapping_count": len(mapping),
                        "replacements": replacements, "deletions": deletions,
                        "insertions": insertions, "examples": examples})
    max_cost = sum(model.w_del(n) for n in nodes(a)) + sum(model.w_ins(n) for n in nodes(b))
    normalized = total / max_cost if max_cost else 0.0
    structural = 0.0 if normalized > model.max_edit_budget_ratio else 1.0 - normalized / model.max_edit_budget_ratio
    return {"total_distance": total, "max_possible_cost": max_cost, "normalized_cost": normalized,
            "structural_similarity": structural, "domains": domains,
            "nodes_a": sum(1 for _ in nodes(a)), "nodes_b": sum(1 for _ in nodes(b))}


def main():
    sw = load_sw()
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    model = sw.SWCostModel(alpha=.8, beta={"T2": 0.0, "T3": .9, "T4": .8}, cso_graph=None, max_depth=19)
    results = []
    for case in CASES:
        a, b = sw.CapstoneNode.from_dict(trees[case["doc_a"]]), sw.CapstoneNode.from_dict(trees[case["doc_b"]])
        result = {**case, "trace": trace_pair(sw, a, b, model)}
        results.append(result)
    (OUT / "canonical_interpretability_trace_3.json").write_text(json.dumps({"cases": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "canonical_interpretability_domain_scores.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["case", "doc_a", "doc_b", "label", "domain", "normalized_similarity", "distance", "replacements", "deletions", "insertions"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for case in results:
            for row in case["trace"]["domains"]:
                writer.writerow({"case": case["case"], "doc_a": case["doc_a"], "doc_b": case["doc_b"], "label": case["label"], **{k: row[k] for k in fields[5:]}, "domain": row["domain"]})
    lines = ["# Canonical Interpretability Trace (3 Cases)", "", "The traces use the canonical four-layer trees, SW-BTED structural-only cost parameters, and APTED mappings.", ""]
    for case in results:
        t = case["trace"]
        lines += [f"## Case {case['case']}: `{case['doc_a']}–{case['doc_b']}`", f"Role: {case['role']}; label={case['label']}.", "", f"Nodes: {t['nodes_a']} vs {t['nodes_b']}; structural similarity: `{t['structural_similarity']:.4f}`.", "", "| Domain | Similarity | Replacements | Deletes | Inserts |", "|---|---:|---:|---:|---:|"]
        for d in t["domains"]: lines.append(f"| {d['domain']} | {d['normalized_similarity']:.4f} | {d['replacements']} | {d['deletions']} | {d['insertions']} |")
        lines += ["", "Representative replacements:"]
        for d in t["domains"]:
            for ex in d["examples"][:2]: lines.append(f"- {d['domain']} (T{ex['depth']}): `{ex['a']}` → `{ex['b']}`")
        lines.append("")
    (OUT / "CANONICAL_INTERPRETABILITY_TRACE_3.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": [{"case": x["case"], "pair": [x["doc_a"], x["doc_b"]], "structural_similarity": x["trace"]["structural_similarity"]} for x in results]}, indent=2))


if __name__ == "__main__": main()
