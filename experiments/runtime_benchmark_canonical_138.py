"""Measure SW-BTED runtime on the canonical four-layer 138-pair dataset."""
from __future__ import annotations

import csv
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repro_candidate_138" / "data" / "dataset"
OUT = ROOT / "reports" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def load_sw():
    spec = importlib.util.spec_from_file_location("sw_runtime", ROOT / "repro_candidate_138" / "src" / "05_sw_bted.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in node.children)


def main():
    sw = load_sw()
    trees = json.loads((DATA / "trees_section.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((DATA / "pairs.csv").open(encoding="utf-8-sig", newline="")))
    nodes = {key: sw.CapstoneNode.from_dict(value) for key, value in trees.items()}
    model = sw.SWCostModel(alpha=.8, beta={"T2": 0.0, "T3": .9, "T4": .8}, cso_graph=None, max_depth=19)

    # Warm-up removes one-time import/object-allocation effects from the table.
    for row in rows[:5]:
        sw.normalize_similarity(nodes[row["doc_a"]], nodes[row["doc_b"]], model)

    measurements = []
    for row in rows:
        a, b = nodes[row["doc_a"]], nodes[row["doc_b"]]
        start = time.perf_counter_ns()
        score = sw.normalize_similarity(a, b, model)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        measurements.append({
            "doc_a": row["doc_a"], "doc_b": row["doc_b"], "label": int(row["label"]),
            "nodes_a": count_nodes(a), "nodes_b": count_nodes(b),
            "nodes_total": count_nodes(a) + count_nodes(b),
            "elapsed_ms": elapsed_ms, "similarity": float(score),
        })

    bins = []
    values = sorted(measurements, key=lambda x: x["nodes_total"])
    for label, subset in (
        ("smallest quartile", values[:len(values)//4]),
        ("middle half", values[len(values)//4:3*len(values)//4]),
        ("largest quartile", values[3*len(values)//4:]),
    ):
        bins.append({
            "size_bin": label, "n_pairs": len(subset),
            "nodes_total_min": min(x["nodes_total"] for x in subset),
            "nodes_total_max": max(x["nodes_total"] for x in subset),
            "mean_ms": statistics.mean(x["elapsed_ms"] for x in subset),
            "median_ms": statistics.median(x["elapsed_ms"] for x in subset),
            "p95_ms": float(np.percentile([x["elapsed_ms"] for x in subset], 95)),
        })

    summary = {
        "protocol": {"dataset": "canonical four-layer 138 real-only pairs", "warmup_pairs": 5,
                     "timed_calls": len(measurements), "function": "normalize_similarity",
                     "model": "SWCostModel(alpha=0.8, beta_T2=0.0, beta_T3=0.9, beta_T4=0.8)",
                     "timing_unit": "milliseconds per pair", "includes": "APTED alignment and Python scoring",
                     "excludes": "model loading, parsing, and embedding inference"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "cpu_count": __import__("os").cpu_count()},
        "overall": {"n_pairs": len(measurements), "mean_ms": statistics.mean(x["elapsed_ms"] for x in measurements),
                    "median_ms": statistics.median(x["elapsed_ms"] for x in measurements),
                    "p95_ms": float(np.percentile([x["elapsed_ms"] for x in measurements], 95)),
                    "min_ms": min(x["elapsed_ms"] for x in measurements),
                    "max_ms": max(x["elapsed_ms"] for x in measurements),
                    "total_seconds": sum(x["elapsed_ms"] for x in measurements) / 1000},
        "size_bins": bins, "pair_measurements": measurements,
    }
    (OUT / "runtime_benchmark_canonical_138.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "runtime_benchmark_canonical_138.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader(); writer.writerows(measurements)
    print(json.dumps({"environment": summary["environment"], "overall": summary["overall"], "size_bins": bins}, indent=2))


if __name__ == "__main__":
    main()
