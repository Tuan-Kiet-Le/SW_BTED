"""Report threshold sensitivity for the secondary observable perturbation audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "audit"


def main():
    frame = pd.read_csv(AUDIT / "observable_structural_perturbation_20.csv")
    thresholds = [round(x, 2) for x in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    rows = []
    for threshold in thresholds:
        rows.append({
            "threshold": threshold,
            "structural_rejected": int((frame["structural_similarity"] < threshold).sum()),
            "embedding_rejected": int((frame["embedding_cosine"] < threshold).sum()),
        })
    output = {
        "n_pairs": int(len(frame)),
        "source": "observable_structural_perturbation_20.csv",
        "thresholds_are": "sensitivity analysis only; no threshold is claimed as a prespecified test threshold",
        "score_summary": {
            "structural_mean": float(frame["structural_similarity"].mean()),
            "structural_min": float(frame["structural_similarity"].min()),
            "structural_max": float(frame["structural_similarity"].max()),
            "embedding_mean": float(frame["embedding_cosine"].mean()),
            "embedding_min": float(frame["embedding_cosine"].min()),
            "embedding_max": float(frame["embedding_cosine"].max()),
        },
        "sensitivity": rows,
    }
    (AUDIT / "observable_threshold_sensitivity_20.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [
        "# Observable perturbation threshold sensitivity — 20 pairs", "",
        "This table is descriptive. The audit does not claim that 0.45 was prespecified or selected independently for this constructed set.", "",
        "| Cutoff | SW-BTED structural rejected | MiniLM rejected |", "|---:|---:|---:|",
    ]
    lines.extend(f"| {r['threshold']:.2f} | {r['structural_rejected']}/20 | {r['embedding_rejected']}/20 |" for r in rows)
    lines += ["", f"Structural similarity range: {output['score_summary']['structural_min']:.4f}–{output['score_summary']['structural_max']:.4f}; MiniLM cosine range: {output['score_summary']['embedding_min']:.4f}–{output['score_summary']['embedding_max']:.4f}."]
    (ROOT / "reports" / "OBSERVABLE_PERTURBATION_THRESHOLD_SENSITIVITY_20.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
