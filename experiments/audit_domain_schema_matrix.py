"""Check symmetry, identity, and triangle inequality of the canonical D2 matrix."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repro_candidate_138" / "src" / "05_sw_bted.py"
OUT = ROOT / "reports" / "audit"


def main() -> None:
    spec = importlib.util.spec_from_file_location("canonical_sw", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    labels = [
        "D1_BUSINESS_CONTEXT",
        "D2_FUNCTIONAL",
        "D3_TECHNICAL_REALIZATION",
        "D4_EXECUTION_PLANNING",
    ]
    matrix = [[float(module.DOMAIN_SCHEMA_DIST.get((a, b), 1.0)) for b in labels] for a in labels]
    symmetry = all(matrix[i][j] == matrix[j][i] for i in range(4) for j in range(4))
    identity = all(matrix[i][i] == 0.0 for i in range(4))
    violations = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            for k, c in enumerate(labels):
                if matrix[i][k] > matrix[i][j] + matrix[j][k] + 1e-12:
                    violations.append({"a": a, "b": b, "c": c, "lhs": matrix[i][k], "rhs": matrix[i][j] + matrix[j][k]})
    result = {
        "labels": labels,
        "matrix": matrix,
        "symmetric": symmetry,
        "identity_of_indiscernibles_on_schema_classes": identity,
        "triangle_inequality_holds": not violations,
        "violations": violations,
        "interpretation": "The domain penalty matrix satisfies the finite-class metric checks, but the complete implementation still uses cosine-derived T3 distance; therefore the manuscript retains a conditional node-level proposition rather than claiming metricity of the implemented cost.",
    }
    (OUT / "domain_schema_matrix_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Domain schema penalty matrix audit", "", "| | D1 | D2 | D3 | D4 |", "|---|---:|---:|---:|---:|"]
    for label, row in zip(["D1", "D2", "D3", "D4"], matrix):
        lines.append(f"| {label} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
    lines += ["", f"Symmetric: `{symmetry}`; diagonal identity: `{identity}`; triangle inequality: `{not violations}`.", "", result["interpretation"]]
    (ROOT / "reports" / "DOMAIN_SCHEMA_MATRIX_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
