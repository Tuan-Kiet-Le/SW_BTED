# SW-BTED v2

Schema-Weighted Bounded Tree Edit Distance for interpretable structured
document similarity.

## Current scientific scope

- Four-layer representation: `Root → Domain → Intent → Terminology`.
- Primary benchmark: 138 real-only document pairs (38 positive, 100 negative).
- Frozen structural-only result: F1 `0.9498 ± 0.0253`.
- Clean baseline suite: TF-IDF F1 `0.9867`, Section Cosine F1 `0.6837`,
  Standard TED F1 `0.4364`, and Genuine Flat Domain SBERT F1 `0.4314`.
- The structural-perturbation benchmark is a controlled diagnostic, not a
  replacement for the natural-document benchmark.

## Project map

```text
src/                       canonical four-layer implementation
tests/                     regression tests
experiments/               canonical 138-pair runners
experiments/archive/       historical and exploratory runners
reports/                   canonical metrics and audit evidence
reports/audit/             machine-readable artifacts and provenance
reports/interpretability/  canonical structural traces
draft/                     current manuscript
docs/                      project notes, feedback, figures, and archive
reproducibility/           reproduction instructions
submission_neutral/        public artifact checklist
config.yaml                canonical configuration
Agents.md                  agent working rules
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the detailed
directory map and [`reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md`](reports/CANONICAL_SCIENTIFIC_MANIFEST_138.md)
for the frozen provenance record.

## Reproduction entry points

```powershell
python -m unittest discover -s tests -q
python experiments/final_canonical_results_138.py
python experiments/clean_baseline_suite_138.py
```

The evaluation scripts require the local dataset and model dependencies,
which are intentionally excluded from the public-safe repository. The local
working areas `data/`, `datasets/`, `data_results/`, `kaggle/`, and `results/`
are ignored by Git.

## Public repository policy

The repository contains source code, manuscript artifacts, figures, reports,
and reproducibility metadata. Raw/private datasets, model caches, Kaggle
working files, and historical result branches are not part of the canonical
public evidence package.
