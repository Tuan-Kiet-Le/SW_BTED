# Current project structure

```text
SW_BTED_v2/
├── src/                         # canonical four-layer implementation
├── tests/                       # regression tests
├── experiments/                 # canonical 138-pair runners
│   └── archive/                 # historical/exploratory runners
├── reports/                     # canonical metrics and audit evidence
│   ├── audit/                   # machine-readable and provenance artifacts
│   └── interpretability/        # canonical structural traces
├── draft/                       # current manuscript
├── docs/
│   ├── submission_figures/      # manuscript/review figures
│   ├── project_management/      # plans and task specifications
│   ├── feedback/                # external feedback
│   ├── research_notes/          # technical notes
│   └── archive/generated/       # superseded generated HTML/plots
├── reproducibility/             # reproduction instructions
├── submission_neutral/          # public artifact checklist
├── config.yaml                  # canonical configuration
└── Agents.md                    # agent working rules
```

The primary workflow is `src/` → `experiments/` → `reports/`. The folders
`data/`, `datasets/`, `data_results/`, `kaggle/`, and `results/` remain local
working areas and are excluded from the public repository.
