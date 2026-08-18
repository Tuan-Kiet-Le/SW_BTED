# RAG 180-scope rerun and 138-pair extraction

Date: 2026-08-14

## Execution

The historical-style inner-train protocol was rerun read-only using:

`D:/FPT/Semester_8/RAG_Research/Data/dataset/`

The run used all 180 rows during outer/inner CV, then extracted the 138 real-only rows by name using `plag_regen_sections.json`. Outputs were written to the current workspace:

`reports/audit/rag_main_scope_180_b1_b5.json`

## Result

| Method | Full 180-pair OOF F1 | Name-matched 138 subset F1 |
|---|---:|---:|
| TF-IDF | 0.9938 | 0.9870 |
| Section Cosine | 0.8000 | 0.5926 |

These values still do not reproduce the historical audit values:

```text
Historical audit: TF-IDF = 0.4364, Section Cosine = 0.4081
```

As a second check, the archived `RAG_Research/results/pair_similarities.csv` has 180 rows. After the same explicit name-based 138-pair filter, its stored predictions give:

```text
TF-IDF F1 = 0.9870
Section Cosine F1 = 0.6667
```

Therefore neither the scope-matched rerun nor the archived per-pair result file supports the historical `0.4364` / `0.4081` pair of numbers.

## Conclusion

The manuscript's TF-IDF and Section Cosine values are currently unverified historical claims. The original generating artifact for those exact values has not been recovered. They should not be retained as if they were reproducible canonical 138-pair results.

The cleanest scientific action is now to choose and document one frozen baseline definition. If the paper keeps the current implementations, the manuscript must use the clean suite values and revise its claims. If it wants to preserve the old values, it needs to recover the exact source artifact and explain the discrepancy; current evidence does not support doing so.
