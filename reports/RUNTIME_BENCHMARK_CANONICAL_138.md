# Canonical SW-BTED Runtime Benchmark

Date: 2026-08-14  
Dataset: 138 real-only pairs, four-layer trees

## Protocol

The benchmark timed `normalize_similarity` from the canonical four-layer
implementation using `SWCostModel(alpha=0.8, beta_T2=0.0, beta_T3=0.9,
beta_T4=0.8)`. Five pairs were used as warm-up; one timed call was then made
for each of the 138 canonical pairs. The timing includes APTED alignment and
Python-side SW-BTED scoring, but excludes parsing, model loading, and embedding
inference. The benchmark is a per-comparison latency measurement, not a claim
about end-to-end service throughput.

Environment: Python 3.13.5, Windows 11, AMD64 Family 25 Model 68, 16 logical
CPUs.

## Results

| Statistic | Runtime |
|---|---:|
| Number of timed pairs | 138 |
| Total alignment/scoring time | 2.1312 s |
| Mean | 15.44 ms/pair |
| Median | 17.94 ms/pair |
| 95th percentile | 23.52 ms/pair |
| Minimum | 1.61 ms/pair |
| Maximum | 56.40 ms/pair |

Runtime increased with tree size in the observed dataset:

| Size group | Pair count | Total nodes | Mean | Median | P95 |
|---|---:|---:|---:|---:|---:|
| Smallest quartile | 34 | 46–127 | 6.53 ms | 6.88 ms | 8.21 ms |
| Middle half | 69 | 127–172 | 17.02 ms | 17.93 ms | 23.77 ms |
| Largest quartile | 35 | 172 | 20.99 ms | 18.88 ms | 26.05 ms |

## Reproducibility artifacts

- Script: `experiments/runtime_benchmark_canonical_138.py`
- Pair-level timings: `reports/audit/runtime_benchmark_canonical_138.csv`
- Machine-readable summary: `reports/audit/runtime_benchmark_canonical_138.json`

## Interpretation

On this machine, the structural alignment component is fast enough for offline
138-pair evaluation and small-batch comparison. The result should be reported
as empirical evidence for this implementation and hardware, while the
theoretical `O(n^3)` bound remains the worst-case complexity statement. Full
end-to-end latency will be higher when parsing and embedding inference are
included.
