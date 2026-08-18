# Remaining baseline-suite provenance reconciliation

Date: 2026-08-14

## Finding

The remaining baseline rows are not yet safe to replace in the manuscript. Multiple reports use different input scopes and/or harnesses:

| Method | Manuscript Table 1 | Reproduction/candidate output | Status |
|---|---:|---:|---|
| TF-IDF | 0.4364 ± 0.0162 | 0.9223 ± 0.1083 in `REPRODUCTION_RUN_REPORT_138.md` | unresolved protocol/source drift |
| Standard TED | 0.4364 ± 0.0162 | 0.3425 ± 0.1922 | unresolved protocol/source drift |
| Section Cosine | 0.4081 ± 0.0548 | 0.5015 ± 0.2323 | unresolved protocol/source drift |
| pq-Gram | 0.9479 ± 0.0478 | 0.9579 ± 0.0386 | unresolved protocol/source drift |
| Genuine Flat Domain SBERT | 0.4314 | separate novelty-test output | likely distinct, needs input/protocol lock |

The recovered candidate report explicitly states that its harness mixes alpha/configuration search and does not reproduce the manuscript's fixed structural-only presentation. It therefore cannot be used to silently replace the Table 1 values.

## What is already safe

The following rows now have clean, directly regenerated provenance:

- SW-BTED Structural-Only: `0.9498 ± 0.0253`.
- Full-document SBERT: `0.9867 ± 0.0267`.
- BGE-small: `0.9882 ± 0.0235`.
- MPNet: `0.9882 ± 0.0235`.

For these rows, the clean prediction vectors and evaluation JSON are available under `reports/audit/`.

## Why the remaining rows cannot be updated yet

The repository contains at least three incompatible result families:

1. Manuscript Table 1 values (`0.4364`, `0.4081`, `0.9479`).
2. Candidate reproduction values (`0.9223`, `0.5015`, `0.9579`).
3. Other audit/novelty reports with additional values and different input scopes.

The reports do not yet establish, for each remaining baseline, all of the following in one chain:

- exact source script;
- exact tree/text input;
- exact pair filter and pair ordering;
- score construction;
- threshold grid and train-fold selection;
- per-pair prediction vector;
- output file hash.

## Decision

Do not edit the remaining Table 1 rows yet. The manuscript's natural-document conclusion should remain provisional until these baselines are regenerated from one frozen suite.

## Next execution step

Create a clean baseline-suite harness that implements TF-IDF, Standard TED, pq-Gram, Section Cosine, and Genuine Flat Domain SBERT over the same canonical 138-pair rows, with one shared evaluation protocol and per-pair outputs. Then compare its results against both the manuscript values and candidate outputs. Only after that comparison should the manuscript table be updated.
