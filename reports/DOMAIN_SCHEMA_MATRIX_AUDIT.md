# Domain schema penalty matrix audit

| | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| D1 | 0.0 | 0.8 | 0.9 | 0.9 |
| D2 | 0.8 | 0.0 | 0.5 | 0.7 |
| D3 | 0.9 | 0.5 | 0.0 | 0.6 |
| D4 | 0.9 | 0.7 | 0.6 | 0.0 |

Symmetric: `True`; diagonal identity: `True`; triangle inequality: `True`.

The domain penalty matrix satisfies the finite-class metric checks, but the complete implementation still uses cosine-derived T3 distance; therefore the manuscript retains a conditional node-level proposition rather than claiming metricity of the implemented cost.