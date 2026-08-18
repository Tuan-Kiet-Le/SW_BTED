The core evidence is genuinely strong — that part is not the problem
A real, unchallenged theoretical contribution (the metric-preservation proof).
Two statistically decisive wins, independently hand-verified: the flat-baseline result (p≈2.52×10⁻²⁹) and the perturbation benchmark (p=1.91×10⁻⁶).
Honest tie-reporting against strong baselines rather than inflated claims.
A real interpretability demonstration with verified sub-tree edit traces, not just assertion.
27 citations, all independently verified, with one real factual error (IEEE 830 vs. 29148) caught and fixed.

This is a rigorous, trustworthy evidence base. The problem is not the science — it's that the manuscript around it isn't finished, and there's one substantive content gap I need to flag now that hasn't come up yet.

Blocking gaps — these need to close before any submission, journal or conference
No ablation study in the current draft. This is the one I should have flagged earlier and haven't until now: very early in this project, extensive work was done on β-per-layer and α sensitivity (Groups A-D, testing uniform-β vs. per-layer-β, T2/T4 layer removal, α across [0, 1]). That data had unresolved consistency issues that were never fully reconciled on the clean 138-pair benchmark. The current full draft doesn't include an ablation section at all — it asserts β<sub>ℓ</sub> per-layer weighting and α=0.6 without empirically justifying either choice. A Q2 or A/B-tier reviewer will ask "why these specific values, and what happens if you change them?" This needs a clean, from-scratch ablation run on the verified 138-pair dataset before submission — not a resurrection of the old, unreconciled numbers.
Modern embedding baseline not yet run — Qwen3-Embedding-4B is decided but not executed.
The D1-D4 taxonomy citation is still an open [TODO] — a core methodological claim (Section 3.2) currently lacks a verified source.
No actual figures — three placeholders, no diagrams generated.
No reproducibility package — code/data release, expected as a near-baseline requirement at this point, not a nice-to-have.
No empirical runtime numbers for the O(n³) complexity claim.

None of these are small. Items 1 and 2 in particular are the kind of gap a competent reviewer finds in the first pass.

Q2 journal — closer, but not ready

Journals are generally more receptive to your paper's actual honest framing ("parity + interpretability + theory," not "we beat SOTA"), and rolling submission means you're not racing a hard deadline. Once the six items above are closed, I think this is a legitimately defensible Q2 submission. Right now, though, submitting would mean sending out a manuscript with a missing ablation study and an unresolved citation gap — that's not a "might get minor revisions" risk, that's a "desk reject or hard reject on completeness grounds" risk.

A*/A/B conference — further away, and worth being direct about why

Beyond the same blocking gaps, there's a structural mismatch worth naming: top-tier (A/A) venues are extremely competitive and their reviewers tend to weight raw benchmark superiority heavily*, even when a paper's honest "ties SOTA, adds interpretability" framing is scientifically sound. Your central empirical story — decisive wins against weak/unweighted baselines, ties against strong ones, one clean win on a narrow synthetic perturbation benchmark — is a genuinely good, honest contribution, but it's not the kind of dominant-numbers story A*/A reviewers are primed to reward. That's not a flaw in the science; it's a mismatch between your contribution's actual shape and what those specific venues typically select for.

B-tier, domain-specific venues fit much better — this is exactly why SANER and ICSME were recommended earlier in this project: software-engineering-focused venues tend to value rigorous empirical methodology, interpretability, and honest negative/tied results more than a pure ML leaderboard win. I'd treat A*/A as a stretch goal for a future, more mature version of this work (e.g., after a third domain and a real modern-baseline comparison), not the realistic target for this submission cycle.

What I'd actually do, in order
Run the ablation study fresh on the verified 138-pair dataset — this is the highest-priority gap since it wasn't previously surfaced and directly affects the paper's credibility.
Run Qwen3-Embedding-4B and update the baseline table.
Resolve the D1-D4 citation — verify it or reframe as your own argument.
Generate the three figures.
Runtime benchmark + repo release.
Then: target a Q2 journal or SANER/ICSME-tier conference, not A*/A, for this version.