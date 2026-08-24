# Perturbation text-identity audit — 20 pairs

All `20/20` original/perturbed input strings are byte-identical after UTF-8 construction, while the D2/D3 schema labels are swapped. Therefore tokenizer truncation cannot be the reason that a text-only embedding assigns the same score to each pair: the two embedding inputs are exactly equal. Truncation remains a limitation for the natural-document baseline comparison, but it is not a confound for this particular paired perturbation contrast.
