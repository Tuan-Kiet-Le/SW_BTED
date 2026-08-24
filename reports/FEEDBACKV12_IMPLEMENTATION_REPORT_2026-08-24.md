# Feedback v12 implementation report

Date: 2026-08-24

## Completed changes

- Softened the Abstract motivation from an absolute tradeoff claim to a claim about common flat-embedding and unweighted-TED approaches.
- Softened the Introduction claim about what a single similarity score can explain.
- Renamed Section 5.6 to `Inspectable structural attribution case studies`.
- Replaced the deployment-style recommendation for Structural-Only with a benchmark-scoped operating-mode statement.
- Narrowed the Related Work wording about prior tree and kernel methods to the specific genre-grounded layer-weighting mechanism studied here.

## Verification

- Confirmed the old phrases `Existing approaches force a tradeoff`, `cannot answer either question`, `Interpretability case studies`, `We recommend Structural-Only`, and `without a mechanism for weighting different structural layers` no longer appear in the manuscript.
- `git diff --check` completed successfully; Git emitted only normal line-ending warnings.
- No experiment was rerun because all changes are claim-strength and presentation corrections.

## Freeze status

The scientific manuscript content is now ready to freeze. Remaining work is venue formatting, double-blind anonymization where required, final commit/tag creation, repository-pointer update, and manifest/artifact hash verification.
