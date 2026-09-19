# Nonblocking boundary amendment

Status: **OWNER-FROZEN — 2026-09-19**

This amendment changes only the downstream consequence of the first Gate-B
execution. It does not change the v1 data, models, priors, likelihood, fitted
parameters, marginal-likelihood endpoint, thresholds, or unresolved result.

## Split decision

- `implementation_equivalence: pass`
- `benchmark_claim_status: unresolved`
- `benchmark_claim_permission: stop`
- `pipeline_permission: continue`

All 858 MAP optimizations converged, and the independent R and Python
log-posterior implementations agreed to a maximum absolute error of
`5.116e-13`. This supports consistency of the reinforcement-learning equations,
likelihood, and priors. It does not resolve the marginal-likelihood approximation
for six constrained solutions at the active `tau=1` boundary.

External-benchmark numerical irregularities therefore block claims about the
benchmark result but do not block the subsequent Qwen/transfer pipeline once
model implementation equivalence has passed.

## Future v2 ordering

If a claim-bearing benchmark result is needed:

1. Recover and use the authors' actual marginal-likelihood implementation if
   possible.
2. Otherwise preregister a genuinely boundary-aware integration rule or an
   independently validated numerical marginal-likelihood estimator.
3. Apply the selected rule prospectively to **all** constrained boundary
   solutions, not only the six currently observed cases.

Until one of the first two options is frozen, v2 is `not_selected`. No aggregate
v1 contrasts, bootstrap intervals, or rankings may be inspected to choose the
method.

The following are prohibited: Hessian jitter, eigenvalue flooring, absolute
determinants, silently excluding participants, or simply dropping `tau` from
the Hessian at the boundary. A constrained optimum may have a nonzero gradient
in the normal direction, so an interior Gaussian approximation over a reduced
Hessian is not automatically the same marginal-likelihood problem.
