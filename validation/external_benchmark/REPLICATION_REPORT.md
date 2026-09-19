# Gate-B external benchmark execution report

Gate B: **NOT EVALUABLE — OWNER REVIEW REQUIRED**  
Benchmark-claim permission: `stop`  
Downstream pipeline permission: `continue`  
Implementation equivalence: `pass`  
Evidence eligibility: `development_only`

## What completed

- The frozen Figshare-v3 factual dataset passed its contract: 143 participants, two 96-trial sessions, and no added filtering.
- All 858 participant/model Rsolnp MAP optimizations converged.
- 852 of 858 exact float64 Hessians were positive definite.
- The independent R and Python log-posteriors agreed to a maximum absolute error of 5.116e-13.

## Why classification stopped

6 gradual-model MAP estimates lie at the active upper boundary for tau and have a non-positive-definite unconstrained Hessian. The frozen benchmark requires a positive-definite Hessian and forbids eigenvalue flooring. A boundary-aware Laplace rule would therefore be a new analysis convention, not a mechanical repair.

| Participant | Model | tau | Minimum Hessian eigenvalue |
|---:|---|---:|---:|
| 22 | perseverance_gradual | 0.999998920315 | -0.796290528 |
| 55 | perseverance_gradual | 0.999998980182 | -12.2688727 |
| 55 | hybrid_gradual | 0.999998996503 | -13.3477352 |
| 71 | perseverance_gradual | 0.999998999985 | -0.271051442 |
| 71 | hybrid_gradual | 0.999998999968 | -0.352620585 |
| 105 | hybrid_gradual | 0.999998995107 | -2.37528735 |

## Results deliberately not computed

No primary contrast, bootstrap interval, mean-LML ranking, Gate-B pass/partial/fail label, or `results.csv` was produced. Computing any of these from only 137 participants, a jittered Hessian, an absolute determinant, or a dropped boundary parameter would violate the owner-frozen specification.

## Required owner decision

No decision is required to continue the Qwen/transfer pipeline. If the benchmark itself will support a scientific claim, first recover the authors' actual marginal-likelihood implementation. If that is unavailable, preregister a distinct v2 with one genuinely boundary-aware integration or independently validated numerical marginal-likelihood rule applying to every boundary solution before examining aggregate contrasts or rankings.

Hessian jitter, absolute determinants, participant exclusion, and simply dropping a boundary parameter are prohibited.

The complete participant/model diagnostics are in `fit_diagnostics.csv`. The earlier finite-difference issue and its pre-contrast correction are recorded in `NUMERICAL_IMPLEMENTATION_ERRATUM.md`.
