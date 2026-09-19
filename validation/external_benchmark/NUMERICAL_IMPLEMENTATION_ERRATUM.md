# Gate-B pre-contrast numerical implementation erratum

The first full Rsolnp integration pass completed all 858 participant/model MAP
optimizations with convergence code zero. Its preregistered `numDeriv` Hessian
step rejected 134 fits before aggregation.

Diagnosis showed that `numDeriv::hessian(method="Richardson")` uses a default
relative step `d=0.1`. For valid MAP learning rates near the published upper
bound, the derivative probe evaluates the Beta-prior objective outside `[0,1]`
and returns `Inf/NaN`. For example, a valid `alpha=0.91065` was probed above one.

No aggregate primary contrast, bootstrap interval, or model ranking was
computed before this diagnosis and correction. The correction keeps every
Rsolnp MAP unchanged and computes the exact float64 autodiff Hessian of the same
negative log posterior in the same parameterization. Positive definiteness
remains mandatory; no jitter or eigenvalue flooring is allowed.

This correction does not alter the data, models, priors, likelihood, optimizer,
estimands, uncertainty procedure, or frozen pass/partial/fail thresholds.
