# Requirements for a possible Gate-B v2

Status: `method_not_selected`

This file preregisters admissibility requirements, not a numerical estimator.
No v2 result may be produced until the exact method, implementation, validation
fixtures, tolerances, and random seeds are frozen in a new versioned config.

An admissible v2 must:

- first attempt to recover the authors' actual marginal-likelihood code and
  software versions;
- otherwise use a mathematically boundary-aware integration rule or an
  independently validated numerical marginal-likelihood estimator;
- use one prospective classification and calculation rule for every active
  lower or upper parameter boundary across every participant and model;
- retain all 143 owner-frozen participants and all six models;
- preserve the published likelihood, priors, constraints, MAP specification,
  primary contrasts, bootstrap, and ranking thresholds;
- validate interior solutions against ordinary exact Laplace calculations and
  boundary solutions against analytic or high-accuracy numerical fixtures;
- report estimator error, convergence, and sensitivity without choosing the
  estimator from the observed Gate-B contrasts.

An admissible v2 must not use:

- Hessian jitter;
- eigenvalue flooring;
- absolute determinants;
- participant exclusion; or
- automatic deletion of a boundary parameter from the Hessian.

Any estimator failure remains explicit.
