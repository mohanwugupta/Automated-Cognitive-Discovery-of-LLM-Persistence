# Intercept (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen null model.

## Variables
(D_t) is the signed semantic persistence logit; no cognitive regressors are used.

## Equations
(D_t=\beta_0+\beta_m m_t), with response mapping (m_t) retained only as nuisance.

## History and initialization
History is unused; the constant is initialized from training outcomes.

## Fitting and prediction
Use the shared fitting and sharing contract in `README.md`.

## Counterfactual
With mapping held fixed, \(\Delta D^{CF}=0\).

## Fixed constants
No cognitive constants.
