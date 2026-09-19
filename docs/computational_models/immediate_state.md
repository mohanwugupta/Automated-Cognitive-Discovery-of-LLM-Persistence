# Immediate state (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen baseline.

## Variables
(X_t=[V_c,V_d,C,P,S,U,I,Q,E,G]): continuation value, disengagement value,
continuation cost, progress evidence, success evidence, uncertainty, prior
investment, controllability, environmental stability, and goal continuity.

## Equations
(D_t=\beta_0+\beta_X^T X_t+\beta_m m_t).

## History and initialization
History is unused. Missing factors follow the availability convention in `README.md`.

## Fitting and prediction
Continuous ridge regression and sharing rules are defined in `README.md`.

## Counterfactual
(\Delta D^{CF}=\beta_X^T(X_{source}-X_{base})), mapping held fixed.

## Fixed constants
No history decay.
