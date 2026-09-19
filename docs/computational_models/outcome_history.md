# Outcome history (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H3 operationalization.

## Variables
Immediate state (X_t), newest outcome (r_{t-1}), and outcome trace (R_t).

## Equations
(R_t=0.7R_{t-1}+r_{t-1}); (D_t=\beta_0+\beta_X^TX_t+\beta_1r_{t-1}+\beta_RR_t+\beta_mm_t).

## History and initialization
Outcomes are ordered oldest-to-newest, numeric signed evidence; (R_0=0). Empty history is missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
(\Delta D^{CF}=D(source)-D(base)) with one frozen fit.

## Fixed constants
History decay is exactly 0.7.
