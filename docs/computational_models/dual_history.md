# Dual history (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H2+H3 operationalization.

## Variables
Immediate state (X_t); newest action/outcome (a_{t-1},r_{t-1}); separate traces (K_t,R_t).

## Equations
(K_t=.7K_{t-1}+a_{t-1}), (R_t=.7R_{t-1}+r_{t-1}), and
(D_t=\beta_0+\beta_X^TX_t+\beta_a a_{t-1}+\beta_KK_t+\beta_r r_{t-1}+\beta_RR_t+\beta_mm_t).

## History and initialization
Oldest-to-newest; continue +1, disengage -1; both traces initialize at zero; empty traces are missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under the same frozen coefficients.

## Fixed constants
Both history decays are exactly 0.7.
