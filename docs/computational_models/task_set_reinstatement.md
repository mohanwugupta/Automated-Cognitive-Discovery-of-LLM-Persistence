# Task-set reinstatement (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H7 interaction operationalization.

## Variables
Immediate state (X_t), goal continuity (G_t), action trace (K_t), and (K_tG_t).

## Equations
(D_t=\beta_0+\beta_X^TX_t+\beta_GG_t+\beta_KK_t+\beta_{KG}K_tG_t+\beta_mm_t).

## History and initialization
(K_t=.7K_{t-1}+a_{t-1}), oldest-to-newest, (K_0=0); empty history is missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
Action decay is exactly 0.7.
