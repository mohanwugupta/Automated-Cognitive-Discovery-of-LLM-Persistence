# Choice perseveration (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H2 operationalization.

## Variables
Immediate-state (X_t), most recent semantic action (a_{t-1}\in\{-1,+1\}), and action trace (K_t).

## Equations
(K_t=0.7K_{t-1}+a_{t-1}); (D_t=\beta_0+\beta_X^TX_t+\beta_1a_{t-1}+\beta_KK_t+\beta_mm_t).

## History and initialization
Actions are ordered oldest-to-newest; continue is +1, disengage is -1; (K_0=0). Empty history is missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction; only preregistered fields may differ.

## Fixed constants
History decay is exactly 0.7.
