# Option termination (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H6 observable proxy, not a learned option policy.

## Variables
Continuation advantage (A_t), progress (P_t), goal continuity (G_t), action trace (K_t).

## Equations
(D_t=\beta_0+\beta_AA_t+\beta_PP_t+\beta_GG_t+\beta_KK_t+\beta_mm_t).

## History and initialization
(K_t=.7K_{t-1}+a_{t-1}), (K_0=0); empty action history is missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
Action decay is exactly 0.7; continuation advantage uses the signs in `dynamic_reevaluation.md`.
