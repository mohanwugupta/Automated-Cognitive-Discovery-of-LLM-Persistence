# Latent motivation (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H4 proxy; it is not a fitted hidden-state dynamical system.

## Variables
Continuation value (V_c), cost (C), progress (P), outcome trace (R_t), and proxy (M_t).

## Equations
(R_t=.7R_{t-1}+r_{t-1}); (M_t=V_{c,t}-C_t+P_t+.7R_t);
(D_t=\beta_0+\beta_MM_t+\beta_cV_c+\beta_CC+\beta_PP+\beta_mm_t).

## History and initialization
(R_0=0), oldest-to-newest; an empty history contributes zero inside (M_t).

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
Outcome-trace decay and its latent-state multiplier are both exactly 0.7.
