# Generic sequential choice (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H9 sequential-control model.

## Variables
Newest action/outcome, their traces (K_t,R_t), and environmental stability (E_t).

## Equations
(D_t=\beta_0+\beta_a a_{t-1}+\beta_KK_t+\beta_rr_{t-1}+\beta_RR_t+\beta_EE_t+\beta_mm_t).

## History and initialization
(K_t=.7K_{t-1}+a_{t-1}), (R_t=.7R_{t-1}+r_{t-1}), zero initialization; empty histories are missing.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
Both decay constants are exactly 0.7.
