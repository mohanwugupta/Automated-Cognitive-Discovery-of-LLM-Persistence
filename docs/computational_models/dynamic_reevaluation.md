# Dynamic re-evaluation (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H1 operationalization.

## Variables
(A_t=V_{c,t}-V_{d,t}-C_t), progress (P_t), success evidence (S_t), uncertainty (U_t).

## Equations
(D_t=\beta_0+\beta_A A_t+\beta_P P_t+\beta_S S_t+\beta_U U_t+\beta_m m_t).

## History and initialization
History affects this model only through current-state inputs; no trace is initialized.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
The continuation-advantage signs are `+ continuation`, `- disengagement`, `- cost`.
