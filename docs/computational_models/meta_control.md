# Meta-control (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H8 interaction operationalization.

## Variables
Continuation value (V_c), cost (C), controllability (Q), uncertainty (U), success (S), interactions (V_cQ,CU).

## Equations
(D_t=\beta_0+\beta_vV_c+\beta_cC+\beta_qQ+\beta_uU+\beta_sS+\beta_{vq}V_cQ+\beta_{cu}CU+\beta_mm_t).

## History and initialization
No direct history trace; missing terms follow the shared availability convention.

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction under frozen coefficients.

## Fixed constants
No history decay.
