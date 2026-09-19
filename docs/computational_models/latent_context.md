# Latent context (`computational-model-spec-v1`)

specification_status: frozen

## Status
Frozen H5 cue-weighted operationalization; not a free latent-state inference model.

## Variables
Immediate state (X_t), recent trace (R_t), context-matched trace (R_t^c), cue reliability (q_t), goal continuity (G_t), stability (E_t).

## Equations
At a change point set (R_t^c=0); otherwise select A/B history from the return cue.
(R_t^*=q_tR_t^c+(1-q_t)R_t).
(D_t=\beta_0+\beta_X^TX_t+\beta_*R_t^*+\beta_G R_t^*G_t+\beta_E R_t^*E_t+\beta_mm_t).

## History and initialization
All traces use oldest-to-newest ordering and zero initialization. If contextual fields are absent, (R_t^*=R_t).

## Fitting and prediction
Use the ridge and sharing contract in `README.md`.

## Counterfactual
Source-minus-base prediction; cue and raw-history interventions must be labeled separately.

## Fixed constants
All outcome traces decay by exactly 0.7; (q_t\in[0,1]).
