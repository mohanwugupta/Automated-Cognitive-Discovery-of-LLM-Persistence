# Frozen transfer predictors, version 1

This document preregisters the directed task-transfer predictors before any
complete off-diagonal matrix is inspected. The machine-readable authority is
`configs/transfer/transfer_predictors_v1.yaml`; its SHA-256 is recorded in each
transfer run. A changed definition requires a new version and cannot replace a
completed v1 analysis.

The outcome is mapping-specific `cognitive_counterfactual_recovery`, measured
only by `global_cfr_v1`. No predictor may be constructed from CFR, its interval,
source-validity, or transfer status. All source objects must be frozen before
the corresponding predictor is computed.

The four families are:

1. **Behavioral/computational similarity:** coefficient-vector cosine and
   distance, theory equality, and signed similarities for outcome history,
   action history, continuation/disengagement value, progress, success
   evidence, and continuation cost.
2. **Task structure:** exact-match similarity over nine frozen descriptors,
   ordinal temporal-horizon similarity, and categorical choice-structure
   match. The seven task rows and all category values are explicit in YAML.
3. **Counterfactual geometry:** distributional similarity, effect variance,
   sign balance, source/base distance, and target-magnitude similarity, using
   only frozen counterfactual pairs and cognitive predictions.
4. **Neural similarity:** relative-depth similarity, rank match, principal
   angles, projection overlap, and streamed pre-DAS linear CKA. These are
   defined now but computed only after diagonal controllers are frozen.

The confirmatory estimator is ridge regression with alpha 10 and complete
leave-one-target-task-out validation. Parameterized modeling is not started by
the diagonal/pilot workflow. The task ontology is a declared analysis coding,
not an empirical result; revisions must be versioned as extensions.
