# Behavioral Theory Resolution Before Mechanistic Analysis

No activation or representation analysis is performed in this workflow.

## Registered questions

1. **Teacher failures.** 6 of 36 checks failed. fitted_dual_history/linear_interactions/fully_shared: sharing_mismatch; fitted_dual_history/mlp/fully_shared: sharing_mismatch; fitted_dual_history/gru/fully_shared: sharing_mismatch; fitted_latent_context/linear_interactions/fully_shared: sharing_mismatch; fitted_latent_context/mlp/fully_shared: sharing_mismatch; fitted_latent_context/gru/fully_shared: sharing_mismatch
2. **Flexible-ceiling validity.** Not an upper bound where a registered check fails.
3. **M2/M3/M4 uncertainty.** 9 paired summaries are available; intervals, medians, and win probabilities are in the audit table.
4. **Ontology variance.** Median ontology fraction Q=0.455.
5. **Stable parameter signs.** factor_success_evidence, factor_progress_evidence, factor_continuation_cost, factor_disengagement_value, history_action_kernel, history_outcome_kernel
6. **Information Sampling.** 313 semantic/mask/scale/mapping rows audited; with/without sensitivity is preserved.
7. **Frozen theories.** dual_history, latent_context, outcome_history
8. **Disagreement space.** 20000 candidates scored across original and contextual domains.
9. **Targeted discrimination.** Overall mean Δerror(DH−LC)=0.0596.
10. **A→B→A reinstatement.** Contextual subset mean Δerror=0.2055.
11. **Cue reliability.** Reliability-specific matched effects are reported in `discrimination/context_reliability_effect.csv` and Figure 5.
12. **Behavioral theory.** unresolved: paired interval crosses zero outside the equivalence margin
13. **Mechanistic variable.** history_integration, contextual_history_relevance
14. **Intervention effects.** 7 task-parameter derivatives with uncertainty are frozen in `theory/final_parameters.csv`.

## Interpretation boundary

The frozen comparison is predictive. Post-update fits are reported separately and cannot retroactively change the frozen test. If the registered interval supports equivalence, no arbitrary winner is named.
