# Broader Qwen fresh-context experiments

Completed **13,440 intervention evaluations**, plus 640 baseline/state evaluations and a secondary diagnostic on 320 natural source conditions. All three previously selected controllers and their cognitive models remained frozen. No new controller fitting, layer/rank search, or test-based tuning occurred.

The design contains four distinct current-state backgrounds per task, four signed history patterns at lengths three and five, and both response mappings. Context-isolated cue contrasts are included for bandit, foraging, and debugging. The 640 rendered conditions represent 320 semantic states, with zero overlap with the original mechanistic manifest. The standard fully counterbalanced loader passes. Wording templates remain fixed; these are new semantic contexts, not a broad paraphrase study.

Each ordinary-history controller is evaluated on 224 pairs, and the context controller on 192 pairs. Each has 20 newly drawn random-subspace controls targeting the same layer, rank, and per-pair displacement norm.

## Primary endpoint: frozen cognitive targets

Recovery is `1 - sum((intervention effect - target effect)^2) / sum(target effect^2)`. Zero is a no-change baseline; negative values are worse than that baseline. It is not a percentage of causal mediation. Intervals resample background clusters within task 1,000 times, retaining all histories and mappings together.

| Controller | Tasks | Cognitive-target recovery [95% interval] | Random comparison p | Natural-effect recovery* |
|---|---|---:|---:|---:|
| Dual history (L28/r2) | Familiar | 0.268 [0.119, 0.411] | 0.048 | 0.894 |
| Dual history (L28/r2) | Held out | 0.026 [-0.254, 0.276] | 0.048 | 0.871 |
| Outcome history (L30/r2) | Familiar | 0.037 [-0.132, 0.206] | 0.048 | 0.897 |
| Outcome history (L30/r2) | Held out | -0.033 [-0.358, 0.258] | 0.048 | 0.835 |
| Latent context (L30/r8) | Familiar | -3.751 [-4.819, -2.659] | 1.000 | 0.403 |
| Latent context (L30/r8) | Held out | 0.019 [-0.282, 0.301] | 0.048 | 0.816 |

Five of six controller/split comparisons exceed every sampled random subspace on the cognitive endpoint. The exception is the contextual controller on familiar context-isolated contrasts. Beating the random controls does not imply good target recovery: the primary held-out score is near zero, with an interval spanning zero, and the outcome-history held-out score is negative. With only 20 controls, the minimum plus-one p-value is 1/21 = 0.0476. These coarse values are descriptive and are not corrected across controllers and endpoints.

## Secondary endpoint: natural effects

*The natural-effect diagnostic was added during the run and is not a replacement for the primary endpoint.* Natural effects are the model's source-minus-base logits on the same controlled pairs. The primary controller has normalized recovery 0.894 on familiar tasks and 0.871 on held-out tasks, with correlations 0.953 and 0.924. Every controller/split comparison exceeds all 20 random subspaces on this secondary endpoint. The primary random maxima are only 0.063 and 0.068.

The cognitive models themselves are imperfect targets: for dual history, cognitive predictions have normalized recovery -0.168/-0.393 against natural effects despite correlations 0.882/0.806. Weak cognitive-target recovery therefore cannot be assigned entirely to neural intervention failure. The discovered controller reproduces actual history-change effects better than its frozen quantitative cognitive targets on these fresh contexts.

This supports a useful controller while leaving its cognitive identity and natural necessity unresolved. Generic decision/output-aligned controls were not rerun on this new design. The earlier E-identification and unrestricted-EOS results are unchanged. The full discovery search has not been replicated.

## Finite-precision norm audit

Targeted norm matching is approximate in bfloat16. Across method/split groups, the largest median relative discrepancy against the achieved frozen-controller norm is 2.69%; the maximum individual discrepancy is 24.1%. The raw values and all discrepancies are retained.

A post hoc sensitivity analysis uses a common subset where every random control is within 5% of the achieved frozen norm. It retains 127/128 familiar and 79/96 held-out pairs for the primary controller. Primary natural-effect recovery is 0.895/0.887, and both cognitive and natural endpoints still exceed every random control. The other comparisons retain their full-data random-ranking pattern. This diagnostic is based on norm mismatch, not recovery scores, and does not replace the full-data analysis.

## Reproducibility and resources

- Qwen checkpoint: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime: Torch `2.8.0.dev20250319+cu128`, Transformers `5.16.1`, L40S; peak allocated VRAM 9.29 GB. Main experiment time: 2,878 seconds, excluding provisioning/setup.
- All 13,440 outcomes are exported in `paper/generated/qwen_fresh_contexts/interventions.csv.gz`. The analysis supports an export-only replay without GPU inference. The full archive includes conditions, predictions, every intervention, all bases, logs, protocols, and executed scripts.
- Verified archive SHA-256: `4d8ca0ad979055b168f55273897931919584013b3d8c7f77e8aa64afbfb8b267`.
- Pod `yq92v60ihexfcg` was deleted after verified download. Empty pod listing confirmed. No network volume was created. Estimated allocation/storage cost is below $1.10; this is not a posted billing figure.

The next scientific step is to improve and independently validate the behavioral counterfactual model, while preserving these fixed-controller results. Broader prompt wording, more contexts, more precise nulls, and decision/output specificity controls remain necessary before stronger mechanism claims.
