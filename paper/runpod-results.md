# Qwen RunPod follow-up: completed

The original test effects reproduce, but the fresh signed-history diagnostic exposes poor calibration and fails matched-random specificity. This supports a bounded controller claim. It does not establish a universal persistence mechanism.

The run completed on September 8, 2026 (local time) using Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on one RunPod L40S. Five cognitive DAS fits, one direct behavior-target DAS fit, four conventional bases, and twenty random bases produced 3,720 intervention evaluations. All five seeds are retained.

## Global counterfactual recovery

| Method | Original test | Original task holdout | Fresh test tasks | Fresh held-out tasks |
|---|---:|---:|---:|---:|
| cognitive_DAS_75001 | 0.973 | 0.875 | 0.109 | 0.019 |
| cognitive_DAS_75002 | 0.900 | 0.773 | 0.297 | -0.398 |
| cognitive_DAS_75003 | 0.966 | 0.891 | 0.072 | -0.319 |
| cognitive_DAS_75004 | 0.949 | 0.899 | 0.046 | -0.492 |
| cognitive_DAS_75005 | 0.983 | 0.877 | 0.305 | -0.073 |
| direct_DAS_75001 | 0.844 | 0.759 | -0.804 | -1.746 |
| frozen_DAS | 0.970 | 0.900 | 0.080 | -0.426 |
| mean_difference_rank2 | 0.252 | 0.229 | 0.334 | 0.101 |
| pca_rank2 | 0.231 | 0.197 | 0.109 | 0.167 |
| ridge_history_rank2 | 0.168 | 0.178 | 0.076 | 0.126 |
| ridge_persistence_rank2 | 0.195 | 0.195 | 0.154 | 0.131 |

Across five seeds, original test recovery is 0.954 ± 0.033 SD; original task-holdout recovery is 0.863 ± 0.051. Maximum pairwise principal angles range from 27.8° to 88.0°, so similar performance does not imply identical recovered coordinates.

## Fresh specificity diagnostic

Four signed history contrasts per task and response mapping yield 32 familiar-task and 24 held-out-task pairs. Each task/mapping has four distinct predicted effects, with both signs. One background template per task/mapping limits coverage. Fitting uses only the original 24 training pairs.

For the frozen controller, fresh target-effect correlations are 0.935 and 0.838, but global recovery is only 0.080 and −0.426. Across 1,000 mapping-synchronized within-task target derangements, every target changes; true assignments score above all shuffles (empirical p = 1/1001 for each task group). This demonstrates correspondence, not accurate quantitative recovery. Against twenty norm-matched random subspaces, empirical p = 0.095 and 0.952. Some conventional bases perform better on the fresh diagnostic. No superiority or specific-mechanism claim follows.

## Execution and interpretation boundaries

- Layer 28/rank 2 fixed from the previous primary dual-history controller. This is optimization stability, not a repeated layer/rank search. Other retained controllers and Llama are not rerun here.
- Conventional/direct/random interventions match the frozen controller's norm. Scalar-direction baselines explicitly use an orthogonal train-PCA direction to complete rank 2. Maximum method-level median norm discrepancy is 0.53%, consistent with bfloat16 arithmetic. New cognitive seeds use native intervention norms.
- Original baseline drift: max 0.125 logits, mean absolute 0.064. Original artifacts have no immutable model SHA; all new effects use fresh baselines.
- New runtime: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.16.1. Peak allocated GPU memory: 9.69 GB.
- The original pipeline's `mean_cfr` divides per row and is unstable for near-zero targets. This report consistently uses `global_cfr`; raw outputs preserve both.
- The signed-history design was recorded before model loading; the local analysis plan preceded inspection of evaluation scores. This was not an external preregistration.
- Twenty random draws limit empirical p-value resolution to 1/21; comparisons remain exploratory.
- Downloaded archive SHA-256 matched the remote archive. Both the Pod and network volume were deleted, and empty listings verified. Estimated total compute/storage cost is under $1; billing records had not posted at the final check.

## Relationship to the original paper

The rewrite develops [Sisyphus in the Loop](https://apartresearch.com/project/sisyphus-in-the-loop-what-makes-an-llm-persist-yqd9), retaining its distinction between decodable reward and causal control and restoring its research context. Original sequential-bandit statistics remain attributed to that report; they are not pooled with these structured-task results. See `paper/source-lineage.md`.

## Files

`paper/generated/runpod_interventions.csv` contains all 3,720 per-pair outcomes; join `pair_index` to `runpod_pairs.csv`. Other generated CSVs contain all seed/method scores, angles, bootstrap intervals, and norm checks. The downloadable reproducibility archive additionally contains every fitted alignment, raw Parquet outputs, logs, and the exact executed runner.

Manifest audit: the fresh JSONL is a per-mapping diagnostic, not a standard fully counterbalanced mechanistic manifest. Inherited `paired_condition_id` values denote original provenance; use the explicit source/base pair table for fresh pairing. Background templates differ across mappings for debugging, effort, foraging, and waiting. The synchronized permutation keeps each within-mapping contrast intact but does not establish response-label invariance.
