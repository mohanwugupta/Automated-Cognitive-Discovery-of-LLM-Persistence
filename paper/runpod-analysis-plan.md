# Prospective analysis plan for qwen_runpod_v1

Fixed before reading neural evaluation scores, September 8, 2026.

- Report every optimization seed; no best-seed selection. This measures stability conditional on the existing L28/rank2 selection, not full discovery stability.
- Separate original validation, original pair test, original task holdout, fresh pair-test tasks, and fresh held-out tasks.
- Compare conventional/direct baselines and random subspaces to the original frozen DAS using equal layer, rank, pair set, and per-pair intervention norm. Scalar-target probe and mean-difference baselines complete rank 2 with an orthogonal train-PCA direction; label this rule explicitly. New cognitive DAS seeds retain their native norm for stability analysis.
- Compute principal angles between all five learned subspaces and against the original frozen subspace.
- For the fresh signed-history diagnostic, compare observed recovery with 1,000 target derangements (seed 76002). Independently permute the four history patterns within each task, use the same pattern permutation for both response mappings, and require no fixed points. Thus mappings remain paired and every target changes. Keep actual neural effects fixed; this is an evaluation-target randomization, not retraining on shuffled labels.
- Bootstrap semantic contrasts for the original test rows. Fresh-history contrasts share one background template per task/mapping, so report their scores as a narrow diagnostic; do not present 56 rows as 56 independent task generalization trials.
- Twenty norm-matched random subspaces imply minimum empirical p = 1/21. Treat comparisons as exploratory and report all methods; do not treat this pilot as a familywise-confirmatory superiority test.
- Report numerical baseline drift relative to old artifacts and use recomputed baselines for all new neural effects. The old run did not record an immutable model revision.
- Preserve failed outcomes, runtime errors, software versions, exact checkpoint SHA, and raw per-pair results. No OOD tuning or second-model claims.

## Post-run implementation clarification

This records a metadata/coverage limitation found during final artifact checks; no neural values or selection rules were changed.

Manifest audit: the fresh JSONL is a per-mapping diagnostic, not a standard fully counterbalanced mechanistic manifest. Inherited `paired_condition_id` values denote original provenance; use the explicit source/base pair table for fresh pairing. Background templates differ across mappings for debugging, effort, foraging, and waiting. The synchronized permutation keeps each within-mapping contrast intact but does not establish response-label invariance.
