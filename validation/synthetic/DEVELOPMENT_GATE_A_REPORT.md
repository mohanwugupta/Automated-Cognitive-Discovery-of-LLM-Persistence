# Gate-A development check (not claim-bearing evidence)

Date: 2026-09-18  
Status: `blocked_owner_review`  
Evidence eligibility: **No** — executed from the implementation worktree before commit.

The non-smoke validation command completed and applied the thresholds frozen in
`configs/validation/computational_models_v1.yaml`. Generated CSVs remained in
scratch and were not promoted to canonical artifacts.

## Preregistered decisions

- Maximum absolute parameter bias: 0.40.
- Minimum predictive parameter-recovery R²: 0.98.
- Minimum diagonal recovery probability for each of `dual_history`,
  `latent_context`, and `outcome_history`: 0.80.
- Five model-recovery replicates per teacher.

## Development result

- Parameter recovery: passed (maximum absolute bias 0.3243; minimum predictive
  R² 0.9832).
- `dual_history` recovery probability: 1.00.
- `latent_context` recovery probability: 1.00.
- `outcome_history` recovery probability: 0.40.
- `outcome_history` was selected as `dual_history` in 0.60 of replicates.
- Mean diagonal recovery across the complete bank: 0.7167.

This is an identifiability result, not an implementation failure to patch away.
The current predictive comparison cannot reliably distinguish the nested
outcome-only model from dual history under this simulation and selection rule.
Per the PRD stop condition, Gate A remains closed pending owner review. The
external benchmark, prospective Qwen run, and claim-bearing transfer matrices
remain `not_run`.
