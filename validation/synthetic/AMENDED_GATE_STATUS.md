# Superseding Gate-A interpretation

Date: 2026-09-18  
Amendment: `PRD_Amendment_Nonblocking_Theory_Identifiability.md`

This document does **not** alter or delete
`DEVELOPMENT_GATE_A_REPORT.md`. It supersedes only that report's decision to
block the pipeline on imperfect theory recovery.

## Split status

- `implementation_validity: pass`
- `identifiability_status: partial`
- `pipeline_permission: continue`

The registered-model hand fixtures, independent core reference comparisons,
fixed predictions, counterfactual sign checks, and synthetic parameter recovery
passed. The observed `outcome_history` → `dual_history` confusion is therefore
recorded as limited behavioral identifiability between nested/near-equivalent
theories, not as evidence of an implementation failure.

This remains a development interpretation: the source matrix was generated
from an uncommitted implementation worktree and is not claim-bearing evidence.
Future committed runs emit `implementation_checks.json`,
`model_identifiability.json`, `validation_summary.json`, and
`amended_gate_status.json` together.

Gate B was subsequently executed. All 858 MAP fits converged, but six exact
Hessians are non-positive-definite at the active `tau=1` boundary. Gate B is
therefore `not_evaluable` with `pipeline_permission: stop` pending an
owner-frozen constrained-boundary Laplace convention. A subsequent owner-frozen
amendment separates benchmark-claim permission (`stop`) from downstream
pipeline permission (`continue`) because implementation equivalence passed. No
primary contrast, bootstrap, or model ranking was computed. See
`validation/external_benchmark/NONBLOCKING_BOUNDARY_AMENDMENT.md` and
`benchmark_execution_status.json`.
