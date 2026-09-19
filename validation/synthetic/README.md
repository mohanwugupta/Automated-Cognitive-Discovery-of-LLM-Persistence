# Generated synthetic-validation artifacts

Run `cognitive-discovery-model-validate --execute` from a clean checkout to
create `parameter_recovery.csv`, `model_recovery_matrix.csv`, and
`model_identifiability.json`, `implementation_checks.json`,
`validation_summary.json`, and `amended_gate_status.json` here (or use
`--output` for scratch). Do not promote a smoke or dirty-worktree run to
evidence.

The preserved `DEVELOPMENT_GATE_A_REPORT.md` records the original decision.
`AMENDED_GATE_STATUS.md` supersedes only its blocking interpretation:
implementation validity remains blocking, while model-recovery ambiguity is a
non-blocking diagnostic propagated as a behavioral survivor set. Generated
numeric CSVs remain outside the repository until a clean, committed run is
approved for evidence.
