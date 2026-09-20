# Computational-model validation and task transfer

This extension is prospective. It does not rewrite historical Qwen, Gemma, or
Llama results and does not change the core estimand: all causal tables use
`cognitive_counterfactual_recovery` with `global_cfr_v1`.

## Gate status

- Gate A is split into blocking implementation validity (A1) and non-blocking
  theory identifiability (A2). The full model bank has frozen v1 documents,
  hand fixtures, independent core references, and synthetic parameter/model
  recovery. A1 passed for the development run. A2 was `partial` because
  `outcome_history` and `dual_history` were not reliably distinguishable. The
  original report is preserved at `validation/synthetic/DEVELOPMENT_GATE_A_REPORT.md`;
  `validation/synthetic/AMENDED_GATE_STATUS.md` supersedes only its blocking
  interpretation.
- Gate B's paper, Figshare Version 3 dataset, factual web-experiment subset,
  six-model bank, LML contrasts, bootstrap, and decision thresholds are
  owner-frozen in `validation/external_benchmark/BENCHMARK_SPEC.md`. The public
  source data are hash-verified and all 858 MAP fits converged. Six exact
  Hessians fail positive definiteness at the active `tau=1` boundary, so Gate B
  is `not_evaluable` and benchmark-claim permission is `stop`. Because
  implementation equivalence passed, the owner-frozen nonblocking amendment
  sets downstream `pipeline_permission: continue`. No aggregate contrast or
  ranking has been computed.
- Gate C is `not_run`. `configs/replication/qwen_prospective_run.yaml` uses the
  pinned Qwen3.5-4B follow-up revision already recorded by the repository.
- Gates D/E are implemented as fail-closed transfer and predictor contracts; no
  transfer result is present until cluster runs create it.

## Computational validation

Dry-run, then execute:

```bash
python scripts/validate_computational_models.py --root "$PWD"
python scripts/validate_computational_models.py --root "$PWD" --execute
```

Outputs are `validation/synthetic/parameter_recovery.csv`,
`model_recovery_matrix.csv`, `model_identifiability.json`,
`implementation_checks.json`, `amended_gate_status.json`, and
`validation_summary.json`. Generated tables are
run artifacts; inspect and approve them before opening Gate A.
Parameter recovery uses three parameter settings at three sample sizes per model;
the full model-recovery matrix uses five independently seeded replicates per
teacher. `--smoke` reduces recovery replicates and is not Gate-A evidence.
Imperfect diagonal model recovery is diagnostic and does not close A1.

## Behavioral survivor sets

The replication harness freezes `behavior/models/frozen_survivor_set.json`
before counterfactual generation. Membership uses only selection-split
behavioral MSE and the equivalence margin in
`configs/replication/default.yaml`. Every survivor gets separate cognitive
targets, DAS search, held-out evaluation, specificity controls, and a theory
index in task transfer. Provenance hashes both the rule and membership, and all
neural stages fail if either is changed. Neural CFR is never used to rewrite the
behavioral survivor set.

## Prospective Qwen

The prospective wrapper refuses a dirty worktree, runs the complete CPU suite,
records the exact commit and `uv.lock` SHA-256, then submits the existing tested
model-agnostic harness. Commit all intended changes before running it. Do not
reuse any historical fit, basis, selected grid point, pairs, or splits:

```bash
export OUTPUT="$SCRATCH/replications/qwen-prospective"
export CONDA_ENV=llm-cognitive-discovery
bash scripts/submit_qwen_prospective.sh
```

The scientific output is rooted at `$OUTPUT` and includes `provenance.json`,
`interface/`, `behavior/`, `frozen_theories/`, `counterfactuals/`, `mechanism/`,
`specificity/`, and `REPLICATION_REPORT.md`. Every stage rechecks the frozen
configuration, run specification, baseline preflight, Git commit, and tracked
scientific files. The report keeps cognitive-target recovery separate from the
excluded natural-effect and fresh-context endpoints, reports every behavioral
survivor independently, and prints the owner-frozen Outcome A/Outcome B
interpretation without historical tuning.

## Transfer pilot and full matrix on Della

The submitter defaults to one work unit so GPU-hours can be measured first:

```bash
export REPO="$PWD"
export REPLICATION="$SCRATCH/replications/qwen-prospective"
export TRANSFER_OUTPUT="$SCRATCH/transfer/qwen-pilot-v1"
export CONDA_ENV=llm-cognitive-discovery
bash scripts/submit_task_transfer.sh
```

The pilot requires a clean committed worktree. It uses one layer, one rank, one
epoch, and two random controls. It is a runtime/integration check and cannot be
promoted to evidence. Because two controls cannot attain the frozen null-test
threshold, the pilot alone may override the diagonal dispatch gate solely to
exercise all seven targets; aggregation still marks its scientific outgoing
cells unavailable. Its report scales
the measured times by the frozen full layer/rank grid, epoch count, random-control
count, and number of source controllers; it does not treat smoke timing as if it
were a full controller.

Gemma and Llama figures in the Qwen pilot report are planning estimates based
on the frozen 12B:8B:4B parameter-count ratios (3:2:1), not measurements. Each
model's own pilot measurement supersedes that estimate before its budget is
approved.

After the pilot completes, inspect:

```bash
cat "$TRANSFER_OUTPUT/PILOT_RESOURCE_REPORT.md"
```

Only after explicitly approving both projected budgets, use a new output
directory for the full matrix:

```bash
export TRANSFER_FULL_MATRIX_APPROVED=1
export TRANSFER_PILOT_REPORT="$TRANSFER_OUTPUT/resource_projection.json"
export TRANSFER_APPROVED_GPU_HOURS=<approved-ceiling>
export TRANSFER_APPROVED_STORAGE_GB=<approved-ceiling>
export TRANSFER_OUTPUT="$SCRATCH/transfer/qwen-full-v1"
export TRANSFER_MAX_CONCURRENT=2
bash scripts/submit_task_transfer.sh
```

Each GPU job trains from source-task `transfer_train`, selects layer/rank using
only source-task `transfer_selection`, and first evaluates only the diagonal
`T_iim`. Gate D is frozen before the dispatcher submits any off-diagonal job. A
failed within-task controller makes its off-diagonal cells `unavailable`, never
zero. Eligible controllers then fill the remaining cells of the 7×7 matrix for
each frozen survivor. Full activations are streamed and never persisted. No
parameterized transfer-explanation model is submitted by this workflow; matrix
predictors are already frozen in
`configs/transfer/transfer_predictors_v1.yaml`, but no parameterized model is
submitted by this workflow.

The source gate is `task_source_validity_v1`: pooled held-out CFR, its lower
bootstrap bound, and target correlation must be positive; the finite-sample
matched-random p-value must be at most .05; and both `continue_x` and
`continue_y` must have positive, sign-consistent CFR with a maximum gap of .25.
Failures are `invalid_source_controller`; their outgoing cells are
`unavailable`, never zero.

Aggregation writes the preregistered contract under `diagonal/` and `matrix/`:
controller manifests, pooled and mapping-specific diagonal results, source
validity, mapping-indexed long results, mapping matrices, and a pooled summary.
The exact endpoint remains `cognitive_counterfactual_recovery` / `global_cfr_v1`.

Run the evidence matrices in order by changing only the completed prospective
replication root and a fresh output directory: Qwen, then Gemma, then Llama.
Each model retains its own frozen behavioral survivor set. Do not reuse a
controller, pair manifest, split, layer, or rank across models.
