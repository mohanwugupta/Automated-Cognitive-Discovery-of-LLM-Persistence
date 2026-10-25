# Red-to-green verification

The initial PRD implementation was written before its tests. The following
regression work was explicitly test-first, performed after the user requested TDD.

## Red

Added nine cases in `tests/test_alignment_diagnosis.py` before changing production
code. They require completed collection checkpoints to reject:

- A different data kind, model ID, model revision, or per-trial sampling seed.
- A raw response inconsistent with its saved parsed choice.
- An altered parsed choice, continuation indicator, observed failure indicator,
  or normalized continuation probability.

Command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_alignment_diagnosis.py \
  -k 'checkpoint_identity_mismatch or inconsistent_derived_measurements'
```

Observed result: **9 failed, 19 deselected**. All failures were `DID NOT RAISE`:
the collector accepted the inconsistent completed checkpoints.

## Green

Changed `_observations` in `alignment_diagnosis/pipeline.py` to verify each saved
participant identity and deterministic seed, recompute derived measurements from
the raw response and action probabilities, and reject inconsistencies. The same
loader protects resumption, fitting, and report generation.

Reran the identical command without weakening the assertions:
**9 passed, 19 deselected**.

## Regression verification

After formatting, ran the complete ordinary CPU suite:

```bash
PYTHONPATH=src python -m pytest -q -m 'not gpu and not model_download'
```

Observed result: **328 passed, 2 skipped**, with two environment compatibility
warnings (Transformers/Hugging Face Hub and frozen sklearn estimator versions).
Canonical validation also passed: **22 stages, 27 edges, 43 verified hashes**.
Shell syntax, whitespace, and tracked artifact size checks passed.

The CPU tests exercise Qwen inference through a fake model and run the complete
synthetic workflow. Real Qwen inference and GPU execution were not run locally.

## Immediate cluster submission regression

After a reported silent submission, added `tests/test_alignment_submission.py`
before changing the submission wrapper. A fake Python executable deliberately
fails if invoked on the login node; fake `sbatch` records the dependency chain.
Both tests initially failed: the wrapper called Python before submitting, and
did not reject an existing output directory before that call.

Moved preparation into a queued CPU job, added an early existing-output check,
and printed progress and job IDs for each submission. The same two tests then
passed. Preparation, freeze, and analysis request CPUs; initial and intervention
collection request GPUs, with `afterok` dependencies between all five jobs.

The focused regression run passed **50 tests** across submission, diagnosis,
Slurm resource contracts, and documentation. Shell syntax and whitespace checks
also passed. Actual Slurm submission was simulated locally, not executed remotely.
