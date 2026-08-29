# Automated Cognitive Discovery of LLM Persistence

This repository implements the coverage-oriented behavioral pipeline in
[`PRD.md`](PRD.md). It asks which computations govern continuing versus
disengaging across seven forms of goal pursuit, without making a persistence
representation or a favored cognitive theory part of the sampling policy.

The implemented pipeline is:

```text
ontology → balanced design compiler → task renderers → Qwen policy logits
         → fixed splits → cognitive tournament + flexible ceilings
         → LOTO + residual discovery → frozen independent validation
```

Discovery Round 2 extends that pipeline with a frozen six-variable task
ontology, M1--M4 parameter sharing, zero/few-shot task transfer, matched
flexible ceilings, nested residual validation, and a 40/40/20 active sampler.
Its specification is
[`PRD_Active_Hierarchical_Discovery_of_Persistence_Computations.md`](PRD_Active_Hierarchical_Discovery_of_Persistence_Computations.md).

Behavioral Theory Resolution (Round 3) completes the validity audits, freezes
the surviving dual-history/latent-context theories, adds operational A→B→A
context reinstatement in three domains, and runs a preregistered active
discrimination test before any mechanistic analysis. Its specification is
[`PRD_Behavioral_Theory_Resolution_Before_Mechanistic_Analysis.md`](PRD_Behavioral_Theory_Resolution_Before_Mechanistic_Analysis.md).

Phase 1 is behavior only. The runner never requests activations and rejects a
participant result containing hidden states.

## What is implemented

- A declarative ten-factor ontology with explicit construct availability.
- Coverage-oriented partial-factorial sampling and paired X/Y mappings.
- An optional SweetPea `CrossBlock` adapter plus a deterministic offline compiler.
- SweetBean-style renderer boundaries for Bandit, Foraging, Solvability,
  Information Sampling, Waiting, Effort, and Debugging.
- Exact semantic probability/logit extraction from Qwen3.5-4B.
- Causal 0→current history-prefix sequences for recurrent modeling; histories are
  prescribed experimental evidence and sampled actions remain a secondary outcome.
- Pair/episode-safe 60/20/20 discovery, interpolation, and structural splits.
- Eleven grounded cognitive models plus the generic sequential-choice control.
- Task-specific, fully shared, and hierarchical parameterizations.
- Regularized interactions, spline/GAM-like, MLP, and GRU flexible models.
- Synthetic teacher recovery, task-macro metrics, LOTO, residual discovery,
  explainable-variance fractions, and frozen independent validation.
- Reproducibility manifests, pilot gates, SLURM arrays, and an automated report
  answering all fifteen registered questions.

See [`docs/MIGRATION.md`](docs/MIGRATION.md) for the exact components selectively
ported from `digital-minds-hackathon`.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,qwen,parquet]'
pytest -q
```

SweetPea is optional because many offline cluster environments do not carry it:

```bash
pip install -e '.[design]'
```

`design.backend: native` and `design.backend: sweetpea` share the same declarative
specification. The native compiler is used for reproducible tests and cluster
jobs; `build_sweetpea_block` materializes the optional SweetPea block.

## Laptop smoke run

This validates every artifact boundary without loading a language model. Its
synthetic behavior is not scientific data.

```bash
cognitive-discovery-run \
  --config configs/discovery_v1.yaml \
  --output artifacts/smoke_v1 \
  --phase collect --conditions 70 --model-free

cognitive-discovery-run \
  --config configs/discovery_v1.yaml \
  --output artifacts/smoke_v1 \
  --phase finalize --conditions 70

cognitive-discovery-fit \
  --config configs/discovery_v1.yaml \
  --output artifacts/smoke_v1 --smoke
```

## Real pilot and discovery

The pilot target is 50–100 semantic conditions per family. A 70-per-family run
is 490 semantic conditions, 980 counterbalanced endpoint renderings, and about
3,080 scored history-prefix states:

```bash
cognitive-discovery-run \
  --config configs/discovery_v1.yaml \
  --output artifacts/pilot_v1 \
  --phase collect --conditions 490 \
  --model /absolute/path/Qwen--Qwen3.5-4B

cognitive-discovery-run \
  --config configs/discovery_v1.yaml \
  --output artifacts/pilot_v1 \
  --phase finalize --conditions 490
```

Inspect `artifacts/pilot_v1/validation/pilot_approval.json`. Full collection is
dependency-gated on that file by the SLURM workflow.

The configured discovery run uses 2,800 semantic conditions, 5,600
counterbalanced endpoint renderings, 17,600 scored history-prefix states, and
eight pair-safe collection shards:

```bash
for shard in 0 1 2 3 4 5 6 7; do
  cognitive-discovery-run \
    --config configs/discovery_v1.yaml \
    --phase collect --conditions 2800 \
    --shard-count 8 --shard-index "$shard" \
    --model /absolute/path/Qwen--Qwen3.5-4B --resume
done

cognitive-discovery-run --phase finalize --conditions 2800
cognitive-discovery-fit
python scripts/validate_predictions.py \
  --model /absolute/path/Qwen--Qwen3.5-4B --conditions 700
```

## Cluster

Download the model on an internet-connected login node, install the project in
the cluster environment, then submit the dependency chain:

```bash
sbatch --export=ALL,PHASE=smoke run_discovery.slurm
bash scripts/submit_discovery.sh
```

Override `MODEL_PATH`, `CONDA_ENV`, `SHARD_COUNT`, `OUTPUT`, or `SCRATCH_ROOT` as
needed. Jobs set Hugging Face and Transformers offline mode and keep semantic
pairs on the same array shard.

## Active hierarchical Discovery Round 2

Round 2 consumes the completed Round-1 standardized behavior; it never alters
the Round-1 sample or frozen validation predictions. To run the entire workflow
on Della from the repository root:

```bash
export ROUND1_OUTPUT=artifacts/discovery_v1
export V2_OUTPUT=artifacts/discovery_v2
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
export SCRATCH_ROOT=/scratch/gpfs/JORDANAT/$USER/llm-cognitive-discovery
export CONDA_ENV=llm-cognitive-discovery

bash scripts/submit_active_discovery.sh
```

The dependency chain performs:

```text
tests → Round-1 audits/M1–M4 fits → active manifest
      → 8-way active Qwen collection → quality gate
      → hierarchy update → untouched coverage-random manifest
      → 8-way final Qwen collection → frozen evaluation
```

The active manifest is immutable and can also be collected directly:

```bash
cognitive-discovery-run \
  --config configs/discovery_v2.yaml \
  --output artifacts/discovery_v2/active_collection \
  --phase collect \
  --manifest artifacts/discovery_v2/active_sampling/selected_conditions.jsonl \
  --model "$MODEL_PATH"
```

The new Della workflow requests full 40 GB A100 GPUs for Qwen and neural-ceiling
phases; tests, shard merging, and final tabulation use CPU-only jobs.

## Behavioral Theory Resolution Round 3

Round 3 consumes the fixed Round‑1 behavior and Round‑2 active observations. It
does not refit on the untouched Round‑3 discrimination subset until after the
frozen comparison has been written. Run the entire Della dependency chain with:

```bash
export ROUND1_OUTPUT=artifacts/discovery_v1
export ROUND2_OUTPUT=artifacts/discovery_v2
export V2_ACTIVE_OUTPUT=artifacts/discovery_v2/active_collection
export V3_OUTPUT=artifacts/theory_resolution_v1
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
export SCRATCH_ROOT=/scratch/gpfs/JORDANAT/$USER/llm-cognitive-discovery
export CONDA_ENV=llm-cognitive-discovery

bash scripts/submit_theory_resolution.sh
```

The dependency chain is:

```text
tests → 36 teacher/audit checks → M2/M3/M4 uncertainty → theory freeze
      → ≥20,000 legal original/contextual candidates → 60/20/20 manifest
      → 8-way Qwen collection → quality gate
      → frozen comparison → post-comparison update → mechanistic handoff
```

SLURM resource routing is phase-specific. Qwen collection and neural-model
audits/tournaments use GPU jobs. Tests, shard merging, quality gates,
hierarchical updates, frozen comparisons, and reporting use CPU-only jobs. The
runner exits immediately if a CPU-only phase is accidentally submitted with a
GPU allocation, and every GPU phase verifies CUDA before beginning expensive
work.

The collector is still behavior-only: no hidden states or activations are
requested or retained. To exercise every artifact boundary without loading
Qwen, run preparation with `--smoke`, collect its manifest using
`scripts/run_experiments.py --model-free`, then evaluate with `--smoke`.

## Artifacts

The run directory follows the PRD layout:

```text
artifacts/discovery_v1/
  design/                 ontology, SweetPea spec, condition manifests
  behavior/raw/           resumable pair-safe shards
  behavior/prompts/       prompts keyed by hash
  behavior/standardized   validated policy data
  splits/                 fixed partition IDs
  analysis/               response surface, tournament, LOTO, residuals
  validation/             untouched sample, predictions, metrics
  report.md
  run_metadata.json
```

Round-2 artifacts follow the separate PRD layout:

```text
artifacts/discovery_v2/
  audits/                 flexible recovery, hierarchy recovery, information audit
  hierarchy/              M1–M4 comparison, LOTO, few-shot, task parameters
  active_sampling/        candidate/selection manifests, scores, budget curves
  active_collection/      Round-2 Qwen observations
  final_validation/       untouched manifest, predictions, calibration, task metrics
  residuals/              nested candidates and held-out confirmation
  figures/                five registered Round-2 figures
  report.md
  run_metadata.json
```

Round‑3 artifacts are isolated under:

```text
artifacts/theory_resolution_v1/
  audits/                 teacher failures, M2/M3/M4, parameters, calibration
  frozen_models/          immutable specifications, fits, hashes, prediction code
  contextual_history/     A→B→A manifest, SweetPea grammar, factor coverage
  active_sampling/        20k candidate pool/predictions and 60/20/20 selection
  round3_collection/      Qwen policy observations only
  discrimination/         frozen errors, paired bootstrap, reinstatement effects
  theory/                 updated comparison, coefficients, mechanistic targets
  figures/                six registered theory-resolution figures
  report.md
  run_metadata.json
```

Parquet is preferred; collection falls back to compressed CSV if a Parquet
engine is unavailable.
