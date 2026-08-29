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

Parquet is preferred; collection falls back to compressed CSV if a Parquet
engine is unavailable.
