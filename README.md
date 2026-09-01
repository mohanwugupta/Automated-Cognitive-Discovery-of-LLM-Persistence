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

The mechanistic follow-up is isolated from that collector. It freezes the
Round-3 targets, streams final-prompt-token residual states through paired
sufficient statistics, and only permits steering and projection patching after
the representation gate. Its specification is
[`PRD_Mechanistic_Implementation_Context_Sensitive_Outcome_History_Integration.md`](PRD_Mechanistic_Implementation_Context_Sensitive_Outcome_History_Integration.md).

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

## Mechanistic outcome-history workflow

This stage consumes the immutable Round-3 handoff. It does not refit behavioral
coefficients from activations or intervention outcomes. On Della:

```bash
export THEORY_OUTPUT=artifacts/theory_resolution_v1
export MECH_OUTPUT=artifacts/mechanistic_v1
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
export MECH_SCRATCH=/scratch/gpfs/JORDANAT/$USER/persistence_mech
export CONDA_ENV=llm-cognitive-discovery

bash scripts/submit_mechanistic.sh
```

The resource-separated dependency chain is:

```text
CPU tests → CPU target/dataset freeze
          → GPU paired direction scan → GPU scalar projection scan
          → CPU probes, LOTO, controls, and representation gate
          → GPU calibrated steering and projection patching
          → CPU causal aggregation, figures, report, and artifact-size check
```

GPU phases verify CUDA before loading Qwen. CPU phases are submitted through
`run_mechanistic_cpu.slurm`, which has no GPU directive; the runner rejects an
accidental GPU allocation for those phases. Hooks observe only the final prompt
token and request `output_hidden_states=False`. Full activation banks are never
written. Temporary storage and model caches live under `MECH_SCRATCH`.

The CPU-only preparation boundary can be checked locally:

```bash
cognitive-discovery-mechanistic \
  --config configs/mechanistic_v1.yaml \
  --theory-output artifacts/theory_resolution_v1 \
  --output /tmp/mechanistic_v1_smoke \
  --phase prepare --smoke
```

Run `python scripts/check_artifact_sizes.py` before committing. It rejects any
tracked file over 10 MB unless explicitly reviewed in
`config/artifact_allowlist.txt`, and rejects tracked mechanistic scratch banks.

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

Mechanistic artifacts are compact and isolated under:

```text
artifacts/mechanistic_v1/
  manifests/              matched conditions, splits, prompt hashes
  directions/             per-layer safetensors directions and metadata
  representation/         scalar projections, decoding, LOTO, controls, gates
  calibration/            frozen behavioral coefficients and neural calibration
  steering/               dose response and predicted-versus-observed effects
  patching/                projection-only patches and mediation summaries
  figures/                 six registered mechanistic figures
  report.md
  run_metadata.json
```

Parquet is preferred; collection falls back to compressed CSV if a Parquet
engine is unavailable.

## Action-history direction disambiguation

The follow-up in `PRD_Action_History_Direction_Disambiguation.md` is implemented
as a separate, read-only extension of `artifacts/mechanistic_v1`. It freezes the
exact original layer-30 vector, residualizes action history using training rows
only, and compares raw, output-subspace-orthogonal, statistically residualized,
persistence-probe, and direct semantic-logit-gradient directions at layers 8 and
30. Full residual-stream activations are never saved.

Run the CPU preparation phase locally after the original mechanistic artifacts
have been copied into the repository:

```bash
python scripts/action_history_disambiguation.py \
  --config configs/action_history_disambiguation_v1.yaml \
  --source artifacts/mechanistic_v1 \
  --phase prepare
```

On a Della login node, submit the complete resource-separated workflow with:

```bash
export CONDA_ENV=llm-cognitive-discovery
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
bash scripts/submit_action_history_disambiguation.sh
```

Its dependency graph is:

```text
CPU tests → CPU freeze/residualization/matching
          → GPU probe and gradient fit → GPU scalar projections
          → CPU representation analysis
          ├→ GPU primary steering → GPU optional patching ─┐
          └→ GPU random-null array (10 shards, max 2 live) ├→ CPU report
```

The CPU wrapper has no GPU directive. Every GPU phase performs Qwen forwards
and a CUDA preflight; this prevents CPU-only jobs from consuming a GPU allocation.
The random-null array runs all 100 directions at both primary layers on the full
seven-dose grid, while limiting simultaneous jobs by default.

The new artifact root is:

```text
artifacts/action_history_disambiguation_v1/
  directions/             frozen and controlled safetensors vectors
  targets/                train-only residual targets and residualizer manifest
  representation/         scalar projections, matches, decoding, geometry
  calibration/            frozen coefficients and matched-norm scales
  steering/               primary curves, random null, behavioral correspondence
  patching/                optional decision-matched projection patches
  figures/                 seven preregistered figures
  gates.json
  report.md
  run_metadata.json
```

## Counterfactual causal mechanistic discovery

`PRD_Counterfactual_Causal_Mechanistic_Discovery.md` is implemented as the
counterfactual-first workflow in
`src/cognitive_discovery/causal_mechanistic/`. It treats the complete frozen
behavioral theory bank as immutable, computes quantitative source/base
counterfactuals before any neural optimization, and labels all probe results as
information access rather than mechanism.

The causal stages are:

```text
frozen theory bank → deterministic counterfactual pair manifest
→ all-layer information diagnostic (Level 2)
→ all-layer whole-state patching (Level 3)
→ rank 1/2/4/8 shared and task-specific DAS
→ untouched test/task-holdout validation (Level 4)
→ random, shuffled, output, value, task, mapping, and unrelated-variable controls
→ causal specificity (Level 5) plus a separately reported necessity test
→ gated attention/linear-token-mixer and MLP tracing (Level 6)
```

On a Della login node:

```bash
conda activate llm-cognitive-discovery
pip install -e '.[qwen,parquet]'
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
bash scripts/submit_causal_mechanistic.sh
```

The submission uses CPU jobs for theory freezing, diagnostics, summaries,
selection, and reporting. GPU allocations are restricted to phases that execute
Qwen forwards. Conditional CPU dispatchers inspect the completed localization
and DAS gates before submitting exact-size GPU arrays, so a stop rule never
creates an idle GPU job.

Circuit analysis is deliberately separate and is submitted only after the
specificity gate marks it eligible:

```bash
bash scripts/submit_causal_circuits.sh
```

If it is not eligible, that command exits without reserving a GPU. Outputs are
isolated under `artifacts/causal_mech_v1/` using the PRD directories:
`behavioral_handoff`, `counterfactuals`, `localization`, `representations`,
`validation`, `circuits`, and `figures`. Only compact alignments, scalar causal
results, hashes, and summaries are retained; full activation banks are forbidden.

## Stable-CFR DAS causal specificity

`PRD_DAS_Causal_Specificity_Stable_CFR.md` is implemented by the frozen-DAS
reanalysis in `src/cognitive_discovery/causal_specificity/`. It does not retrain
the three Level-4 alignments. It freezes their exact safetensors and hashes, then
replaces unstable mean per-example recovery with aggregate global CFR:

```text
CFR_G = 1 - Σ(neural effect - frozen counterfactual effect)²
            / Σ(baseline effect - frozen counterfactual effect)²
```

The workflow recomputes Level 3, bootstraps Level 4 at the semantic-pair level,
and evaluates every frozen candidate against 500 matched random subspaces,
shuffled sources and targets, direct persistence, output, generic-value,
task-ID, response-mapping, and cross-variable controls. It also produces the
functional specificity matrix, principal-angle diagnostics, context-conflict
theory comparison, and task-transfer results. Only Level-5A-passing candidates
receive a subsequent task-conditioned necessity job. Level 5A specificity and
Level 5B necessity are reported as separate gates; circuit work remains blocked
unless specificity passes.

Run the isolated reanalysis from a Della login node after `causal_mech_v1` is
complete:

```bash
conda activate llm-cognitive-discovery
pip install --upgrade -e '.[qwen,parquet]'
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
bash scripts/submit_causal_specificity.sh
```

CPU jobs freeze inputs, calculate corrected summaries and bootstrap intervals,
and build the final report. The dispatcher submits an exact three-candidate GPU
array (or the exact frozen candidate count) only for Qwen intervention forwards.
Outputs are written under `artifacts/causal_specificity_v2/`; no full activation
bank is stored.

## Causal abstraction discovery

`PRD_Causal_Abstraction_Level_Persistence_Computation.md` is implemented by
`src/cognitive_discovery/causal_abstraction/`. This stage starts from the
completed Level-4 result and its unresolved variable-specificity finding. It
hash-freezes the existing layer-28/rank-2 DAS controller and all behavioral
models before constructing a dedicated 50,000-condition candidate pool.

The CPU design stage selects at least 200 contrasts from each of six factorial
dissociation families, balances the predicted effect distribution, and freezes
competing raw-history (O), contextual-history (O*), integrated-history (H), and
total-evidence (E) predictions. The same untouched neural interventions are
then scored against all four abstractions. Natural-state geometry, matched
persistence/output/random controls, shuffled targets, held-out tasks, and only
the preregistered layers 16/20/24/28/30 are analyzed without retraining the
primary DAS rotation.

Run the workflow on a Della login node after `causal_specificity_v2` is complete:

```bash
conda activate llm-cognitive-discovery
pip install --upgrade -e '.[qwen,parquet]'
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
bash scripts/submit_causal_abstraction.sh
```

Candidate generation, behavioral prediction, bootstrap inference, figures, and
the final report use CPU-only jobs. The dispatcher submits an exact six-element
GPU array only for Qwen natural-state and intervention forwards. Results are
isolated under `artifacts/abstraction_discovery_v1/`, and circuit work remains
blocked unless one abstraction passes every neighboring-dissociation and
matched-control gate.

## OOD free-generation validation

`PRD_OOD_Generalization_Persistence_Evidence_Subspace.md` is implemented in
`src/cognitive_discovery/ood_generation/`. It copies and hash-verifies the
existing layer-28/rank-2 DAS artifact, then derives a signed E axis using only
the structured-task coverage projections and frozen behavioral E values. The
axis, structured activation scale, prompts, doses, sampling policy, controls,
exclusions, and statistical plan are hash-frozen before any free generation.

The GPU phase uses a custom KV-cached autoregressive loop. It intervenes only
on the newly processed final-token residual, leaves every canonical EOS token
sampleable, and has no application-level output-token cap. EOS is an event;
architectural context exhaustion, infrastructure timeout, and resource
exhaustion are recorded as distinct right-censoring reasons. The default
compute-efficient profile uses the PRD-permitted first stage of 6 prompts × 50
matched seeds × 5 doses = 1,500 primary generations. Small pulse, fixed-topic,
full-generation random, and direct-EOS batteries bring the total to 2,380 full
generations, down from 7,600 in the original all-secondary configuration. All
100 random directions still receive the inexpensive immediate EOS-logit test.

After the abstraction artifacts are present on Della, run:

```bash
conda activate llm-cognitive-discovery
pip install --upgrade -e '.[qwen,parquet]'
export MODEL_PATH=/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B
export ABSTRACTION_OUTPUT=artifacts/abstraction_discovery_v1
export OOD_OUTPUT=artifacts/ood_free_generation_v1
bash scripts/submit_ood_generation.sh
```

The dependency chain is CPU tests → CPU protocol freeze → CPU dispatcher →
GPU evaluation array (maximum two live by default) → CPU survival analysis and
reporting. CUDA placement is enforced inside every evaluation shard, while the
test, freeze, dispatch, and aggregation jobs reject accidental GPU allocations.
Each GPU shard requests at most six hours (rather than the earlier 24-hour
ceiling) and exits immediately when its assigned generations finish.
