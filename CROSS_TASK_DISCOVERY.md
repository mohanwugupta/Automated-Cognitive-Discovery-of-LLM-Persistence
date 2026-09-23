# Three-model cross-task discovery

This is the prospective pipeline defined by
`PRD_Cross_Task_Computational_Mechanistic_Discovery_Three_LLMs.md`. It supersedes
the controller-first 7×7 plan. The old transfer package remains available as
historical/supporting infrastructure; its fitted controllers and outcomes are not
inputs to this run.

## Frozen identities

The machine-readable contract is `configs/discovery/cross_task_v1.yaml` and the
task × variable ontology is
`configs/discovery/task_variable_compatibility_v1.yaml`. The latter expands to 91
explicit cells. Unsupported cells remain unavailable instead of receiving invented
manipulations.

| Key | Model | Revision |
|---|---|---|
| qwen | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| gemma | google/gemma-4-12b-it | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| llama | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |

## Cluster launch: behavioral/computational phase

Commit the implementation first; final initialization rejects a dirty worktree.
On a Della login node:

```bash
cd /scratch/gpfs/JORDANAT/$USER/Automated-Cognitive-Discovery-of-LLM-Persistence
module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate llm-cognitive-discovery
export PYTHONNOUSERSITE=1

export REPO="$PWD"
export CROSS_TASK_OUTPUT="/scratch/gpfs/JORDANAT/$USER/cross-task/discovery-$(date +%Y%m%d-%H%M%S)"
export CROSS_TASK_MODEL_ROOT="/scratch/gpfs/JORDANAT/$USER/models"
bash scripts/submit_cross_task_discovery.sh
```

For a real-model integration smoke test, add `export CROSS_TASK_SMOKE=1`. The
submission validates each response interface, collects the same frozen semantic
conditions from all three models, estimates `B[a,t,v]`, and compares M1–M4 sharing
for every implemented cognitive family. A failed interface exits nonzero and blocks
downstream GPU collection. GPU jobs load a model and perform model forward passes;
CPU model fitting requests no GPU.

## Scientific freeze and neural stages

Behavioral survivor sets, active-discrimination outcomes, representation targets,
and causal hypotheses must be frozen together before neural outcomes are opened:

```bash
python scripts/cross_task_discovery.py freeze \
  --output "$CROSS_TASK_OUTPUT" --manifest frozen_hypotheses.json
```

The JSON must contain `behavioral_survivors`, `representation_targets`,
`causal_hypotheses`, and `active_discrimination_outcomes`. Re-freezing is refused.

Active rounds use behavior-only score tables (`coverage`, `uncertainty`, and
`disagreement`) and protect a separately listed untouched validation set:

```bash
python scripts/cross_task_discovery.py active-select \
  --output "$CROSS_TASK_OUTPUT" --model qwen --candidates scored_candidates.parquet \
  --observed-groups observed.json --validation-groups final_validation.json \
  --batch-size 350 --round 0
```

After the last refit, score the untouched behavioral validation observations
without refitting:

```bash
python scripts/cross_task_discovery.py final-validation \
  --output "$CROSS_TASK_OUTPUT" --model qwen \
  --observations qwen_untouched_validation.parquet
```

Representation extraction writes temporary feature rows to scratch. Analyze them
without target-task refitting:

```bash
python scripts/cross_task_discovery.py representation \
  --output "$CROSS_TASK_OUTPUT" --model qwen --features "$SCRATCH/qwen_features.parquet"
python scripts/cross_task_discovery.py effect-geometry \
  --output "$CROSS_TASK_OUTPUT" --model qwen --features "$SCRATCH/qwen_contrasts.parquet"
```

`representation` requires grouped `train/selection/test` identities and rejects
source/target group leakage. Its output is Level 3 evidence only. The feature file
is a scratch input and must not be committed.

Causal execution may use the existing DAS/intervention primitives, but its result
table is accepted only when the endpoint is
`cognitive_counterfactual_recovery`, the metric is `global_cfr_v1`, the registered
basis hash matches, and a target task was never used to fit an off-diagonal basis:

```bash
python scripts/cross_task_discovery.py validate-causal \
  --output "$CROSS_TASK_OUTPUT" --model qwen --results "$SCRATCH/qwen_causal.csv"
```

Run Qwen first as the integration model. Before causal work, use measured
representation throughput to freeze the GPU-hour/storage estimate. Do not submit
causal work if it exceeds the approved budget.

## Synthesis and reports

After all three models have complete frozen tables:

```bash
for model in qwen gemma llama; do
  python scripts/cross_task_discovery.py synthesize --output "$CROSS_TASK_OUTPUT" --model "$model"
  python scripts/cross_task_discovery.py report --output "$CROSS_TASK_OUTPUT" --model "$model"
  python scripts/cross_task_discovery.py figures --output "$CROSS_TASK_OUTPUT" --model "$model"
done
python scripts/cross_task_discovery.py cross-model --output "$CROSS_TASK_OUTPUT"
python scripts/cross_task_discovery.py report --output "$CROSS_TASK_OUTPUT"
python scripts/cross_task_discovery.py figures --output "$CROSS_TASK_OUTPUT"
python scripts/cross_task_discovery.py validate-final --output "$CROSS_TASK_OUTPUT"
```

The reports preserve the evidence hierarchy: behavioral causality (Level 1),
computational explanation (Level 2), representation (Level 3), causal neural effect
(Level 4), and cross-task causal sharing (Level 5). No absence of transfer, shared
subspace, or unique behavioral theory is converted into a pipeline failure.

## Artifact policy

Final tables and compact decoder/subspace artifacts belong under
`cross_task_discovery/`. Full activation banks do not. Stream activations on scratch,
persist only compact sufficient objects, and remove temporary activation files after
their hashes and analyses are recorded.
