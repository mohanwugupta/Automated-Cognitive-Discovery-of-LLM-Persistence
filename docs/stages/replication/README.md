# Replication

## Frozen conceptual replication

`llama_interface_v2` validates the Yes/No response interface. `llama_behavior_v2` is the canonical conceptual behavioral replication, and `llama_mechanistic_v2` is the qualified conceptual mechanistic replication. Earlier X/Y, A/B, and ordering attempts are measurement-validity history rather than negative replications.

Cognitive-counterfactual and natural-effect recovery remain separate endpoints. The selected Llama controller is layer 24, rank 2, but the exact basis object is external/unavailable; C11 is therefore compact-replayable rather than a clean-clone neural rerun.

Replay C10–C12 with `python -m cognitive_discovery.reproduce claim C11` (replace the claim ID as needed). The authoritative limitations are in [CLAIMS.md](../../CLAIMS.md) and [canonical_manifest.yaml](../../../canonical_manifest.yaml).

The model-agnostic harness has a separate purpose: prospectively apply the approved structured-task method to a new model. Its invariant is that discovery, generalization, and every specificity control use `cognitive_counterfactual_recovery` with `global_cfr_v1`. It never substitutes the natural source-minus-base effect.

## Phase A gate

Run the compact frozen Llama replay before starting a new model:

```bash
python -m cognitive_discovery.replicate_model \
  --frozen-llama-replay \
  --output artifacts/replications/llama-phase-a \
  --execute
```

This verifies the approved Yes/No interface, behavioral summaries, layer 24/rank 2 protocol, and held-out cognitive recovery from committed inputs. It does not reconstruct the unavailable Llama basis or run a natural-effect endpoint.

## New-model dry run and execution

Use an immutable 40-hex Hugging Face revision. The command is validation-only until `--execute` is present:

```bash
python -m cognitive_discovery.replicate_model \
  --model <hf-model-id> \
  --revision <40-hex-commit> \
  --adapter <qwen|llama|gemma|mistral> \
  --config configs/replication/default.yaml \
  --output artifacts/replications/<run-id>
```

After reviewing the plan, add `--execute` to run all core stages. The interface gate runs first. If no candidate is valid, the run ends as `measurement_failure`; this is not a negative scientific replication.

Resume one stage without changing the frozen config or splits:

```bash
python -m cognitive_discovery.replicate_model \
  --resume artifacts/replications/<run-id> \
  --stage mechanism \
  --execute
```

Valid core stages are `interface`, `behavior`, `model_comparison`, `freeze_theory`, `counterfactuals`, `mechanism`, `generalization`, `specificity`, and `report`. `--stage all` skips already completed stages. Optional abstraction and OOD work is disabled in the default protocol and is not silently entered.

## Frozen prospective choices

- Seven task families are split into four neural-fitting tasks and three whole-task holdouts.
- Behavioral groups are assigned to train, selection, and test before model comparison.
- Counterfactual pairs are assigned to neural train, selection, test, and task holdout before DAS.
- Layers are resolved from relative depths (25%, 50%, 75%, 90%); no Qwen or Llama layer is hard-coded.
- Ranks 2 and 8 are searched by default.
- Behavioral theories are selected on `behavior_selection`; test data only reports preregistered confirmation.
- DAS is trained on `neural_train`, selected on `neural_selection`, and frozen before either held-out split is touched.
- Specificity uses actual matched orthonormal random subspaces, shuffled sources, output/readout geometry, a predictive ridge direction, and a PCA variance subspace. A target-preserving shuffle is recorded as `uninformative_control`.

## Artifacts and provenance

Every run writes `effective_config.yaml`, `protocol.json`, `run_state.json`, and `provenance.json` before inference. Progressive provenance records the repository state, exact model and tokenizer commits, adapter version, environment lock hash, seeds, realized behavioral design and split hashes, frozen-model hashes, counterfactual and neural split hashes, resolved layer/rank grid, controller hash, endpoint, metric, and output path.

The final `REPLICATION_REPORT.md` keeps measurement, behavior, controller recovery, held-out examples, held-out tasks, specificity, abstraction, and OOD statuses separate. Full activation banks are not retained.

## Della

Initialize and submit the dependency chain with:

```bash
MODEL_ID=<hf-model-id> \
REVISION=<40-hex-commit> \
ADAPTER=<qwen|llama|gemma|mistral> \
OUTPUT="$PWD/artifacts/replications/<run-id>" \
bash scripts/submit_replication.sh
```

The submitter first runs only the model-forward interface stage, then schedules
an `afterany` CPU dispatcher. A passing interface causes the dispatcher to
submit the remaining resource-separated chain. A measurement failure submits
only a CPU report, so a behavior job cannot reserve a GPU merely to discover
that the interface gate stopped the run. Model-forward stages use
`slurm/run_replication_gpu.slurm`; fitting, hashing, counterfactual construction,
dispatch, and reporting use `slurm/run_replication_cpu.slurm`. Set
`ONLINE_FLAG=--online` only when compute nodes may access Hugging Face;
otherwise pre-cache the exact revision.

## Short GPU smoke suite before a full replication

Run the smoke suite before submitting the multi-stage pipeline for a new model.
It uses the production renderers, response-token checks, paired collection,
generic residual runner, intervention hooks, and DAS primitive. In addition to
checkpoint and chat validation, it now measures each candidate interface's
actual full-vocabulary action mass and whether an action token is the top token.
It collects both response mappings for every task and writes tiny-sample previews
of the same interface gates used by the full run. Those previews test plumbing
only: they are explicitly not measurement-gate evidence. Neural checks cover
layer discovery, streamed residual capture, an identity edit, a nonzero edit,
DAS gradient flow with frozen model weights, EOS IDs, and peak CUDA memory. The
smoke suite does **not** fit a cognitive model, compute CFR, test specificity, or
provide scientific evidence.

Each model is one one-hour GPU array task, with one task active at a time by
default. CPU regression tests run first; an `afterany` CPU job then writes a
combined report even when a GPU task fails:

```bash
cd /scratch/gpfs/JORDANAT/$USER/Automated-Cognitive-Discovery-of-LLM-Persistence

export SMOKE_MODEL_PATHS="/scratch/gpfs/JORDANAT/$USER/models/google--gemma-4-12b-it /scratch/gpfs/JORDANAT/$USER/models/meta-llama--Llama-3.1-8B-Instruct"
export SMOKE_REVISIONS="<gemma-40-hex-commit> <llama-40-hex-commit>"
export SMOKE_ADAPTERS="gemma llama"
export SMOKE_ROOT="/scratch/gpfs/JORDANAT/$USER/replication-smoke/$(date +%Y%m%d-%H%M%S)"
bash scripts/submit_replication_smoke.sh
```

`SMOKE_ADAPTERS` is optional and defaults to `auto`. Automatic detection is
fail-closed: Qwen, Llama, Gemma, and Mistral model types map to registered
adapters. Nemotron is deliberately reported as unsupported until a dedicated
adapter has been implemented and tested; it is never silently treated as
Mistral. To include all three candidate directories in the diagnostic array,
use three paths/revisions and `SMOKE_ADAPTERS="auto auto auto"`; the aggregate
report will preserve Nemotron's adapter failure alongside the other results.

Tokenizer paths and revisions default to their corresponding model values. For
separate tokenizers, provide equal-length `SMOKE_TOKENIZER_PATHS` and
`SMOKE_TOKENIZER_REVISIONS` lists. Set `SMOKE_MAX_CONCURRENT=2` only if running
two checkpoint loads simultaneously is appropriate for the allocation. The
default is one, limiting scheduler and filesystem pressure.

Inspect `SMOKE_ROOT/SMOKE_REPORT.md` and each
`job_XXXX/smoke_result.json`. No full activations or residual vectors are saved;
each result contains scalar diagnostics, shapes, hashes, and peak memory only.
Do not start `submit_replication.sh` for a model unless its smoke result passes.

Do not start Phase B until the owner has selected the genuinely new model and immutable revision. A required ontology change, target mismatch, missing identity, or frozen-gate change is a stop condition requiring review.
