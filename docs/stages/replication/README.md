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

The submitter routes model-forward stages to `slurm/run_replication_gpu.slurm` and fitting, hashing, counterfactual construction, and reporting to `slurm/run_replication_cpu.slurm`. This prevents CPU-only work from reserving a GPU. Set `ONLINE_FLAG=--online` only when compute nodes may access Hugging Face; otherwise pre-cache the exact revision.

Do not start Phase B until the owner has selected the genuinely new model and immutable revision. A required ontology change, target mismatch, missing identity, or frozen-gate change is a stop condition requiring review.
