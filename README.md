# Cognitive models as guides to causal discovery

## 1. Scientific question

When a language model persists rather than disengages, can behavioral cognitive models identify the computation and guide a causal search for a low-dimensional neural controller? This repository separates four questions that are easy to conflate: behavioral predictability, recovery of a frozen cognitive counterfactual, recovery of the model's natural effect, and identification of a unique computational abstraction.

The approved scientific record is [CLAIMS.md](docs/CLAIMS.md). Machine-readable identities, dependencies, statuses, and artifact hashes are in [canonical_manifest.yaml](canonical_manifest.yaml).

## 2. Discovery methodology

The study first explores a counterbalanced seven-task behavioral design space, compares computational models, and actively seeks conditions that discriminate among surviving theories. It then freezes the unresolved behavioral handoff, searches for low-rank causal neural controllers with DAS, reanalyzes specificity with a stable aggregate recovery metric, tests downstream abstraction and free-generation boundaries, and finally evaluates Qwen extensions and a Llama conceptual replication.

The primary recovery endpoints are deliberately distinct:

- `cognitive_counterfactual_recovery`: recovery of a frozen behavioral model's predicted source-minus-base effect.
- `natural_effect_recovery`: recovery of the unmodified language model's empirical source-minus-base effect.

Both use `global_cfr_v1` where applicable. They must never be pooled or renamed as one another.

## 3. Canonical pipeline

```text
behavioral discovery/data (discovery_v1)
  -> active model comparison (discovery_v2)
  -> unresolved behavioral handoff (theory_resolution_v1)
  -> matched-condition manifest (mechanistic_manifest_v1)
  -> DAS causal search (causal_mech_v1)
  -> stable metric/specificity (causal_specificity_v2)
  -> abstraction identity test (abstraction_discovery_v1)
  -> free-generation boundary (ood_free_generation_v1)
  -> Qwen generalization + Llama conceptual replication
```

See [CANONICAL_PIPELINE.md](docs/CANONICAL_PIPELINE.md) for the full dependency graph and stage-by-stage estimands. The original hackathon study is external historical provenance, not a stage in the current causal pipeline.

## 4. Main supported and bounded findings

- History-sensitive models predict structured Qwen persistence, but the frozen comparison among dual history, latent context, and outcome history is unresolved.
- Low-dimensional Qwen DAS controllers recover cognitive-model counterfactuals on held-out pairs and, for selected controllers, held-out tasks. This is Level-4 causal representation evidence, not necessity or unique theory identity.
- The corrected specificity analysis uses `global_cfr_v1`; Level-5B necessity is `not_run`. Two target-preserving shuffled controls are `uninformative_control`.
- E is the strongest descriptive downstream abstraction candidate, but no abstraction uniquely passes the full identity gate.
- The OOD free-generation result is a boundary result and does not establish E identity.
- Independent Qwen confirmation supports natural-effect recovery while cognitive-target recovery remains a separate, weaker endpoint.
- Llama Yes/No behavior and mechanistic results are a qualified conceptual replication. Earlier failed label interfaces are measurement-validity history, not negative replications.

Exact values, limitations, and artifact identities are indexed by C01–C13 in [CLAIMS.md](docs/CLAIMS.md).

## 5. Reproduce manuscript numbers

Install the locked CPU environment, then replay all claims from committed compact artifacts:

```bash
uv sync --frozen --extra dev --extra parquet
source .venv/bin/activate
python -m cognitive_discovery.reproduce claims
```

Inspect one claim or one stage:

```bash
python -m cognitive_discovery.reproduce claim C07
python -m cognitive_discovery.reproduce stage causal_specificity_v2
```

These commands require no GPU, model download, or network access. They use `canonical_manifest.yaml`, verify the relevant frozen hashes, and show claim → analysis → code → config → artifact → manuscript navigation. See [REPRODUCING_RESULTS.md](docs/REPRODUCING_RESULTS.md) for installation, runtimes, and clean-clone limits.

## 6. Rerun experiments

All new reruns use one guarded pattern and default to validation-only dry runs:

```bash
python -m cognitive_discovery.run \
  --stage qwen_fixed_setting_stability_v1 \
  --config configs/canonical/qwen_fixed_setting_stability_v1.yaml
```

Add `--execute` only after reviewing the printed command and provenance. New outputs go under `artifacts/reruns/`; the runner rejects writes into a frozen canonical output root. It also requires an exact model revision, verifies committed inputs, and writes `run_provenance.json` before starting the delegated job.

Historical core Qwen stages intentionally fail closed because their resolved revision was not recorded. Other compact-only stages fail closed where raw inputs or neural objects are unavailable. The preserved scripts and Della submission wrappers remain available for historical inspection; they are not substitutes for the guarded rerun command.

For a genuinely new checkpoint, the model-agnostic replication harness runs the core behavioral-to-causal method with a single pinned config and immutable model revision. It is a dry run unless `--execute` is supplied:

```bash
python -m cognitive_discovery.replicate_model \
  --model <hf-model-id> \
  --revision <40-hex-commit> \
  --adapter <qwen|llama|gemma|mistral> \
  --config configs/replication/default.yaml \
  --output artifacts/replications/<run-id>
```

Review the plan, replay the frozen Llama regression, and only then add `--execute`. See [the replication harness guide](docs/stages/replication/README.md) for restart commands, phase gates, artifacts, and the CPU/GPU-separated Della submission path.

## 7. Historical and supporting analyses

`mechanistic_v1` has two roles: its matched-condition manifest is a live canonical dependency, while its probe/steering interpretation is supporting/legacy. `action_history_disambiguation_v1`, fixed-setting Qwen stability, and broader fresh-context Qwen work are supporting. Failed Llama label interfaces are measurement-validity history. The external `digital-minds-hackathon` study remains historical provenance.

No scripts were relocated in Stage 3 because several historical entry points still serve imports, artifact reconstruction, or public compatibility. [legacy/README.md](legacy/README.md) records this inventory without hiding it from the canonical path.

## 8. Repository structure

```text
canonical_manifest.yaml       machine-readable graph, identities, hashes, claim navigation
docs/CLAIMS.md                manuscript claim-to-evidence contract
docs/CANONICAL_PIPELINE.md    scientific stages and dependency diagram
configs/canonical/            one provenance/rerun config per canonical or supporting experiment
configs/replication/          frozen model-agnostic replication protocol
src/cognitive_discovery/      scientific implementation plus reproducibility and stage guards
scripts/                      preserved analysis, experiment, and cluster compatibility wrappers
slurm/                        CPU/GPU resource wrappers, including the replication harness
artifacts/                    lightweight frozen canonical scientific artifacts
paper/generated/              committed compact manuscript evidence
docs/stages/                  short behavioral/mechanistic/generalization/replication guides
tests/provenance/             identity, hash, gate, and provenance contract
tests/regression/             frozen C01–C13 numerical contract
legacy/                       inventory and relocation policy; no frozen artifacts moved
```

For audit history, see [REPO_AUDIT.md](docs/REPO_AUDIT.md), [STAGE2B_REPORT.md](docs/STAGE2B_REPORT.md), and [STAGE3_REFACTOR_REPORT.md](docs/STAGE3_REFACTOR_REPORT.md).
