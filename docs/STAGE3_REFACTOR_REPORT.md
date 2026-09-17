# Stage 3 repository-structure and usability report

## Outcome

Stage 3 completed the GREEN → REFACTOR → GREEN repository-structure pass on branch `stage3-repository-structure`, starting from approved Stage-2B commit `9518c15`. No frozen scientific artifact, target, split, metric, model identity, layer/rank choice, gate, or claim value changed. No GPU job, model download, or external write was performed.

The repository now provides manifest-driven claim navigation, one explicit provenance/rerun configuration for each canonical or supporting experiment, a guarded rerun entry point, a research-first README, stage guides, and an isolated clean-checkout audit.

## Old → new source and configuration paths

No scientific implementation was moved. The following paths are additive navigation and safety layers:

| Existing path retained | New path | Purpose |
|---|---|---|
| `canonical_manifest.yaml` stage graph | `canonical_manifest.yaml` `claims` registry | Adds machine-readable claim → analysis → code → config → artifact → manuscript navigation without changing stage/artifact identities. |
| `scripts/replay_canonical_claims.py` | `src/cognitive_discovery/reproduce.py` | Adds the documented module interface while retaining the Stage-2B wrapper. |
| Individual experiment scripts and public console commands | `src/cognitive_discovery/run.py` | Adds a guarded, dry-run-first rerun dispatcher; existing entry points remain intact. |
| Stage-specific configs such as `configs/causal_mech_v1.yaml` | `configs/canonical/<stage_id>.yaml` | Wraps historical scientific defaults with explicit provenance, identities, dependencies, availability, and safe output policy. Historical defaults remain authoritative and byte-identified. |
| Ad hoc stage discovery across source/scripts | `src/cognitive_discovery/stages/registry.py` | Validates the 16-config inventory against the canonical manifest. |
| Stage-2B endpoint/metric identities | Extended identity vocabulary in `reproducibility/identities.py` | Names non-recovery estimands and metrics; `global_cfr_v1` implementation and endpoint firewall are unchanged. |
| Stage-2B new-run provenance schema | Backward-compatible explicit `not_applicable` support | Prevents fabricated hashes for stages with no behavioral/neural object or pair split. Existing complete records remain valid. |

## Commands added

CPU-only replay and navigation:

```bash
python -m cognitive_discovery.reproduce claims
python -m cognitive_discovery.reproduce claim C07
python -m cognitive_discovery.reproduce stage causal_specificity_v2
```

Guarded experiment validation/execution:

```bash
python -m cognitive_discovery.run \
  --stage qwen_fixed_setting_stability_v1 \
  --config configs/canonical/qwen_fixed_setting_stability_v1.yaml

python -m cognitive_discovery.run \
  --stage qwen_fixed_setting_stability_v1 \
  --config configs/canonical/qwen_fixed_setting_stability_v1.yaml \
  --output artifacts/reruns/qwen_fixed_setting_stability_v1-NEW \
  --execute
```

The rerun command defaults to dry-run. Before execution it validates the manifest/config relationship, pinned revision, scientific-default hash, required input hashes, and output separation. It writes the Stage-2B provenance record before invoking the preserved script. Existing non-empty output directories and any path inside the frozen stage root are rejected.

Historical core Qwen configs intentionally set `rerun_allowed: false` because their resolved model revision is unknown. Later compact-only jobs remain blocked where raw conditions, activations, or a selected neural basis are unavailable. This is a provenance result, not a software failure.

## Compatibility wrappers retained

All pre-Stage-3 scripts and console entry points remain at their original paths. In particular:

- `scripts/{active_discovery,theory_resolution,mechanistic,action_history_disambiguation}.py`;
- `scripts/{causal_mechanistic,causal_specificity,causal_abstraction,ood_free_generation}.py`;
- Qwen and Llama run/analysis scripts;
- `scripts/submit_*.sh` and `scripts/dispatch_*.sh` Della wrappers;
- `scripts/replay_canonical_claims.py` and `scripts/validate_canonical.py`;
- all `cognitive-discovery-*` console commands declared in `pyproject.toml`.

No compatibility wrapper was redirected to a new scientific implementation.

## Legacy relocation

No source, script, or frozen artifact was relocated. This was deliberate: the older mechanistic scripts still supply the live matched-condition manifest path and/or public compatibility, and the later extension scripts remain needed to explain compact artifacts. Moving them would risk imports and published commands without changing their scientific status.

`legacy/README.md` now inventories supporting/legacy interpretations and directs readers back to the manifest. `mechanistic_manifest_v1` remains canonical; `mechanistic_probe_steering_v1` remains supporting with a legacy interpretation.

## Documentation added or revised

- Rewrote `README.md` in the required order: question, method, pipeline, findings, CPU replay, GPU reruns, historical/supporting work, structure.
- Added `REPRODUCING_RESULTS.md` with locked installation, CPU replay, guarded GPU reruns, runtime/compute expectations, external dependencies, and exact clean-clone limitations.
- Added `docs/stages/{behavioral,mechanistic,generalization,replication}/README.md`.
- Updated every C01–C13 replay command in `CLAIMS.md` to the manifest-driven interface.
- Added `legacy/README.md` without moving scientific code or artifacts.
- Extended CI to exercise the new replay/config/documentation tests and the module replay command.

## Canonical experiment configurations

Sixteen configs under `configs/canonical/` cover behavioral discovery, computational theory, the mechanistic manifest and supporting diagnostics, causal discovery/specificity, abstraction/boundary analyses, Qwen extensions, and Llama replication. Every config records:

- stage and pipeline phase;
- exact upstream stage/artifact IDs;
- model descriptor, checkpoint, revision or explicit historical unknown;
- endpoint and metric IDs;
- split/design path and identity;
- behavioral and neural objects, or explicit `not_applicable`/unavailable status;
- seeds and the byte hash of the original scientific-default source;
- frozen and new-run output roots;
- rerun availability and required inputs.

Only `qwen_fixed_setting_stability_v1` currently passes the full guarded-rerun preflight. This is the only stage for which the pinned revision, lightweight input objects, safe output override, and executable wrapper are all present. Other configs remain useful, exact provenance records and explain why execution is blocked.

## Clean-checkout question audit

An isolated candidate checkout was created at `/private/tmp/cognitive-stage3.CWTaqU/checkout` from the current branch, overlaid with only the Stage-3 candidate changes, and staged so Git-tracked clean-clone checks saw the exact proposed file set.

1. **What are the main claims?** `README.md` summarizes their supported/bounded form; `CLAIMS.md` gives the exact C01–C13 records.
2. **Which analysis supports each claim?** `canonical_manifest.yaml:claims` names the primary and supporting stage IDs. `python -m cognitive_discovery.reproduce claim CXX` prints the analysis, code, config, artifacts, and manuscript location.
3. **How are manuscript values reproduced?** `python -m cognitive_discovery.reproduce claims` replays all thirteen records from committed compact artifacts without GPU/network access.
4. **Which experiment regenerates each result?** The claim registry points to the stage config. C01–C05 use the corresponding core configs but cannot be exactly rerun because the historical Qwen revision is unknown; C06 maps to the executable fixed-setting config; C07–C09 map to pinned Qwen extension configs with explicit missing-raw-input blocks; C10–C12 map to Llama interface/behavior/mechanistic configs with their clean-clone limitations; C13 names the external repository audit command.
5. **Which results are compact-replayable only?** C06–C12. C01–C05 are numerically replayable but retain `historical_unknown`; C13 is an `external_dependency`.
6. **Which historical analyses are not canonical?** The old mechanistic probe/steering interpretation, action-history diagnostic, early failed Llama label interfaces, superseded per-example CFR, supporting Qwen stability/fresh-context extensions, and the external original study. Their precise statuses differ (`supporting`, `legacy`, or historical provenance) and remain explicit rather than collapsed.

Documentation failures found and fixed during this audit were: no machine-readable claim navigation, no single module replay command, no explicit per-experiment rerun availability, no safe-output default, and no concise separation of canonical mechanistic evidence from historical probes. Remaining unknown model revisions, raw inputs, and neural objects are scientific-provenance limitations, not documentation ambiguities, and are left explicit.

## Verification results

### Isolated candidate checkout

```text
canonical validation: 22 stages, 27 edges, 43 verified hashes — PASS
C01–C13 manifest-driven compact replay — 13/13 PASS
Stage-2B + Stage-3 focused suite — 36 passed in 5.50s
ordinary CPU suite — 178 passed, 2 skipped, 1 warning in 42.33s
```

### Local working tree

```text
Stage-2B + Stage-3 focused suite — 36 passed in 5.61s
ordinary CPU suite — 178 passed, 2 skipped, 1 warning in 43.54s
```

The two skips remain the approved explicit clean-clone/model-environment skips: absent raw Qwen confirmation conditions and an optional local Transformers/torchvision incompatibility. The warning remains the local scikit-learn 1.7.1 versus frozen-object 1.9.0 mismatch; `pyproject.toml` and `uv.lock` retain the intended environment identity.

### CI

The local command equivalent of the updated GitHub Actions workflow passes. Remote GitHub Actions status is `not_run` because this branch has not been pushed; it is not reported as a remote pass.

## Frozen-result confirmation

`scripts/validate_canonical.py` verifies all 43 pre-existing canonical artifact SHA-256 values. `git diff` contains no file under `artifacts/` or `paper/generated/`. The manifest-driven replay reproduces all thirteen approved claim records, including the separate C07 natural-effect and C08 cognitive-target endpoints. No canonical artifact hash or frozen claim value changed.
