# Stage 2B report

## Outcome

Stage 2B completed the RED → GREEN reproducibility layer on branch `stage2b-reproducibility`, using scientific baseline commit `3a5c8749d881632beccd2f12cc1016992ccc358f`. No experiment runner, saved scientific result, or public experiment entry point was refactored. No GPU experiment or model download was run.

The final canonical check validates 22 stages, 27 dependency edges, and 43 byte-level artifact hashes. All C01–C13 compact claim checks reproduce their frozen values or categorical outcomes.

## Baseline freeze

The machine-readable baseline record is `provenance/stage2b_baseline.json`. It records:

- baseline commit and cleanup branch;
- pre-Stage-2B tree status;
- local Python, platform, and relevant package versions;
- SHA-256 hashes of the four approved Stage-2A provenance files before Stage-2B additions;
- an explicit declaration that no historical provenance was backfilled.

The four approved file hashes at the boundary were:

| File | SHA-256 |
|---|---|
| `CANONICAL_PIPELINE.md` | `56f9db4f851f21dce0cd1e192211b82148dead9da3762e0b99c6782b73118cf4` |
| `CLAIMS.md` | `988499f74ebd7af70e8dce4e580067e3969b0af8cb212cf86398461e6a7bbc7f` |
| `canonical_manifest.yaml` | `fbe7d3611a7d88d9910bdbf552c997b46601d21bcc92f0e653703f971c325b3e` |
| `STAGE2_REFACTOR_PLAN.md` | `b2ec310e70ff2e29795db3cdb07a0f3f3a1055eb6585d2b3e9e3365de9f8f973` |

`canonical_manifest.yaml` was subsequently extended only with explicit `replay_status` fields. Its scientific stage statuses, artifact identities, and results were not changed.

## RED → GREEN record

The initial RED command was:

~~~bash
pytest -q tests/provenance tests/regression
~~~

It stopped with six collection errors because `cognitive_discovery.reproducibility` did not exist. This was the intended RED boundary: the repository had no shared manifest, identity, split, provenance, or replay layer. The failures were not caused by a GPU, network access, model download, or missing untracked artifact.

| Safeguard | Test | Intended RED failure | Minimal GREEN change |
|---|---|---|---|
| Manifest schema and DAG | `test_manifest_and_artifacts.py` | No schema/status/dependency validator | Added loader, schema checks, dependency resolution, and cycle detection. |
| Artifact hashes | `test_manifest_and_artifacts.py` | No end-to-end canonical hash verifier | Added streaming SHA-256 verification for all non-null manifest outputs. |
| Replay availability | `test_manifest_and_artifacts.py` | Analyses had no machine-readable availability class | Added one of `fully_replayable`, `compact_replay_only`, `external_dependency`, or `historical_unknown` to every stage. |
| Metric identity | `test_metrics_and_targets.py` | No canonical metric ID rejected legacy per-example averaging | Added `MetricID.GLOBAL_CFR_V1`, exact aggregate implementation, legacy rejection, and equality check against the existing stable implementation. |
| Endpoint identity | `test_metrics_and_targets.py` | Cognitive and natural target selection was procedural | Added typed endpoint IDs and fail-closed column selection. |
| Persistence/interface identity | `test_metrics_and_targets.py` | No shared target/interface identity guard | Added signed semantic-logit and binary-interface validators; existing token-mapping tests remain intact. |
| Split/holdout semantics | `test_splits_pairs_and_neural.py` | No shared co-assignment/leakage validator | Added semantic-group co-assignment, overlap checks, and frozen familiar/held-out task sets. |
| Pair/design identity | `test_splits_pairs_and_neural.py` | Pair hashes/counts were not asserted in one canonical suite | Froze pair/prediction hashes, split counts, target counts, abstraction counts, and wording/mapping grouping. |
| Neural identity | `test_identity_and_provenance.py` | Layer/rank/path/hash checks were stage-local | Added a neural descriptor and exact L28/rank-2/basis verification that fails on any mismatch. |
| Model revision identity | `test_identity_and_provenance.py` | Historical and pinned identities were heterogeneous | Added typed model descriptors; core Qwen remains explicitly unknown, while later Qwen and Llama revisions are exact and required. |
| Behavioral object identity | `test_identity_and_provenance.py` | Architecture, file, in-memory, and training-data identities were not checked together | Added registry verification across all three frozen behavioral models without unpickling or rewriting them. |
| Claim/gate consistency | `test_claim_gate_consistency.py` | Owner decisions were prose rather than fail-closed assertions | Added checks for unresolved theory, `not_run` necessity, uninformative shuffles, descriptive E, boundary OOD, and mechanistic-manifest status. |
| Frozen numerical claims | `test_frozen_claim_replay.py` | No one CPU path covered C01–C13 | Added compact claim replay with frozen-value assertions and separate endpoint labels. |
| Clean-clone availability | manifest/replay tests plus `test_confirmation_protocol.py` | Qwen wording test required an uncommitted raw condition manifest | The test now reports/skips with `compact_replay_only`; committed compact rows still replay both endpoints. No raw or large artifact was added. |
| New-run provenance | `test_identity_and_provenance.py` | No complete fail-closed schema/writer | Added validation and atomic writing for commit, dirty state, model revision, seeds, hashes, objects, endpoint, metric, config, and output directory. |
| Artifact size | existing `scripts/check_artifact_sizes.py` in CI | The existing policy was not part of reproducibility CI | Reused the existing 10 MB policy, scratch/activation bans, and reviewed `config/artifact_allowlist.txt`; no duplicate policy was retained. |

An intermediate GREEN run exposed five implementation defects in the new test/support code: one omitted test argument and a set-operation precedence bug in the split validator. Both were corrected without touching scientific code or data. The new suite then passed 25/25.

## Claim replay table

| Claim | Endpoint | Replay status | Frozen value reproduced? | Remaining dependency |
|---|---|---:|---:|---|
| C01 | Behavioral model comparison | `historical_unknown` | Yes: unresolved, winner `null` | Core Qwen resolved revision was not recorded. |
| C02 | `cognitive_counterfactual_recovery` | `historical_unknown` | Yes: all six test/holdout controller CFR values | Core Qwen revision; GPU inference is not replayed. |
| C03 | `cognitive_counterfactual_recovery` and specificity gates | `historical_unknown` | Yes: two Level-4 candidates, empty necessity, unchanged rank-2 shuffles | Core Qwen revision; Level-5B was not run. |
| C04 | Abstraction identity gate | `historical_unknown` | Yes: descriptive winner E, full gate false | Core Qwen revision; committed interchange results only. |
| C05 | Free-generation survival boundary | `historical_unknown` | Yes: coefficient and interval; OOD gate false | Core Qwen revision and raw model inference. |
| C06 | `cognitive_counterfactual_recovery` stability | `compact_replay_only` | Yes: five seeds, L28/rank-2 setting | Five seed-specific fitted basis files are not committed. |
| C07 | `natural_effect_recovery` | `compact_replay_only` | Yes: 0.778584 familiar, 0.815848 held out | Raw Qwen confirmation condition manifest is not committed. |
| C08 | `cognitive_counterfactual_recovery` | `compact_replay_only` | Yes: 0.250536 familiar, -0.427508 held out | Raw Qwen confirmation condition manifest is not committed. |
| C09 | Separate cognitive and natural endpoints | `compact_replay_only` | Yes: 0.268406/0.025816 cognitive and 0.894428/0.870528 natural | Raw inference and activation banks are not committed. |
| C10 | Behavioral prediction | `compact_replay_only` | Yes: 0.835387 dual history and 0.821201 immediate state | Llama inference is not replayed; prediction rows are committed. |
| C11 | Separate cognitive and natural endpoints | `compact_replay_only` | Yes: 0.648886/0.011162 cognitive and 0.344851/0.318750 natural | Exact selected Llama basis is external/unavailable. |
| C12 | Measurement validity | `compact_replay_only` | Yes: all seven Yes/No task gates approved | Historical interface attempts are compact exports only. |
| C13 | External historical audit | `external_dependency` | Yes: all three reported audit values | External repository and LFS payloads. |

## Clean-checkout result

A candidate snapshot was made from a local clone in `/tmp`, then overlaid with only the proposed working-tree changes. From that isolated path:

~~~text
canonical validation: 22 stages, 27 edges, 43 hashes — PASS
C01–C13 compact replay — PASS
Stage-2B provenance/regression suite — 25 passed
artifact-size policy — PASS
~~~

The manifest test also confirms every hashed evidence artifact is already Git-tracked. The absent Qwen raw condition manifest is recognized as `compact_replay_only`; it is no longer silently required. The unavailable Llama basis remains explicit and was not reconstructed.

## Full test result

Local ordinary CPU command:

~~~bash
pytest -q -p no:cacheprovider -m "not gpu and not model_download"
~~~

Result:

~~~text
167 passed, 2 skipped, 1 warning in 40.44s
~~~

The skips are explicit:

1. Qwen confirmation wording-level reconstruction requires the absent raw condition manifest and reports `compact_replay_only`.
2. The local optional Transformers Llama backend could not load because the installed torchvision lacks its expected `torchvision::nms` operator. The compact Llama behavioral and mechanistic replays remain active and pass.

The warning records a local environment mismatch: the frozen RidgeCV objects were serialized with scikit-learn 1.9.0, while the local baseline environment has 1.7.1. `pyproject.toml` already pins 1.9.0. No model was refit and no saved result changed.

## CI result

`.github/workflows/reproducibility.yml` now performs, on a clean checkout:

1. canonical manifest and artifact-hash validation;
2. the CPU-only provenance/regression suite;
3. C01–C13 compact claim replay;
4. the existing artifact-size/activation-bank check;
5. the complete ordinary CPU test suite.

GPU and model-download markers are registered and excluded from ordinary CI with an explicit workflow comment. The command-equivalent local and isolated-snapshot runs pass. Remote GitHub Actions has not been executed from this uncommitted local branch, so its status is honestly `not_run`; it must not be reported as a remote pass until pushed.

## Discrepancies and remaining historical unknowns

- `artifacts/causal_specificity_v2/gates.json` contains a legacy `passes_level5b: false`/`passed: false` representation, but `necessity_jobs.json` is empty. The owner-approved canonical status is `not_run`; the saved gate file was not rewritten.
- The dual-history and outcome-history shuffled-target controls change zero target values. They remain `uninformative_control`; the saved control rows were not rewritten.
- The core Qwen checkpoint name is known (`Qwen/Qwen3.5-4B`), but its resolved historical revision remains unknown.
- Qwen confirmation compact interventions are committed, but its raw condition manifest is absent.
- Llama compact intervention rows are committed, but the exact selected neural basis object/hash remains external/unavailable.
- The original study remains an external historical dependency.
- Local scikit-learn and optional torchvision environments differ from the recorded/pinned environments as described above.

No canonical artifact SHA-256 changed, and all frozen numerical claims still match.

## Files changed

Stage-2B additions:

- `.github/workflows/reproducibility.yml`
- `provenance/stage2b_baseline.json`
- `scripts/validate_canonical.py`
- `scripts/replay_canonical_claims.py`
- `src/cognitive_discovery/reproducibility/` (manifest, identities, metrics, splits, claims, provenance, replay)
- `tests/provenance/`
- `tests/regression/`
- `STAGE2B_REPORT.md`

Minimal compatibility changes:

- `canonical_manifest.yaml`: replay-availability fields only.
- `pyproject.toml`: registers `gpu` and `model_download` pytest markers.
- `tests/test_confirmation_protocol.py`: explicit compact-only handling for the absent raw manifest.
- `tests/test_llama_runner_compatibility.py`: explicit skip when the optional local Transformers backend cannot import.

The existing artifact-size script and `config/artifact_allowlist.txt` are reused unchanged. No scientific result artifact, runner, public entry point, directory name, or legacy location was modified.

## Stop condition

The RED → GREEN suite is complete. Structural REFACTOR work has not begun. Owner review is required before moving scripts, creating `legacy/`, consolidating pipelines, renaming artifact directories, or changing public entry points.
