# Stage 2 refactor plan

This is the owner-review plan for the next phase. No scientific code is moved or substantially refactored in Stage 2A. The approved base is `origin/main` at `3a5c8749d881632beccd2f12cc1016992ccc358f`; `origin/computational-discrimination-audit` stays separate.

## Governing sequence

Every canonical stage follows strict RED → GREEN → REFACTOR:

1. **RED:** Add a narrowly scoped regression or provenance test and demonstrate that it fails for the missing safeguard, ambiguous identity, or unreproducible input.
2. **GREEN:** Add the smallest configuration, metadata, validation, or compatibility layer needed to make that test pass without changing a saved scientific result.
3. **REFACTOR:** Only after the frozen-result suite is green, move or modularize code. Run all regression tests after each movement.

Scientific discrepancies are recorded, never repaired by changing one analysis to agree with another. Large activations remain excluded from Git.

## Phase 0 — freeze the review baseline

Before adding tests:

- Tag or record commit `3a5c8749d881632beccd2f12cc1016992ccc358f` as the Stage 2A scientific baseline.
- Preserve `REPO_AUDIT.md`, `CANONICAL_PIPELINE.md`, `CLAIMS.md`, and `canonical_manifest.yaml` as reviewed provenance inputs.
- Record the exact diff, if any, between the baseline and future configurations. Configuration extraction must not change defaults silently.
- Do not merge `origin/computational-discrimination-audit`. Its only adopted conclusions are the two owner decisions listed in the canonical manifest.

## RED suite required before any code movement

The proposed tests live under `tests/provenance/` and `tests/regression/`. A RED test must fail for the stated reason—not because of an unrelated missing package, GPU, or network call. All Stage 2 RED tests should be CPU-only and use committed compact artifacts.

### 1. Manifest schema and graph identity

**Test:** `tests/provenance/test_canonical_manifest_schema.py`

Assertions:

- Every stage has a known status and every dependency resolves.
- The graph is acyclic.
- Every claim-bearing stage identifies its model entry, target/estimand, metric, outputs, and availability.
- Historical unknowns are represented as explicit `null` plus an identity-status explanation.
- `computational_discrimination_audit` is declared separate and unmerged.

Expected RED condition: there is not yet a repository validator/schema that enforces these invariants.

GREEN criterion: a small schema/validator validates `canonical_manifest.yaml` without importing scientific pipelines.

### 2. Committed artifact hashes

**Test:** `tests/provenance/test_canonical_artifact_hashes.py`

Assertions:

- Every non-null SHA-256 in the manifest matches the file in a clean checkout.
- Missing artifacts are allowed only when `availability` or `identity_status` explicitly records why they are absent.
- Hashing is byte-level and deterministic; directories are never treated as identities without a manifest.

Expected RED condition: no current test checks the canonical artifact registry end to end.

GREEN criterion: all committed lightweight artifacts match and every uncommitted dependency is surfaced explicitly.

### 3. Metric identity and old-CFR exclusion

**Test:** `tests/regression/test_canonical_metric_identity.py`

Assertions:

- `global_cfr_v1` exactly equals `1 - SSE(observed,target) / SSE(0,target)` on hand-computable fixtures.
- Grouped/bootstrapped recomputation uses the same aggregate numerator and denominator.
- A mean of per-example CFR values produces a deliberately different fixture result and cannot satisfy the canonical metric selector.
- Core Level-4, Qwen confirmation, fresh-context, and Llama mechanistic reports declare `global_cfr_v1`.

Expected RED condition: metric naming/version dispatch is not centralized and legacy per-example fields remain easy to select accidentally.

GREEN criterion: an explicit metric registry/version check protects the existing implementation and rejects the legacy interpretation for canonical claims.

### 4. Cognitive versus natural target identity

**Test:** `tests/regression/test_recovery_endpoint_identity.py`

Assertions:

- Cognitive recovery consumes the frozen behavioral predicted effect column only.
- Natural recovery consumes the unmodified-model source-minus-base effect column only.
- Renaming, swapping, pooling, or silently falling back between target columns raises an error.
- Qwen confirmation reproduces cognitive CFR values 0.251/-0.428 separately from natural CFR values 0.779/0.816 within frozen tolerances.
- Llama reports cognitive and natural endpoints separately.

Expected RED condition: current analysis scripts distinguish these endpoints procedurally but no cross-pipeline guard prevents accidental substitution.

GREEN criterion: endpoint names and required source columns become explicit configuration fields with validation.

### 5. Behavioral target definition

**Test:** `tests/regression/test_persistence_target_identity.py`

Assertions:

- Persistence logit is computed from the intended continue and disengage tokens after the documented two-token renormalization.
- Reversing token order flips sign and fails the expected identity.
- Qwen historical unknown token/checkpoint metadata remains unknown rather than borrowing the later checkpoint.
- Llama uses the validated Yes/No semantic mapping, including polarity reversal where specified.

Expected RED condition: token identities and semantic polarity are distributed across runners/protocols rather than enforced by one versioned target definition.

GREEN criterion: stage configurations pin target-definition IDs and interfaces without altering stored observations.

### 6. Split and holdout semantics

**Test:** `tests/regression/test_split_holdout_semantics.py`

Assertions:

- Source/base members, both mappings, prompt wordings, and all rows for a semantic pair remain in the same split/bootstrap cluster where the protocol requires it.
- `causal_mech_v1` neural fitting tasks are bandit, debugging, foraging, and solvability; effort, information sampling, and waiting remain task holdouts.
- Behavioral-model fitting data are not mislabeled as neural-controller fitting data.
- Qwen confirmation familiar and held-out semantic identities do not overlap.
- Llama train, selection, final-test, familiar-neural, and held-out-neural roles remain distinct.

Expected RED condition: no single regression test spans the separate split builders and compact exports.

GREEN criterion: versioned split manifests and shared invariant checks reproduce all frozen counts and memberships.

### 7. Pair and design identity

**Test:** `tests/regression/test_pair_manifest_identity.py`

Assertions:

- The core pair manifest hash is `6591f8430e52232932e56f79a768409363ddece9c9da0bcd1b6bd089b4e266e4`.
- Predicted effects hash is `dc87d567f8560c079f6671c24d25df81f13f6990b795daf9f7ac4f44ef681351`.
- Expected core split and target-family counts match the frozen inventory.
- Abstraction contrasts retain their frozen semantic IDs and split roles.
- A duplicated or previously observed active-discovery design is rejected by stable semantic identity rather than row order.

Expected RED condition: some identities are implicit in Parquet content and are not asserted by one canonical test.

GREEN criterion: identity functions and frozen inventories are shared without changing generated pairs.

### 8. Neural basis, layer, and rank identity

**Test:** `tests/regression/test_neural_object_identity.py`

Assertions:

- The primary Qwen controller is layer 28, rank 2, SHA-256 `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`.
- Loading a different layer, rank, target, or basis fails before analysis.
- The Qwen confirmation references that frozen basis rather than a refitted controller.
- The missing Llama basis hash produces an explicit availability result, not a fabricated identity or accidental pass.

Expected RED condition: identity checks are stage-local and the Llama compact export cannot prove basis-byte identity.

GREEN criterion: a shared neural-artifact reference structure verifies Qwen and clearly marks Llama as externally unavailable.

### 9. Model/checkpoint revision identity

**Test:** `tests/provenance/test_model_revision_identity.py`

Assertions:

- Qwen confirmation requires revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Llama replication requires revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- The core Qwen revision remains `null/historical_unknown` and cannot inherit either later revision.
- Runtime provenance records the resolved revision returned by the model loader and fails on mismatch for new runs.

Expected RED condition: historical and newer model identities are recorded in heterogeneous locations and loaders do not share one fail-closed check.

GREEN criterion: explicit experiment configurations and runtime provenance enforce pinned revisions for new runs while preserving honest historical unknowns.

### 10. Behavioral-model hash identity

**Test:** `tests/regression/test_behavioral_model_hash_identity.py`

Assertions:

- File hashes for dual history, latent context, and outcome history match the canonical registry.
- In-memory behavioral hashes, specs, parameters, and training-condition hash agree with `behavioral_hash.json`.
- A model with the right class but different parameters or training-condition identity is rejected.
- Qwen and Llama behavioral objects cannot be interchanged.

Expected RED condition: some stages check filenames or architecture labels without a uniform object-and-training-data identity contract.

GREEN criterion: a shared behavioral-object descriptor validates all fields before causal analysis.

### 11. Claim/gate consistency

**Test:** `tests/provenance/test_claim_gate_consistency.py`

Assertions:

- `theory_resolution_v1` is unresolved and has no winner.
- Level-5B status is `not_run`; an empty job list cannot be translated to failed.
- Target-preserving dual-history/outcome-history shuffle controls are `uninformative_control`; they cannot be counted as passed or failed.
- Abstraction E is strongest descriptive candidate while the full identity gate remains false/unresolved.
- OOD is `boundary` and cannot be cited as establishing E identity.
- Mechanistic-v1 probe/steering interpretation is supporting/legacy, whereas its matched manifest is canonical.
- Failed early Llama labels are measurement-validity history, not negative replication evidence.

Expected RED condition: these owner decisions are presently prose/inventory conventions, not fail-closed executable assertions.

GREEN criterion: claims and figures are generated or checked against stage-status fields and saved gates.

### 12. Frozen numerical results

**Test:** `tests/regression/test_frozen_claim_values.py`

Assertions:

- The three core selected-controller test/task-holdout CFR values and intervals reproduce from compact inputs.
- Qwen confirmation reproduces natural 0.779/0.816 and cognitive 0.251/-0.428 at frozen precision.
- Llama behavioral final-test scores reproduce from committed predictions.
- Llama mechanistic cognitive 0.649/0.011 and natural 0.345/0.319 reproduce from committed interventions.
- Abstraction and OOD gate summaries reproduce from their compact tables.

Expected RED condition: several scripts can replay subsets, but there is no one CPU regression suite with explicit numerical tolerances and endpoint labels.

GREEN criterion: compact-artifact replay tests pass without a GPU or network and without rewriting committed outputs.

### 13. Clean-clone availability contract

**Test:** `tests/provenance/test_clean_clone_availability.py`

Assertions:

- Clone only tracked files into a temporary directory and run all CPU claim replays there.
- Each replay reports one of: `fully_replayable`, `compact_replay_only`, `external_dependency`, or `historical_unknown`.
- Qwen confirmation flags its absent raw condition manifest while successfully replaying compact arithmetic.
- Llama mechanistic flags the absent selected basis while successfully replaying compact arithmetic.
- The original study reports an external-repository/LFS dependency.
- No test silently reads an untracked local artifact.

Expected RED condition: at least one existing Qwen test expects an uncommitted condition manifest, and local worktrees can mask missing tracked inputs.

GREEN criterion: clean-clone CI passes the declared availability contract; full-replay tests are separately skipped with a structured reason when inputs are intentionally external.

### 14. Provenance record completeness for new runs

**Test:** `tests/provenance/test_run_provenance_contract.py`

Assertions for any newly generated canonical run:

- Git commit and dirty-tree state.
- Model repository and resolved revision.
- All seeds.
- Dataset, pair, and split hashes.
- Behavioral-model file and object hashes.
- Neural artifact hash, layer, rank, and target definition.
- Metric ID/version and endpoint ID.
- Configuration hash and output directory.

Expected RED condition: historical run metadata is heterogeneous and omits some required fields.

GREEN criterion: new runs fail before expensive execution if required provenance cannot be resolved. Historical runs retain explicit unknowns rather than backfilled guesses.

## GREEN implementation order

After every test above has been introduced and observed RED for its intended reason:

1. Add a minimal manifest schema and read-only validator.
2. Extract explicit configuration files for each canonical/replication stage, preserving every current default byte-for-byte where known.
3. Add endpoint, metric, split, behavioral-object, and neural-object identity descriptors.
4. Add a provenance writer used only by new executions; do not rewrite historical metadata as though it had been captured at runtime.
5. Add CPU replay adapters that consume committed compact exports without mutating them.
6. Add clean-clone CI jobs for provenance validation and compact arithmetic replay.
7. Resolve the untracked Qwen condition-manifest issue by either committing a lightweight identity-safe manifest or changing the test to the explicitly reviewed compact-availability contract. This requires owner review if the manifest contains sensitive or large content.
8. For the missing Llama basis, either recover and hash the exact external object or retain `unknown_external_archive`; never reconstruct a look-alike basis and call it identical.

## REFACTOR only after GREEN

Once the full frozen suite is green:

- Introduce stable modules for configuration, identities/hashes, split validation, endpoint selection, metrics, and provenance.
- Convert scripts into thin entry points while keeping old command wrappers during a deprecation window.
- Move superseded scripts and their documentation under `legacy/` without deleting them or changing artifact paths referenced by old metadata.
- Keep the mechanistic-v1 matched-condition manifest in the canonical dependency path even if its probe/steering runners move to `legacy/`.
- Create one documented command per major result and one top-level core-pipeline command.
- Write `PIPELINE.md`, `REPRODUCING_RESULTS.md`, and stage READMEs for behavioral, mechanistic, and generalization work.
- Preserve compact tables, bases, protocols, and hashes; do not commit large activations.

## Refactor stop conditions

Stop and request owner review if any of the following occurs:

- A frozen numerical result changes beyond its declared tolerance.
- A target, split, pair identity, checkpoint, basis, layer, rank, or metric cannot be matched exactly.
- A historical unknown would need to be guessed.
- A clean-clone fix would require committing a large artifact.
- A script that appears duplicated yields scientifically different outputs under the same apparent configuration.
- The unmerged audit branch would need to be merged to make progress.

The correct response to any such discrepancy is a documented divergence, not alteration of a result to force agreement.
