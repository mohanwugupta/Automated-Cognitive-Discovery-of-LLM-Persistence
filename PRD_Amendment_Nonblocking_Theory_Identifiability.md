# PRD Amendment: Non-Blocking Theory Identifiability and Behavioral Survivor Sets

**Status:** Proposed amendment  
**Applies to:** `PRD_Computational_Model_Validation_and_Task_Transfer.md`  
**Purpose:** Amend Gate A so computational-model implementation validity remains blocking, while behavioral theory non-identifiability is propagated forward as a survivor set rather than treated as a reason to stop the research pipeline.

## 1. Motivation

The first full synthetic validation run found:

- parameter recovery passed;
- predictive recovery passed;
- `dual_history` teacher recovery = 1.00;
- `latent_context` teacher recovery = 1.00;
- `outcome_history` teacher recovery = 0.40;
- `outcome_history` was selected as `dual_history` in 0.60 of replicates.

Interpret this as limited identifiability between nested or highly similar theories under the current design/selection rule, not as evidence that the implementations are incorrect.

The project should therefore distinguish:

1. **implementation validity** — does the code implement the intended mathematics correctly?
2. **theory identifiability** — can the current behavioral design distinguish among plausible theories?

Only the first should block downstream scientific work.

## 2. Core amendment

Replace the original blocking Gate A with:

```text
Gate A1 — Implementation validity
Gate A2 — Theory identifiability and survivor-set propagation
```

Gate A1 is blocking. Gate A2 is diagnostic and non-blocking.

The governing principle is:

**Behavioral ambiguity should be propagated, not artificially resolved.**

## 3. Gate A1 — Implementation validity

Gate A1 passes only if:

- hand-calculated fixtures pass;
- production and independent reference implementations agree;
- fixed-parameter predictions match expected values;
- counterfactual computations match expected values;
- synthetic parameter recovery satisfies preregistered tolerances;
- no unexplained sign, ordering, normalization, or parameterization discrepancy remains.

Allowed values:

```text
implementation_validity = pass | fail | owner_review
```

If `fail` or `owner_review`, stop. If `pass`, continue regardless of Gate A2.

## 4. Gate A2 — Theory identifiability

### Goal

Characterize which theories the current behavioral design can and cannot reliably distinguish.

This gate is diagnostic, not blocking.

### Required outputs

Produce:

```text
validation/synthetic/model_recovery_matrix.csv
validation/synthetic/model_identifiability.json
```

The JSON should summarize, for every relevant model pair:

```text
teacher_model
confused_with
teacher_recovery_probability
confusion_probability
identifiability_status
notes
```

Recommended statuses:

```text
well_identified
partially_identified
poorly_identified
nested_or_equivalent
not_evaluated
```

For the current outcome-history / dual-history result, record the ambiguity rather than failing the pipeline.

## 5. Behavioral survivor set

### Replace winner-only logic

The behavioral pipeline must no longer assume that exactly one cognitive theory is selected.

Instead produce a frozen:

```text
behavioral_survivor_set
```

containing all theories that remain behaviorally plausible under the preregistered rule.

Example:

```yaml
behavioral_survivor_set:
  - dual_history
  - outcome_history
```

or:

```yaml
behavioral_survivor_set:
  - dual_history
  - latent_context
  - outcome_history
```

### Survivor-set rule

Before any neural analysis, freeze a model-inclusion rule based only on behavioral evidence.

A theory survives if it is either:

1. the best model under the frozen behavioral selection metric; or
2. practically/statistically indistinguishable from the best model under a preregistered equivalence rule.

The rule must not inspect neural results.

Recommended implementation:

For model `m`, relative to best model `m*`:

```text
delta_m = score(m*) - score(m)
```

Retain `m` if the uncertainty interval on `delta_m` is compatible with a preregistered practical-equivalence margin.

The exact metric and margin belong in config and must be frozen before final evidence runs.

Do not tune the margin to preserve a preferred theory.

## 6. Mechanistic search under theory uncertainty

For every theory in the frozen survivor set, run an independent mechanistic search.

Each surviving theory gets:

- its own counterfactual targets;
- its own DAS search;
- its own layer/rank selection;
- its own held-out evaluation;
- its own specificity controls;
- its own transfer matrix where computationally feasible.

Conceptually:

```text
theory
→ cognitive counterfactual target
→ causal subspace search
→ held-out cognitive recovery
→ specificity
→ task transfer
```

### No neural retro-selection

Do not redefine the behavioral winner based on which theory yields the highest neural CFR.

Forbidden:

> “Latent context had the highest DAS CFR, therefore latent context is the true behavioral theory.”

Allowed:

> “Dual history and latent context were behaviorally unresolved; latent context yielded stronger causal recovery under the current neural test.”

Neural evidence is a separate evidential axis.

## 7. Results schema

For every surviving theory, report behavioral and neural evidence side by side:

| Theory | Behavioral support | Within-task CFR | Task-transfer CFR | Specificity | Interpretation |
|---|---:|---:|---:|---|---|
| Dual history | plausible | ... | ... | ... | ... |
| Outcome history | plausible | ... | ... | ... | ... |
| Latent context | plausible / weaker | ... | ... | ... | ... |

Keep these distinct:

```text
behavioral support
causal recovery
cross-task transfer
specificity
```

## 8. Transfer-matrix amendment

The transfer object should allow a theory index:

```text
T[i,j,m] = CFR_G(controller trained on source task i under theory m, evaluated on target task j)
```

where:

- `i` = source task;
- `j` = target task;
- `m` = surviving behavioral theory.

This allows the program to ask:

1. which tasks transfer?
2. which computational descriptions transfer?
3. whether different theories induce different transfer topology.

If compute is limited, prioritize theories in the frozen survivor set and freeze any pruning rule before inspecting transfer outcomes.

## 9. Updated project flow

Replace:

```text
behavioral model comparison
→ select one winning theory
→ mechanistic search
```

with:

```text
behavioral model comparison
→ characterize identifiability
→ freeze behavioral survivor set
→ mechanistic search for each survivor
→ compare causal evidence
→ task-transfer matrix by theory
```

Full flow:

```text
Phase 1   Implementation validation
Phase 1B  Synthetic identifiability characterization
           ↓
           Freeze behavioral survivor-set rule
           ↓
Phase 2   External benchmark
Phase 3   Prospective Qwen replication
Phase 4   Task-transfer matrices for surviving theories
Phase 5   Explain transfer
Phase 6   Cross-model synthesis
```

## 10. Required implementation changes

### 10.1 Update validation status logic

Replace a single blocking status such as:

```text
blocked_owner_review
```

with separate fields:

```text
implementation_validity
identifiability_status
pipeline_permission
```

Example:

```yaml
implementation_validity: pass
identifiability_status: partial
pipeline_permission: continue
```

Do not delete or rewrite the original development report. Add a superseding interpretation/status artifact.

### 10.2 Update validation config

Add explicit config fields:

```yaml
implementation_gate:
  blocking: true

identifiability_gate:
  blocking: false

behavioral_survivor_set:
  metric: <frozen metric>
  equivalence_rule: <frozen rule>
```

Do not alter existing synthetic results.

### 10.3 Update behavioral selection outputs

Replace or supplement:

```text
selected_model
```

with:

```text
best_model
behavioral_survivor_set
behavioral_theory_status
```

where:

```text
behavioral_theory_status = resolved | unresolved
```

Maintain backward compatibility if older code expects `selected_model`.

### 10.4 Update mechanistic orchestration

Iterate over every survivor rather than assuming one selected theory.

Pseudo-code:

```python
for theory in behavioral_survivor_set:
    targets = build_counterfactual_targets(theory)
    controller = fit_das(targets)
    evaluate_within_task(controller, theory)
    evaluate_task_transfer(controller, theory)
    evaluate_specificity(controller, theory)
```

### 10.5 Update reports

All replication and transfer reports must explicitly state whether:

```text
behavioral_theory_status = resolved
```

or:

```text
behavioral_theory_status = unresolved
```

If unresolved, list all surviving theories.

## 11. Tests to add

Use strict RED → GREEN → REFACTOR.

### Survivor-set behavior

- one clearly superior model → one survivor;
- two equivalent models → both survive;
- three equivalent models → all survive;
- clearly inferior model → excluded;
- neural outcomes cannot alter survivor membership.

### Gate behavior

- implementation failure blocks pipeline;
- identifiability ambiguity does not block pipeline;
- ambiguous recovery matrix is serialized correctly;
- pipeline proceeds to later phases when A1 passes.

### Mechanistic orchestration

- every survivor receives a separate counterfactual target;
- every survivor receives a separate neural search;
- result files cannot overwrite one another;
- endpoint remains `cognitive_counterfactual_recovery`.

### Provenance

- survivor-set rule is hashed/frozen;
- survivor-set membership is recorded before neural execution;
- no post-neural mutation of survivor membership is permitted.

## 12. Updated Gate A acceptance criteria

### A1 — Implementation validity

Required:

- hand fixtures pass;
- independent reference implementations agree;
- parameter recovery meets tolerance;
- no unresolved implementation discrepancy remains.

**A1 must pass.**

### A2 — Identifiability characterization

Required:

- model-recovery matrix exists;
- ambiguous model pairs are documented;
- survivor-set logic is implemented;
- ambiguity is propagated rather than forced into one winner.

**A2 does not require universal diagonal recovery.**

The prior requirement:

```text
minimum diagonal recovery probability >= 0.80 for every key model
```

must no longer block downstream work. It may remain as a diagnostic.

## 13. Continue with the rest of the approved implementation

Once this amendment is implemented and tests pass, continue with the remaining approved PRD without waiting for another owner review unless a separate substantive stop condition is encountered.

Proceed in this order:

1. finalize Gate A1/A2 amendment;
2. freeze the external benchmark specification;
3. implement and run the external benchmark;
4. run prospective Qwen self-replication;
5. build task-specific controllers;
6. build task-transfer matrices;
7. parameterize transfer;
8. perform cross-model synthesis.

Stop only for substantive issues such as:

- implementation/reference disagreement;
- benchmark ambiguity requiring an owner scientific choice;
- major prospective Qwen divergence suggestive of a pipeline bug;
- data leakage;
- endpoint drift;
- unapproved compute overrun.

Behavioral theory ambiguity by itself is **not** a stop condition.

## 14. Scientific interpretation rule

Preserve:

```text
predictive superiority != computational identity
```

and downstream:

```text
causal recovery != unique abstraction identity
```

The project should represent uncertainty explicitly at both levels.

A behaviorally unresolved theory set is a valid scientific outcome. A mechanistically heterogeneous theory set is also a valid scientific outcome.

The purpose of the pipeline is not to force a winner. It is to determine which computational descriptions remain plausible, which have causal support, and where their mechanisms generalize.
