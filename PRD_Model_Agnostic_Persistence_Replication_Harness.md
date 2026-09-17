# PRD: Model-Agnostic Replication Harness for Computational-Cognitive Persistence Discovery

**Status:** Proposed  
**Purpose:** Build a reusable pipeline that applies the core computational-cognitive discovery method to a new language model with minimal manual intervention.  
**Primary scientific question:** Can behavioral computational models guide discovery of a causal neural mechanism for structured persistence, and does that mechanism generalize across held-out states and task families?

---

## 1. Motivation

The repository contains a historical Qwen pipeline, supporting diagnostics, later extensions, and a Llama conceptual replication. The next step is not to reproduce every historical analysis. It is to turn the **core scientific method** into a reusable, model-agnostic replication harness.

The harness should test:

**behavioral exploration → computational model comparison → computational counterfactuals → causal neural search → held-out generalization**

The primary mechanistic estimand remains **cognitive counterfactual recovery** throughout.

## 2. Scientific scope

### Core questions

1. Can structured continue/disengage behavior be validly measured in the new model?
2. Are persistence decisions systematically predicted by task variables and history-sensitive computational models?
3. Which candidate computational models best describe persistence, and are multiple theories observationally unresolved?
4. Can counterfactual predictions from frozen computational model(s) guide discovery of a low-dimensional causal neural subspace?
5. Does the frozen mechanism reproduce the same cognitive counterfactual relation on held-out examples and whole task families excluded from neural fitting?
6. Does the controller outperform appropriate random and generic controls?
7. Optionally, if the structured mechanism is robust, does it generalize to a qualitatively different continuation setting such as free-form generation?

### Explicitly out of scope

The following are **not part of the core replication pipeline**:

- Qwen fresh-context follow-up analyses.
- `natural_effect_recovery` as a replication endpoint.
- Independent confirmation of natural source-minus-base effects.
- Any requirement to reproduce later collaborator-led Qwen extensions.
- Any requirement that the neural subspace literally equal a named cognitive variable.
- Any requirement that the same Qwen layer/rank be reused.
- Any requirement that the new model select the same behavioral theory as Qwen.
- Free-generation OOD unless the structured mechanistic result first passes its gates.

These analyses may remain in the repository for provenance, but they must not define the replication harness.

---

## 3. High-level pipeline

```text
Stage 0  Model/interface validation
    ↓
Stage 1  Behavioral design-space collection
    ↓
Stage 2  Computational model comparison
    ↓
Stage 3  Freeze behavioral theory/theories
    ↓
Stage 4  Generate mechanistic counterfactual pairs
    ↓
Stage 5  Search for causal neural subspaces
    ↓
Stage 6  Held-out cognitive-counterfactual generalization
    ↓
Stage 7  Specificity and control comparisons
    ↓
Stage 8  Optional abstraction analysis
    ↓
Stage 9  Optional far-OOD construct-boundary test
```

The pipeline must be restartable stage-by-stage.

---

## 4. Stage 0 — Model and measurement-interface validation

### Goal

Verify that the new model supports a valid semantic continue/disengage measurement before collecting scientific data.

A failed interface is a **measurement failure**, not a failed replication.

### Candidate interfaces

Test a small preregistered set such as:

- Yes / No
- Continue / Stop
- A / B
- X / Y

The adapter/config layer should allow model-specific candidates.

### Required checks

- response tokens are valid under the tokenizer;
- semantic polarity can be reversed;
- reversing the semantic mapping reverses the measured persistence logit;
- wording/order effects are acceptably small;
- both mappings/polarities can be counterbalanced;
- every task family passes a minimum measurement-validity gate.

### Output

```text
replication/<model_id>/interface/
    protocol.json
    candidate_interfaces.csv
    calibration_results.csv
    gates.json
    selected_interface.json
```

### Gate

Proceed only if at least one interface passes all validity gates.

If none pass:

```text
replication_status = measurement_failure
```

and stop.

---

## 5. Stage 1 — Behavioral design-space collection

### Task families

Use the canonical structured persistence battery:

- bandit
- foraging
- solvability
- information sampling
- waiting
- effort
- debugging

### Design variables

Preserve the canonical ontology where applicable:

- continuation value
- disengagement value
- continuation cost
- progress
- success evidence
- uncertainty
- prior investment
- controllability
- environmental stability
- goal/policy continuity
- action history
- outcome history
- contextual history variables where defined

### Sampling

Use broad, counterbalanced behavioral sampling before neural analysis:

```text
broad space-filling sample
→ model comparison
→ optional active/discriminating behavioral sample
→ untouched behavioral validation
```

Reuse the ontology/design code, but generate **fresh model-specific observations**.

### Splits

Freeze:

- `behavior_train`
- `behavior_selection`
- `behavior_test`

Repeated renderings, mappings, wording variants, and semantic-pair members must remain grouped appropriately.

### Output

```text
replication/<model_id>/behavior/
    condition_manifest.parquet
    observations.parquet
    split_manifest.json
    validity_summary.json
```

---

## 6. Stage 2 — Computational model comparison

### Goal

Determine whether persistence is better captured by history-sensitive models than by immediate-state baselines.

### Minimum model bank

- immediate state
- outcome history
- dual history
- latent context

Optional full bank:

- dynamic re-evaluation
- choice perseveration
- latent motivation
- option termination
- task-set reinstatement
- meta-control
- flexible baselines such as MLP/GRU

### Primary behavioral comparison

On untouched behavioral test data report:

- pooled R²
- task-macro R²
- per-task R²
- error/MSE
- optional calibration
- improvement relative to immediate-state baseline

### Replication criterion

Do **not** require the same theory winner as Qwen.

The important question is whether history-sensitive computational structure explains persistence better than an immediate-state baseline and whether one or more candidate models survive for mechanistic testing.

Allow:

```text
behavioral_replication = pass | partial | fail
behavioral_theory_status = resolved | unresolved
```

### Output

```text
replication/<model_id>/behavior/models/
    model_comparison.csv
    per_task_metrics.csv
    selected_models.json
    frozen_models/
```

---

## 7. Stage 3 — Freeze behavioral theory/theories

Behavioral selection must be complete **before** neural fitting.

If one theory clearly passes the preregistered comparison, freeze it.

If multiple theories remain unresolved, freeze the unresolved set and carry them forward. Do not force a winner.

For every frozen behavioral model record:

- file hash
- model/specification ID
- training-data hash
- feature definition
- hyperparameters
- random seed
- split identity

---

## 8. Stage 4 — Generate mechanistic counterfactuals

### Goal

Turn the frozen computational theory into a causal search target.

For a base state `b`, source state `s`, and candidate computational variable `X`:

\[
D^{CF} = f(X_s, Z_b)
\]

where `Z_b` denotes the other base-state variables held fixed.

The target effect is:

\[
\Delta D^{CF} = D^{CF} - D_b
\]

### Pair design

Generate a fresh mechanistic pair set with:

- controlled source/base differences;
- explicit target variable;
- response mapping fixed within pair;
- nuisance variables matched;
- source/base orientation recorded;
- no overlap between neural train/selection/test groups.

### Prospective neural splits

Freeze **before DAS fitting**:

- `neural_train`
- `neural_selection`
- `neural_test`
- `neural_task_holdout`

Recommended fitting tasks:

- bandit
- debugging
- foraging
- solvability

Recommended holdout tasks:

- effort
- information sampling
- waiting

Assignments must be configurable but frozen before neural training.

### Primary endpoint

The only primary mechanistic endpoint is:

```text
cognitive_counterfactual_recovery
```

using:

```text
global_cfr_v1
```

No natural-effect endpoint is required.

---

## 9. Stage 5 — Causal neural search

### Goal

Find a low-dimensional neural subspace whose intervention reproduces the frozen computational counterfactual.

### Model-agnostic depth search

Do not hard-code Qwen layer 28.

Search by relative depth, e.g.:

- 25%
- 50%
- 75%
- 90%

Map fractions to valid residual-stream layers for the target model.

### Rank search

Preregister a small grid, e.g.:

- rank 2
- rank 8

Rank 4 may be included if budget allows, but the grid must be frozen in advance.

### Search protocol

- Fit on `neural_train`.
- Select layer/rank on `neural_selection`.
- Freeze the selected controller.
- Touch `neural_test` and `neural_task_holdout` only after selection.

### Intervention

Use the DAS-style residual subspace interchange:

\[
h'_b = h_b + UU^	op(h_s-h_b)
\]

### Output

```text
replication/<model_id>/mechanism/
    search_grid.json
    train_results.parquet
    selection_results.csv
    selected_controller.safetensors
    selected_controller.json
```

---

## 10. Stage 6 — Held-out cognitive-counterfactual generalization

### Goal

Test whether the **same frozen computational relation** generalizes.

Do not change the estimand after neural selection.

### Primary tests

**A. Held-out within-task examples**

Evaluate cognitive-counterfactual recovery on `neural_test`.

**B. Held-out task families**

Evaluate the same frozen controller on `neural_task_holdout`.

### Metrics

Primary:

\[
CFR_G =
1 -
rac{
\sum_i(\Delta D_i^{neural}-\Delta D_i^{CF})^2
}{
\sum_i(\Delta D_i^{base}-\Delta D_i^{CF})^2
}
\]

Also report:

- intervention vs counterfactual correlation;
- slope/calibration;
- bootstrap confidence interval;
- per-task recovery.

### Suggested gate

On untouched data:

- positive held-out global CFR;
- bootstrap lower bound above zero;
- positive intervention-target correlation;
- performance above matched random-subspace null;
- no dependence on response-mapping artifacts.

Freeze exact thresholds before running the new model.

### Explicit exclusion

Do **not** compute or optimize:

- natural source-minus-base effect recovery;
- fresh-context natural-effect recovery;
- collaborator Qwen confirmation endpoints.

Those are outside the core replication question.

---

## 11. Stage 7 — Specificity and controls

### Required controls

At minimum:

- matched random subspaces;
- shuffled source/base assignments where the target actually changes;
- output/readout direction;
- predictive ridge/probe direction;
- PCA or generic variance subspace;
- direct behavior-target DAS only if intentionally included as a comparator.

### Principle

Controls must answer:

> Is cognitive-counterfactual recovery specific to the discovered controller?

They must not silently switch the target.

### Random controls

- norm matched where appropriate;
- at least 100 random subspaces for development;
- preferably 500 for final claims;
- finite-sample corrected random p-value.

### Target-shuffle integrity

Before scoring a shuffled-target control, verify that the shuffle actually changes target values.

If not:

```text
status = uninformative_control
```

and do not count it as passed or failed.

---

## 12. Stage 8 — Optional abstraction analysis

Run only if the structured causal controller passes the mechanistic replication gates.

### Question

At what abstraction level does the controller operate?

Possible candidates:

- upstream raw history;
- context-transformed history;
- integrated history contribution;
- downstream integrated decision evidence.

### Constraint

Do not assume that the behavioral model's named variable is literally the neural variable.

Interpret:

```text
behavioral theory → causal search target
```

not necessarily:

```text
behavioral variable = neural representation
```

### Outcomes

Allow:

- `candidate_supported`
- `descriptive_only`
- `identity_unresolved`
- `no_candidate_passed`

Do not force a winner.

---

## 13. Stage 9 — Optional far-OOD construct-boundary test

Run only after the structured mechanistic replication is complete.

A free-generation EOS experiment may test whether the structured persistence mechanism generalizes to unconstrained continuation.

This is a **construct-boundary experiment**, not a required replication endpoint.

If run, preregister:

- prompts;
- doses;
- seeds;
- EOS hazard analysis;
- restricted mean generation length;
- degeneration/repetition checks;
- random-subspace control;
- direct-EOS positive control.

A negative result is a boundary on generality, not retroactive invalidation of the structured-task mechanism.

---

## 14. Model adapter architecture

The scientific pipeline must not depend on architecture-specific internals.

Create an interface such as:

```python
class ReplicationModelAdapter:
    def load_model(self): ...
    def load_tokenizer(self): ...
    def render_chat(self, messages): ...
    def validate_response_tokens(self, labels): ...
    def get_response_logits(self, messages, labels): ...
    def num_layers(self): ...
    def get_residual_state(self, messages, layer, position): ...
    def run_with_residual_intervention(
        self,
        messages,
        layer,
        intervention,
    ): ...
    def eos_token_ids(self): ...
```

Implement adapters separately, e.g.:

```text
QwenAdapter
LlamaAdapter
GemmaAdapter
MistralAdapter
```

Shared scientific code must call the adapter rather than branch throughout on model family.

---

## 15. CLI

Target interface:

```bash
python -m cognitive_discovery.replicate_model   --model <hf-model-id>   --revision <immutable-revision>   --adapter <adapter-name>   --config configs/replication/default.yaml   --output artifacts/replications/<run-id>
```

Support resume/stage execution:

```bash
python -m cognitive_discovery.replicate_model   --resume artifacts/replications/<run-id>   --stage behavior
```

```bash
python -m cognitive_discovery.replicate_model   --resume artifacts/replications/<run-id>   --stage mechanism
```

Default to dry-run/validation until `--execute` is provided for expensive stages.

---

## 16. Provenance requirements

Every new replication run must record:

- Git commit and dirty state;
- model ID and immutable revision;
- tokenizer identity;
- adapter version;
- environment/lockfile hash;
- all seeds;
- behavioral-design hash;
- behavioral split hash;
- frozen-model hashes;
- counterfactual pair-manifest hash;
- neural split hash;
- layer/rank grid;
- selected controller hash;
- endpoint ID;
- metric ID;
- output path.

No historical unknowns are acceptable for a **new** replication run.

If a required identity cannot be recorded, fail before expensive execution.

---

## 17. Artifact policy

Do not save full activation banks unless necessary.

Prefer:

- streamed activations;
- compact subspace/basis tensors;
- scalar projections;
- pair-level intervention outcomes;
- manifests;
- aggregate tables.

Large temporary activations should live only in scratch storage and be deleted after compact outputs are written.

Preserve the existing artifact-size safeguards.

---

## 18. TDD and implementation order

Use strict **RED → GREEN → REFACTOR**.

### Phase A — Harness regression target

Before using a genuinely new model, make the generic harness reproduce the existing **core Llama behavioral and cognitive-counterfactual mechanistic summaries** from the same frozen inputs.

Do **not** require reproduction of:

- natural-effect recovery;
- fresh-context Qwen work;
- collaborator confirmation endpoints.

Required regression checks:

- interface selection matches the approved Yes/No Llama interface;
- behavioral model comparison reproduces frozen Llama behavioral metrics within tolerance;
- selected mechanistic layer/rank matches the frozen Llama protocol when using the same frozen search inputs;
- cognitive-counterfactual recovery reproduces frozen familiar and held-out values within tolerance;
- endpoint identity remains cognitive-counterfactual recovery.

### Phase B — New model

After Phase A passes:

1. choose one genuinely new model;
2. freeze its revision;
3. run the complete structured replication pipeline;
4. do not modify gates after seeing results.

---

## 19. Replication report

Automatically produce:

```text
artifacts/replications/<run-id>/REPLICATION_REPORT.md
```

Required sections:

1. model identity;
2. measurement-interface result;
3. behavioral model comparison;
4. frozen behavioral theory status;
5. mechanistic search grid;
6. selected controller;
7. held-out within-task cognitive recovery;
8. held-out task-family cognitive recovery;
9. specificity controls;
10. optional abstraction result;
11. optional OOD boundary result;
12. limitations;
13. exact provenance hashes.

Include a component summary:

```text
Measurement interface:                  PASS
Behavioral history-sensitive structure: PASS
Behavioral theory uniquely resolved:    NO / UNRESOLVED
Low-dimensional causal controller:      PASS
Held-out example generalization:        PASS
Held-out task-family generalization:    PASS
Specificity vs random controls:         PASS
Abstraction identity:                   UNRESOLVED
Far-OOD continuation generalization:    NOT RUN
```

Do **not** collapse these into one overall success score.

---

## 20. Acceptance criteria

The project is complete when:

1. A single model/revision/config can initialize a fresh replication run.
2. The interface gate prevents invalid response measurements from being misclassified as replication failures.
3. Fresh behavioral data can be collected and fit with the computational model bank.
4. Behavioral models are frozen before neural fitting.
5. Counterfactual pair generation is deterministic and hashed.
6. Neural train/selection/test/task-holdout partitions are frozen prospectively.
7. Layer search is model-relative rather than Qwen-specific.
8. The selected controller is evaluated with cognitive-counterfactual recovery on untouched examples and untouched task families.
9. Specificity controls use the same endpoint.
10. Natural-effect/fresh-context collaborator extensions are absent from the core pipeline.
11. All outputs carry complete provenance.
12. No canonical historical artifact is overwritten.
13. The generic harness reproduces the approved Llama cognitive-counterfactual results from frozen inputs before use on a new model.
14. A new model can run end-to-end with one top-level command plus explicit `--execute`.
15. The final report can be understood without reading historical Qwen scripts.

---

## 21. Stop conditions

Stop and request owner review if:

- no valid response interface can be found;
- the new model requires a scientific change to the task ontology rather than an implementation adaptation;
- the behavioral theory cannot produce well-defined counterfactual targets;
- target definitions differ between training and generalization;
- a control silently changes the endpoint;
- a model-specific implementation requires hard-coding a Qwen/Llama assumption into shared scientific code;
- a frozen split or gate would need to be changed after seeing results;
- a required model revision or scientific artifact identity cannot be recorded;
- reproducing the existing Llama cognitive-counterfactual results requires changing frozen scientific outputs.

The correct response is a documented divergence, not post hoc adjustment.

---

## 22. Scientific invariant

The harness should embody one invariant throughout:

\[
oxed{	ext{Generalization preserves the original scientific estimand.}}
\]

If the causal mechanism is discovered by asking whether a neural intervention reproduces a computational model's counterfactual, then the primary generalization test must ask whether the **same neural intervention reproduces the same kind of computational counterfactual on new data and new tasks**.

That is the core replication target.
