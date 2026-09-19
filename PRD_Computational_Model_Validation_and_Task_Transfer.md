# PRD: Computational-Model Validation and Task-Transfer Mapping

**Status:** Proposed  
**Audience:** Coding/research agent  
**Primary objective:** Validate that the computational models are implemented correctly, verify that the prospective replication harness reproduces Qwen under the same modern pipeline, and then explain why causal persistence controllers transfer across some task families but not others.

---

## 1. Scientific motivation

Recent prospective replications support a qualified cross-model conclusion:

- history-sensitive persistence replicates behaviorally;
- low-rank causal control can replicate within familiar task families;
- a universal controller that transfers to entirely held-out task families does not reliably replicate.

The next question is therefore not simply:

> “Is there a universal persistence representation?”

It is:

> **What computational structure determines when a causal persistence mechanism generalizes across tasks and models?**

Before pursuing that question, the project must validate a more basic methodological assumption:

> **Are the computational models themselves implemented correctly?**

A reproducible pipeline can reproduce a systematic implementation mistake. Therefore this extension has four linked goals:

1. validate the computational models;
2. self-replicate Qwen with the new prospective harness;
3. build task-by-task causal transfer matrices;
4. explain transfer using behavioral, computational, task, and neural features.

---

## 2. Core scientific hypotheses

The project should distinguish the following possibilities.

### H1 — Shared computation, shared neural coordinates

The same computational variable and similar neural subspace support persistence across tasks:

\[
f_i \approx f_j
\qquad\text{and}\qquad
U_i \approx U_j.
\]

Prediction:

- strong off-diagonal transfer;
- similar controllers across tasks;
- transfer weakly dependent on task identity.

### H2 — Shared computational role, task-specific neural coordinates

Tasks instantiate a similar computational function but encode it in different neural subspaces:

\[
f_i \approx f_j
\qquad\text{but}\qquad
U_i \neq U_j.
\]

Prediction:

- task-specific controllers work;
- zero-shot transfer fails;
- controller geometry differs by task;
- an alignment or task-conditioned mapping may recover transfer.

### H3 — Family of related computations

Tasks share ingredients but combine them differently:

\[
D_t = f_t(H,\text{progress},\text{cost},\text{context},\ldots).
\]

Prediction:

- partial transfer;
- transfer clusters by behavioral/computational similarity;
- task-pair features predict transfer.

### H4 — Persistence fractures across task families

The ordinary-language label “persistence” groups together behaviors that do not share one common computational mechanism.

Prediction:

- weak or idiosyncratic transfer;
- task-specific computational models;
- limited transfer structure beyond task identity.

The project should distinguish these hypotheses rather than assume H1.

---

## 3. Scope

### In scope

- Mathematical specification of every computational model used in the core pipeline.
- Hand-calculated unit tests.
- Synthetic parameter recovery.
- Synthetic model recovery.
- Independent reference implementations.
- One external published-model benchmark.
- Prospective rerun of Qwen through the new replication harness.
- Pairwise task-to-task controller transfer matrices.
- Multi-task/subset-to-task transfer.
- Leave-one-task-out transfer.
- Parameterized prediction of transfer using behavioral, computational, task, and neural features.
- Cross-model comparison across Qwen, Gemma, and Llama.
- Complete provenance and clean-worktree execution.

### Out of scope

- Natural-effect recovery.
- Fresh-context collaborator-led Qwen extensions.
- Reproduction of every historical analysis in the repository.
- Free-generation OOD unless separately approved.
- Neural necessity unless separately approved.
- Forcing one universal behavioral theory.
- Forcing one universal neural controller.
- Treating transfer failure as failure of the overall computational-cognitive methodology.

---

## 4. High-level work plan

```text
Phase 1  Computational-model validation
   ↓
Phase 2  External benchmark validation
   ↓
Phase 3  Qwen self-replication with prospective harness
   ↓
Phase 4  Task-specific and task-pair transfer matrix
   ↓
Phase 5  Parameterized explanation of transfer
   ↓
Phase 6  Cross-model synthesis
```

Each phase must be independently reviewable and restartable.

---

## 5. Phase 1 — Computational-model validation

### Goal

Establish that every model used by the automated discovery pipeline matches its intended mathematical specification and can be recovered under controlled conditions.

The validation pyramid is:

\[
\boxed{
\text{hand-calculated tests}
\rightarrow
\text{synthetic recovery}
\rightarrow
\text{independent implementation}
\rightarrow
\text{external benchmark}
}
\]

The first three layers are mandatory before new scientific inference.

### 5.1 Freeze mathematical specifications

Create:

```text
docs/computational_models/
    immediate_state.md
    outcome_history.md
    dual_history.md
    latent_context.md
    choice_perseveration.md
    ...
```

Each specification must include:

- exact equations;
- variable definitions;
- sign conventions;
- history ordering;
- decay convention;
- initialization;
- normalization/standardization;
- task-specific vs shared parameters;
- mapping nuisance terms;
- regularization/fitting objective;
- prediction function;
- counterfactual function;
- fixed constants.

**The coding agent must not infer equations from production code.**  
The written mathematical specification is authoritative.

### 5.2 Hand-calculated fixtures

For every model, construct tiny examples that can be calculated manually.

Example:

\[
H_t = \sum_{k=1}^{K}\lambda^{k-1}r_{t-k}.
\]

Test:

- positive/negative histories;
- empty history;
- one-element history;
- different decay values;
- reversed history ordering;
- contextual/history-switch cases;
- response-mapping nuisance features where applicable.

Required tests:

```text
tests/model_validation/test_<model>_hand_fixtures.py
```

Production implementation must match the hand-calculated oracle to numerical tolerance.

### 5.3 Synthetic parameter recovery

For each model \(M_k\):

1. choose multiple parameter settings \(	heta^*\);
2. generate synthetic observations from \(M_k(	heta^*)\);
3. fit the same model;
4. recover parameters and predictions.

Report:

- parameter bias;
- RMSE;
- predictive R²;
- recovery vs sample size;
- recovery under realistic noise.

Artifact:

```text
validation/synthetic/parameter_recovery.csv
```

### 5.4 Synthetic model recovery

Generate datasets from every candidate model and fit the complete model bank blind.

Construct:

\[
R_{ij}
=
P(\hat M=j\mid M_{\text{true}}=i).
\]

Artifact:

```text
validation/synthetic/model_recovery_matrix.csv
```

Questions:

- Can dual history be distinguished from outcome history?
- Can latent context be distinguished from dual history?
- Under which designs do they become observationally equivalent?
- How much data is needed for identification?
- Are some theories structurally non-identifiable under the current battery?

This result must constrain interpretation of unresolved behavioral comparisons.

---

## 6. Phase 1B — Independent reference implementations

### Goal

Protect against production code that consistently implements the wrong mathematics.

For each core model, create a simple reference implementation.

Rules:

- reference code may read only the mathematical specification;
- do not import production model classes;
- do not reuse production feature builders;
- prioritize clarity over efficiency;
- no inheritance from production code;
- no shared helper functions for core equations.

Suggested path:

```text
validation/reference_models/
    immediate_state_reference.py
    outcome_history_reference.py
    dual_history_reference.py
    latent_context_reference.py
```

### Equivalence tests

Feed identical raw histories/conditions to production and reference implementations.

Compare:

- constructed features;
- predicted values under fixed parameters;
- counterfactual predictions;
- fitted predictions where fitting algorithms are intentionally identical.

Required:

```text
tests/model_validation/test_reference_equivalence.py
```

Any disagreement stops the project until resolved.

---

## 7. Phase 2 — External literature benchmark

### Goal

Show that the modeling/fitting infrastructure can reproduce at least one independently published computational-history result.

This validates:

- history-kernel construction;
- behavioral fitting;
- model comparison;
- parameter recovery;
- reporting.

### Preferred benchmark profile

Choose one published study with:

- an explicit perseverance / choice-history model;
- open behavioral data;
- a clear model-comparison or simulation result;
- sufficient methodological detail for independent implementation.

A perseverance/choice-history study such as Sugawara & Katahira is a suitable candidate **if** the required data and methods are accessible.

Before implementation, create:

```text
validation/external_benchmark/BENCHMARK_SPEC.md
```

Freeze:

- exact paper;
- dataset;
- subset;
- equations;
- preprocessing;
- fitting method;
- target result(s);
- tolerance;
- success criterion.

### Success criterion

Do not require exact numerical equality unless warranted.

Preregister one or two robust targets, such as:

- perseverance/history model improves fit in the reported direction;
- a key parameter/sign pattern is reproduced;
- synthetic recovery is reproduced;
- qualitative model-selection result is reproduced.

### Isolation

The benchmark implementation must not modify project models to force agreement.

If the published model differs from the project model, implement it separately.

Required output:

```text
validation/external_benchmark/
    BENCHMARK_SPEC.md
    results.csv
    REPLICATION_REPORT.md
```

---

## 8. Phase 3 — Qwen self-replication using the new harness

### Goal

Determine whether the new prospective replication pipeline reproduces the original Qwen phenomenon under the same modern methodology used for Gemma and Llama.

This is **not** an exact historical replay because the original core Qwen revision was not pinned.

Use the pinned Qwen revision now used by the reproducible follow-up infrastructure.

Interpret this as:

```text
pipeline self-replication
```

not:

```text
byte-for-byte historical replication
```

### Requirements

Run Qwen from scratch through the prospective harness:

1. measurement-interface validation;
2. behavioral collection;
3. computational model comparison;
4. behavioral theory freeze;
5. prospective neural pair construction;
6. neural train/selection/test/task-holdout split;
7. relative-depth × rank causal search;
8. held-out cognitive-counterfactual recovery;
9. specificity controls.

Do not reuse:

- historical fitted behavioral models;
- historical DAS bases;
- historical selected layer/rank;
- historical counterfactual pairs.

### Comparison table

| Quantity | Historical Qwen | Prospective Qwen |
|---|---:|---:|
| Interface | | |
| History-sensitive behavioral gain | | |
| Surviving theory/theories | | |
| Selected relative depth | | |
| Selected rank | | |
| Familiar-task CFR | | |
| Held-out-task CFR | | |
| Random-control specificity | | |

### Interpretive fork

If prospective Qwen **also fails whole-task transfer**:

> The older task-general Qwen result was likely contingent on the earlier design/analysis pipeline.

If prospective Qwen **retains whole-task transfer** while Gemma and Llama fail:

> Cross-task neural transfer is genuinely model-dependent.

This distinction is central.

---

## 9. Phase 4 — Task-transfer matrix

### Goal

Replace binary “task holdout” with a structured map of which task mechanisms transfer to which other tasks.

For a controller trained on task \(i\) and evaluated on task \(j\):

\[
T_{ij}
=
CFR_G(U_i\rightarrow j).
\]

Compute separately for each viable computational theory where appropriate.

### 9.1 Primary 7 × 7 matrix

Tasks:

1. bandit
2. foraging
3. solvability
4. information sampling
5. waiting
6. effort
7. debugging

Train a controller separately on each source task.

Evaluate on all target tasks.

Produce:

```text
transfer/<model>/single_task_transfer.csv
```

Fields:

```text
model
theory
source_task
target_task
layer
rank
n_train
n_test
global_cfr
correlation
slope
bootstrap_low
bootstrap_high
random_p
```

Diagonal values = within-task generalization.  
Off-diagonal values = transfer.

### 9.2 Controller fitting policy

A source-task controller uses only:

- source-task neural train examples;
- source-task selection examples for layer/rank selection.

Target-task data must never participate in fitting or selection for zero-shot transfer.

### 9.3 Multi-task transfer

Also evaluate:

#### Pair-trained controllers

\[
U_{\{i,k\}}\rightarrow j.
\]

Use preregistered pairs, or all pairs if compute permits.

#### Leave-one-task-out

\[
U_{-j}\rightarrow j.
\]

Train on all tasks except \(j\), test on \(j\).

#### Training-diversity curves

For target task \(j\), vary source-task count:

\[
1,2,3,\ldots,6
\]

and estimate whether transfer improves with diversity.

This distinguishes:

```text
"needs diverse neural training"
```

from:

```text
"uses genuinely different neural computation."
```

---

## 10. Controller-geometry analysis

For task-specific controllers \(U_i\) and \(U_j\), quantify:

- principal-angle similarity;
- subspace overlap;
- canonical correlations;
- Procrustes alignment where appropriate;
- cross-projection of task-specific causal effects;
- layer-depth correspondence.

Question:

> Do tasks fail to transfer because the same computation occupies different neural coordinates?

If task-specific controllers both work but \(U_i\rightarrow j\) fails, test whether an explicit alignment \(R_{ij}\) improves transfer.

Any alignment must be fitted without using target-task test outcomes.

---

## 11. Phase 5 — Parameterize transfer

### Goal

Explain the transfer matrix rather than merely visualize it.

For task pair \(i,j\), construct:

\[
z_{ij}
=
[
\text{behavioral similarity},
\text{computational similarity},
\text{task ontology similarity},
\text{counterfactual similarity},
\text{neural geometry similarity}
].
\]

Predict:

\[
T_{ij}
=
\beta_0+\beta^\top z_{ij}+u_i+v_j+\epsilon_{ij}.
\]

Use regularized or hierarchical models because task-pair sample size is limited.

### 11.1 Candidate predictors

#### Behavioral-model similarity

- cosine/distance between fitted coefficient vectors;
- same selected/surviving theory;
- outcome-history sensitivity difference;
- action-history sensitivity difference;
- continuation-value sensitivity difference;
- disengagement-value sensitivity difference;
- progress sensitivity difference;
- success-evidence sensitivity difference;
- cost sensitivity difference.

#### Task ontology

- terminal vs reversible disengagement;
- whether continuation produces new information;
- explicit continuation cost;
- reward stochasticity;
- temporal horizon;
- controllability;
- environmental stability;
- prior-investment relevance;
- explicit progress signal;
- explicit success/failure feedback;
- stay/leave vs retry/give-up vs wait/stop structure.

#### Counterfactual geometry

- similarity of \(\Delta D^{CF}\) distributions;
- effect-size variance;
- target sign balance;
- source/base state similarity.

#### Neural properties

- task-specific controller layer;
- controller rank;
- principal-angle overlap;
- representational similarity before DAS fitting.

### 11.2 Avoid overfitting

There are only 42 directed off-diagonal task pairs per model.

Therefore:

- preregister predictor families;
- do not fit dozens of unconstrained predictors;
- use leave-one-target-task or comparable held-out validation;
- use regularization/hierarchical shrinkage;
- report uncertainty;
- prefer predictor-family conclusions over exploratory coefficient fishing.

Across models, pool with model-level random effects where justified.

---

## 12. Transfer clustering and discovered ontology

Use transfer structure to discover task groupings.

Possible methods:

- hierarchical clustering on symmetric transfer;
- spectral clustering;
- multidimensional scaling;
- graph community detection;
- low-rank factorization.

This is exploratory.

Purpose:

> Let causal transfer reveal the model's task ontology.

Do not force predefined clusters.

---

## 13. Phase 6 — Cross-model synthesis

After Qwen, Gemma, and Llama have transfer matrices, ask:

1. Is transfer structure correlated across models?
2. Do the same task pairs transfer?
3. Do models differ mainly in transfer magnitude or topology?
4. Is task similarity more predictive than model family?
5. Are latent-context controllers especially transferable?
6. Does controller layer/rank predict transfer?
7. Do some task pairs consistently fail across all architectures?

Compute cross-model matrix correlations with task-pair bootstrap or permutation tests.

Do not claim universal structure unless it replicates prospectively.

---

## 14. Required artifacts

```text
validation/
    computational_models/
    synthetic/
    reference_models/
    external_benchmark/

replications/
    qwen-prospective/
    gemma-.../
    llama-.../

transfer/
    qwen/
    gemma/
    llama/
    cross_model/
```

Each model transfer directory should contain:

```text
single_task_transfer.csv
leave_one_task_out.csv
training_diversity.csv
controller_geometry.csv
transfer_predictors.csv
TRANSFER_REPORT.md
```

---

## 15. Visualization requirements

At minimum:

### Figure 1 — Computational-model validation
- synthetic model-recovery confusion matrix;
- parameter-recovery summary.

### Figure 2 — Prospective Qwen self-replication
- behavioral comparison;
- familiar vs held-out cognitive CFR.

### Figure 3 — Task transfer matrices
One matrix per model with identical ordering.

### Figure 4 — Cross-model transfer comparison
Task-pair transfer consistency across models.

### Figure 5 — What predicts transfer?
Observed vs predicted held-out transfer plus preregistered predictor-family effects.

### Figure 6 — Controller geometry vs causal transfer
Subspace similarity vs transfer.

Figures must read frozen artifact tables; never manually enter values.

---

## 16. Provenance requirements

Every new scientific run must record:

- Git commit;
- `git_dirty`;
- exact diff hash if dirty;
- model ID;
- immutable revision;
- tokenizer revision;
- environment/lockfile hash;
- model/specification hashes;
- train/selection/test identities;
- pair-manifest hash;
- controller hash;
- layer;
- rank;
- endpoint;
- metric;
- seeds;
- output root.

### Clean-worktree rule

Final manuscript-grade replication and transfer runs should require:

```text
git_dirty = false
```

unless an exact diff is archived and explicitly approved.

Development runs may be dirty but cannot be promoted to final evidence.

---

## 17. Compute/storage constraints

- Never save full activation banks by default.
- Stream activations during DAS search.
- Persist only:
  - low-rank bases;
  - scalar outcomes;
  - small projection tables;
  - pair manifests;
  - compact metrics.
- Use scratch activations only temporarily.
- Reuse model loading across controllers when safe.
- Cache tokenized prompts.
- Parallelize source-task × target-task evaluation without changing seeds or split identity.
- Estimate GPU-hours from a one-task pilot before the full matrix.

---

## 18. TDD requirements

Use strict:

```text
RED → GREEN → REFACTOR
```

Mandatory tests:

1. mathematical hand fixtures;
2. synthetic parameter recovery;
3. synthetic model recovery;
4. production/reference equivalence;
5. frozen Llama regression;
6. frozen Gemma regression;
7. prospective Qwen harness test;
8. transfer-matrix leakage tests;
9. endpoint identity;
10. provenance completeness.

No scientific refactor before corresponding regression tests exist.

---

## 19. Stage gates

### Gate A — Computational-model validity

Required before new interpretation:

- all hand fixtures pass;
- reference implementations agree;
- synthetic parameter recovery is acceptable;
- model-recovery matrix is understood;
- non-identifiable model pairs are documented.

### Gate B — External benchmark

Required before declaring the cognitive-model infrastructure independently validated.

If it fails, diagnose:

- preprocessing;
- mathematical specification;
- optimizer;
- likelihood;
- model comparison;
- reporting.

Do not tune project models to force agreement.

### Gate C — Prospective Qwen

Required before interpreting cross-model differences strongly.

### Gate D — Task-specific causal recovery

A task contributes as a **source** in transfer analysis only if its own task-specific controller shows interpretable within-task recovery.

If source task \(i\) has no valid controller, off-diagonal transfer from \(i\) is `unavailable`, not zero.

### Gate E — Transfer explanation

Fit transfer-prediction models only after transfer matrices and predictor definitions are frozen.

---

## 20. Main interpretation logic

### Possible result 1
Task-specific controllers work and transfer broadly.

Interpretation:

> Evidence for a shared task-general neural persistence computation.

### Possible result 2
Task-specific controllers work, but zero-shot transfer fails and geometry differs.

Interpretation:

> Computational roles recur across tasks but are implemented in task-specific neural coordinates.

### Possible result 3
Transfer clusters by behavioral/computational similarity.

Interpretation:

> Persistence is a family of related computations rather than one universal computation.

### Possible result 4
Little structured transfer remains after validation.

Interpretation:

> The ordinary-language construct “persistence” fractures into distinct model computations across tasks.

None of these should be treated as experimental failure.

---

## 21. Acceptance criteria

The project is complete when:

1. Every core computational model has a frozen mathematical specification.
2. Hand-calculated fixtures pass.
3. Synthetic parameter recovery passes preregistered tolerances.
4. A full model-recovery confusion matrix is produced.
5. Independent reference implementations agree with production code.
6. One external published computational-history result is independently reproduced to preregistered criteria.
7. Qwen is rerun prospectively from fresh observations.
8. Qwen/Gemma/Llama are compared under the same pipeline.
9. A 7×7 transfer matrix is produced for every model with sufficient valid task-specific controllers.
10. Leave-one-task-out and training-diversity transfer are evaluated.
11. Task-pair predictors are frozen before explanatory modeling.
12. Transfer-prediction models use held-out validation.
13. Cross-model transfer topology is quantified.
14. Final evidence runs are clean-worktree or carry an approved archived diff.
15. No natural-effect/fresh-context collaborator extension is introduced into the core estimand.
16. The final report distinguishes computation-level commonality from neural-coordinate commonality.

---

## 22. Stop conditions

Stop and request owner review if:

- production and reference model implementations disagree;
- synthetic model recovery shows key theories are non-identifiable under the current design;
- the external benchmark cannot be reproduced after faithful implementation;
- prospective Qwen differs radically from historical Qwen in a way suggesting a pipeline bug;
- target-task data leak into controller fitting or selection;
- the transfer metric changes across tasks/models;
- transfer predictors are chosen after inspecting the complete matrix without being labeled exploratory;
- final evidence runs have untracked scientific code changes;
- full-matrix compute exceeds approved budget.

Document divergence rather than patching results post hoc.

---

## 23. Research-program framing

The intended conceptual shift is:

From:

> “Does an LLM contain a universal persistence representation?”

To:

> **“What computational equivalence classes organize persistence-like behavior, how are those computations neurally implemented across tasks and models, and what determines when their causal mechanisms transfer?”**

The transfer matrix is therefore not merely a robustness check.

It is a new behavioral-mechanistic object:

\[
\boxed{
T_{ij}
=
\text{causal generalization from task } i \text{ to task } j
}
\]

and the main discovery problem becomes:

\[
\boxed{
\text{What explains } T_{ij}?
}
\]
