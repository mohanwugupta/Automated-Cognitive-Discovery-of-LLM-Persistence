# PRD: Task-Specific Controller Validation and 7×7 Transfer Matrices

**Status:** Proposed
**Audience:** Coding/research agent
**Primary objective:** Build and validate task-specific causal controllers, then measure and explain task-to-task transfer across Qwen, Gemma, and Llama while preserving behavioral-theory uncertainty and response-mapping robustness.

## 1. Scientific motivation

The prospective Qwen replication changes the transfer question.

Under the modern replication harness:

- behavioral history-sensitive structure replicates;
- behavioral theory remains unresolved;
- low-dimensional causal recovery exists within familiar task distributions;
- whole-task transfer is heterogeneous rather than universally positive;
- response-mapping robustness is not uniformly satisfied.

Therefore the next question is:

> **Which task-specific persistence mechanisms transfer to which other tasks, under which cognitive theories, and what properties of source/target tasks predict transfer?**

The central transfer object is:

`T[a,i,j,m,r]`

where:
- `a` = model;
- `i` = source task;
- `j` = target task;
- `m` = behaviorally surviving computational theory;
- `r` = response mapping.

The core endpoint remains:

`cognitive_counterfactual_recovery`

using:

`global_cfr_v1`

Do not introduce natural-effect recovery.

---

## 2. Models, tasks, and theories

### Models

Run under the existing prospective harness for:

- Qwen/Qwen3.5-4B
- Gemma 4 12B
- Llama 3.1 8B

Use the exact pinned revisions already recorded in the corresponding replication artifacts.

### Tasks

Use the canonical seven-task battery:

1. bandit
2. foraging
3. solvability
4. information sampling
5. waiting
6. effort
7. debugging

### Theories

For each model, use its frozen `behavioral_survivor_set`.

Do not force the same survivor set across models.

Do not alter survivor membership based on neural results.

---

## 3. Core design principle

Before interpreting off-diagonal transfer, establish that the source task supports a valid task-specific causal controller.

Implement in two stages:

```text
Stage A — diagonal validation
Stage B — off-diagonal transfer
```

A source task is eligible for off-diagonal transfer only if its diagonal controller passes the preregistered source-validity gate.

---

## 4. Stage A — Task-specific diagonal controllers

For every model `a`, task `i`, and surviving theory `m`:

1. use only source-task `i` neural-training data;
2. select layer/rank using only source-task `i` neural-selection data;
3. freeze the selected controller;
4. evaluate on held-out examples from the same task;
5. evaluate both response mappings separately.

The diagonal object is:

`T[a,i,i,m,r]`

### Required outputs

For each model:

```text
transfer/<model>/diagonal/
    controller_manifest.csv
    diagonal_results.csv
    source_validity.csv
    response_mapping_results.csv
```

Each row should include at minimum:

```text
model
theory
task
response_mapping
layer
rank
n_train
n_selection
n_test
global_cfr
bootstrap_low
bootstrap_high
correlation
slope
random_mean
random_max
random_p
mapping_gap
controller_hash
split_hash
```

---

## 5. Source-validity gate

Freeze the source-validity rule before running the full transfer matrix.

Recommended gate:

```text
source_controller_valid =
    heldout_global_cfr > 0
    AND bootstrap_lower_bound > 0
    AND intervention_target_correlation > 0
    AND beats_matched_random_subspace_null
    AND response_mapping_robustness_passes
```

The exact mapping-robustness threshold must be frozen in config before final evidence runs.

If a theory/task pair fails source validity:

```text
source_status = invalid_source_controller
```

Do not encode its off-diagonal transfer as zero.

Mark all outgoing cells for that source/theory as:

```text
unavailable
```

unless the analysis is explicitly labeled exploratory.

---

## 6. Response mapping as a first-class transfer dimension

Do not collapse mappings prematurely.

Store transfer separately for each mapping:

`T[a,i,j,m,r]`

Then derive:

- mapping-specific CFR;
- mapping-averaged CFR;
- mapping gap;
- mapping sign consistency;
- mapping-robustness status.

Do not claim task-general transfer if it depends strongly on one arbitrary response mapping.

The mapping-robustness rule must be preregistered.

---

## 7. Freeze transfer predictors before off-diagonal execution

Before inspecting the complete off-diagonal matrices, define and commit all planned transfer predictors.

Create:

```text
configs/transfer/transfer_predictors_v1.yaml
```

and:

```text
docs/transfer/TRANSFER_PREDICTOR_SPEC.md
```

Freeze predictor definitions, coding, and direction before the full matrix is viewed.

Use four predictor families.

### 7.1 Behavioral/computational similarity

Examples:

- distance/cosine between fitted coefficient vectors;
- same surviving theory;
- difference in outcome-history sensitivity;
- difference in action-history sensitivity;
- difference in continuation-value sensitivity;
- difference in disengagement-value sensitivity;
- difference in progress sensitivity;
- difference in success-evidence sensitivity;
- difference in continuation-cost sensitivity.

### 7.2 Task-structure similarity

Examples:

- terminal vs reversible disengagement;
- continuation produces new information;
- explicit continuation cost;
- stochastic vs deterministic outcomes;
- temporal horizon;
- controllability;
- environmental stability;
- prior-investment relevance;
- progress observability;
- success/failure feedback;
- stay/leave vs retry/give-up vs wait/stop structure.

All categorical encodings must be frozen before transfer results are inspected.

### 7.3 Counterfactual geometry

Examples:

- similarity of `delta_D_CF` distributions;
- effect-size variance;
- target sign balance;
- source/base feature distance;
- target magnitude similarity.

### 7.4 Neural similarity

Examples:

- selected relative layer;
- rank;
- principal-angle overlap;
- subspace similarity;
- representational similarity before DAS fitting.

Neural similarity variables that require task-specific controllers may be computed only after diagonal controllers are frozen, but their definitions must be frozen before off-diagonal transfer is analyzed.

---

## 8. Stage B — 7×7 off-diagonal transfer matrix

For every valid source controller `U[a,i,m]`, evaluate on target task `j` without target-task fitting or selection.

Primary object:

`T[a,i,j,m,r] = CFR_G(U[a,i,m] -> j)`

### Leakage rule

Target-task data must never contribute to:

- source controller fitting;
- source layer selection;
- source rank selection;
- source hyperparameter selection;
- source validity.

Only frozen target test examples may be used for zero-shot transfer evaluation.

### Required outputs

```text
transfer/<model>/matrix/
    transfer_long.csv
    transfer_matrix_<theory>_<mapping>.csv
    transfer_summary.csv
```

Required row fields:

```text
model
theory
source_task
target_task
response_mapping
source_valid
layer
rank
global_cfr
bootstrap_low
bootstrap_high
correlation
slope
mapping_gap
random_p
transfer_status
```

Suggested transfer status vocabulary:

```text
positive_transfer
inconclusive
negative_transfer
unavailable
```

Do not collapse uncertainty into binary pass/fail unless required by a preregistered analysis.

---

## 9. Pilot before full compute

Before launching every model/theory/task combination:

1. run one full-source Qwen pilot;
2. include all seven target tasks;
3. verify leakage checks;
4. verify mapping-specific output;
5. verify artifact naming and no overwrite;
6. estimate GPU-hours and storage for the full matrix.

Produce:

```text
transfer/PILOT_RESOURCE_REPORT.md
```

The report must estimate:

- GPU-hours per source-task controller;
- evaluation cost per target task;
- full Qwen cost;
- projected Gemma cost;
- projected Llama cost;
- storage footprint.

If projected cost exceeds the approved budget, stop for owner review before launching the full matrix.

---

## 10. Run order

After the pilot:

1. complete Qwen 7×7 transfer for all behaviorally surviving theories with valid sources;
2. inspect only for software/provenance failures, not scientific tuning;
3. run Gemma;
4. run Llama;
5. freeze all three transfer matrices;
6. only then fit explanatory transfer models.

Do not change predictor definitions after seeing Qwen’s completed matrix unless new variables are explicitly labeled exploratory and versioned separately.

---

## 11. Multi-task transfer analyses

After the single-task 7×7 matrices are frozen, add:

### 11.1 Leave-one-task-out

For target task `j`:

`U[-j,m] -> j`

Train on all other valid task families and evaluate on `j`.

Question:

> Does task diversity rescue transfer?

### 11.2 Training-diversity curves

For each target task, vary the number of source task families:

`1, 2, 3, ..., 6`

Use a preregistered or exhaustive subset policy.

Estimate whether transfer improves with source-task diversity.

This distinguishes:

```text
insufficient source diversity
```

from:

```text
genuinely task-specific neural computation
```

### 11.3 Pair-trained controllers

Optional if compute allows.

Do not let pair selection become post hoc hypothesis fishing.

---

## 12. Controller geometry

For each valid pair of task-specific controllers `U_i` and `U_j`, quantify:

- principal angles;
- subspace overlap;
- canonical correlations;
- relative layer difference;
- rank difference.

Primary question:

> When transfer fails, do both tasks possess valid controllers in different neural coordinates?

If:

```text
T_ii > 0
T_jj > 0
T_ij <= 0
```

this is evidence consistent with task-specific neural implementations.

Any alignment experiment should be separately frozen and must not use target-test outcomes for fitting.

---

## 13. Transfer-explanation model

Do not fit this until transfer matrices are frozen.

For directed task pair `i,j`, model transfer as a function of frozen task-pair features plus source, target, and model effects.

Because the sample is small, use regularization or hierarchical shrinkage.

### Validation

Prefer leave-one-target-task-out validation because the scientific question is prediction of transfer to unseen task families.

### Reporting

Report predictor-family performance first.

Do not overinterpret individual coefficients unless stable under held-out validation.

---

## 14. Cross-model analysis

After all matrices are frozen:

1. correlate Qwen/Gemma/Llama transfer topology;
2. test whether the same source-target pairs transfer across models;
3. distinguish magnitude similarity from topology similarity;
4. compare theory-specific transfer topology;
5. test whether task-pair predictors generalize across models.

Use permutation/bootstrap uncertainty that respects repeated task-pair structure.

---

## 15. TDD requirements

Use strict:

```text
RED → GREEN → REFACTOR
```

Add tests for:

### Diagonal stage

- source task train/selection/test separation;
- target task cannot leak into source selection;
- each surviving theory gets a separate controller;
- both mappings are evaluated;
- invalid source is marked unavailable, not zero.

### Matrix stage

- diagonal cells reproduce frozen diagonal evaluations;
- off-diagonal evaluation never refits controller;
- source/target identity is correct;
- mapping dimension preserved;
- metric remains `global_cfr_v1`;
- endpoint remains `cognitive_counterfactual_recovery`.

### Predictor stage

- predictor definitions load only from frozen spec;
- no transfer outcomes are used to construct behavioral/task predictors;
- categorical coding is deterministic.

### Provenance

- `git_dirty == false` for final runs;
- exact model revision recorded;
- split and controller hashes recorded;
- source-validity rule hash recorded;
- predictor-spec hash recorded.

---

## 16. Provenance and artifact policy

Every final transfer run must record:

```text
git_commit
git_dirty
model_id
model_revision
tokenizer_revision
adapter_version
behavioral_survivor_set
survivor_rule_hash
source_validity_rule_hash
transfer_predictor_spec_hash
counterfactual_pair_hash
neural_split_hash
controller_hash
layer
rank
response_mapping
endpoint_id
metric_id
seeds
```

Final evidence requires:

```text
git_dirty = false
```

Never save full activation banks unless necessary.

Persist only low-rank bases, pair-level intervention results, compact projections, manifests, and summary tables.

---

## 17. Interpretation rules

### Broad off-diagonal transfer
Evidence for a shared task-general neural implementation.

### Strong diagonals, weak off-diagonals
Similar computational roles may be implemented in task-specific neural coordinates.

### Clustered transfer
Persistence may form computational equivalence classes rather than one universal mechanism.

### Weak diagonals
The corresponding theory/task does not support a stable causal controller; off-diagonal failure is not interpretable as failed transfer.

### Mapping-dependent transfer
Evidence is not robust to arbitrary response-interface geometry and should not be described as a stable cognitive mechanism.

---

## 18. Acceptance criteria

Complete when:

1. diagonal task-specific controllers are fit for all eligible model × task × surviving-theory combinations;
2. source-validity is frozen and applied prospectively;
3. response mapping is preserved as an explicit dimension;
4. task-pair predictor definitions are frozen before full off-diagonal analysis;
5. a clean Qwen pilot verifies leakage, compute, and artifact structure;
6. 7×7 matrices are completed for Qwen, Gemma, and Llama where valid source controllers exist;
7. unavailable source cells are distinguished from zero/negative transfer;
8. leave-one-task-out transfer is completed;
9. training-diversity effects are estimated;
10. controller geometry is quantified;
11. transfer-prediction models use held-out validation;
12. cross-model topology is quantified;
13. no natural-effect or fresh-context endpoint is introduced;
14. final evidence runs are clean-worktree and fully hashed.

---

## 19. Immediate instruction to the coding agent

Implement this PRD beginning with the diagonal stage.

Do **not** launch the full 7×7 × theory × model matrix immediately.

First:

1. freeze source-validity and mapping-robustness criteria;
2. implement diagonal task-specific controllers;
3. freeze the transfer-predictor specification;
4. run one Qwen source-task pilot across all seven targets;
5. report projected compute/storage and any leakage/provenance failures.

If the pilot is clean and within budget, continue automatically to the full Qwen matrix, then Gemma, then Llama, without requesting further owner approval unless a substantive stop condition occurs.

Stop only for:

- data leakage;
- endpoint drift;
- invalid provenance;
- implementation inconsistency;
- response-mapping logic failure;
- projected compute above budget;
- inability to obtain a valid task-specific source controller where the code indicates one should exist.

Do not stop because transfer is negative, theory identity is unresolved, or different models/tasks choose different layers/ranks.
