# PRD: Cross-Task Computational–Mechanistic Discovery of LLM Persistence

**Status:** Proposed — supersedes the current task-specific-controller / 7×7 DAS plan  
**Audience:** Coding/research agent  
**Models:** Qwen3.5-4B, Gemma 4 12B, Llama 3.1 8B using the exact pinned revisions already recorded in the repository  
**Primary objective:** Recreate the full automated discovery program across three LLMs: systematically design the persistence task space, map which variables causally affect persistence, discover which computational models explain those effects across tasks, identify what internal information is shared across tasks, and test whether any shared neural subspaces causally alter persistence.

---

## 1. Scientific question

The project is **not**:

> Find one persistence controller and test whether it transfers.

The project is:

> **What behavioral variables, computational structures, internal representations, and causal neural mechanisms are shared across persistence-like behavior in different tasks and different LLMs?**

The existence of a single task-general persistence representation is one possible outcome, not an assumption.

The discovery pipeline should distinguish:

1. **Behavioral commonality** — Which experimentally manipulated variables causally influence persistence across tasks?
2. **Computational commonality** — Which computational model families explain persistence across tasks, and what parameters/forms are shared versus task-specific?
3. **Representational commonality** — Which behaviorally/computationally relevant variables are represented in similar neural subspaces across tasks?
4. **Causal commonality** — Which shared neural representations, if any, causally alter persistence across tasks?

The final product should map the structure of persistence rather than force a universal mechanism.

---

## 2. Core output objects

The project should produce four primary objects for each model `a`.

### 2.1 Behavioral causal-effect map

`B[a,t,v]`

where:
- `t` = task family;
- `v` = experimentally manipulated persistence-relevant variable.

Interpretation:

> How much does an experimental intervention on variable `v` change persistence in task `t`?

### 2.2 Computational-model map

`M[a,t,m]`

where `m` indexes candidate computational models.

Interpretation:

> Which computational descriptions explain persistence in each task, and how much parameter/form sharing is supported across tasks?

### 2.3 Representational-sharing tensor

`R[a,i,j,v,l]`

where:
- `i` = source task;
- `j` = target task;
- `v` = behavioral/computational variable or discovered latent quantity;
- `l` = model layer / relative depth.

Interpretation:

> Does information about variable `v` learned in task `i` generalize to task `j`?

### 2.4 Causal-transfer tensor

`C[a,i,j,v]`

Interpretation:

> Does intervening on a `v`-related neural subspace discovered from source task(s) causally produce the predicted persistence change in target task `j`?

A universal persistence controller would be one special high-sharing pattern in `C`. It is not presupposed.

---

## 3. Models

Run the same scientific program independently on:

1. **Qwen/Qwen3.5-4B**
2. **Gemma 4 12B**
3. **Llama 3.1 8B**

Use the exact immutable revisions already recorded by the prospective replication harness.

The same semantic experimental design and condition IDs should be used across models wherever possible.

Model-specific differences are allowed only for:
- tokenizer/interface adaptation;
- chat-template rendering;
- architecture adapter;
- valid response-token interface.

Do not change the scientific ontology or design separately for a model after observing its results.

---

## 4. Task space

Use the canonical seven-task battery:

1. bandit
2. foraging
3. solvability
4. information sampling
5. waiting
6. effort
7. debugging

Reuse the existing SweetPea / SweetBean task-generation infrastructure where valid.

The experiment must preserve the original reason for using formal design tooling:

> **The task space should be systematically generated and counterbalanced rather than manually sampled.**

---

## 5. Persistence ontology

Use the existing project ontology as the starting point.

Candidate factors include, where semantically valid:
- continuation value;
- disengagement value;
- continuation cost;
- progress;
- success evidence;
- uncertainty;
- prior investment;
- controllability;
- environmental stability;
- goal/policy continuity;
- action history;
- outcome history;
- contextual history variables.

### 5.1 Compatibility matrix

Before generating conditions, create and freeze:

```text
configs/discovery/task_variable_compatibility_v1.yaml
```

For every task × variable cell record:

```text
task
variable
manipulable: true | false
levels
semantic definition
nuisance variables to hold fixed
history requirements
notes
```

A variable does not need to exist in every task.

Do **not** fabricate manipulations solely to make the matrix rectangular.

---

## 6. Stage 0 — Measurement-interface validation

Validate the continue/disengage response interface separately for each model before scientific data collection.

Candidate interfaces may include:
- A/B;
- Yes/No;
- Continue/Stop;
- other predeclared interfaces if necessary.

Required checks:
- semantic reversal;
- response-mapping reversal;
- task-level validity;
- acceptable order effects;
- valid tokenizer behavior.

A failed interface is a measurement failure, not a failed persistence result.

Freeze the selected interface and both counterbalanced mappings before Stage 1.

---

## 7. Stage 1 — Automated behavioral discovery

### 7.1 Goal

Map which variables causally affect persistence across the seven-task space.

The behavioral dependent variable is the semantic persistence logit:

`D = log p(continue) - log p(disengage)`

Response-token mappings must be semantically aligned before computing `D`.

### 7.2 Experimental design

Use SweetPea to generate a broad, counterbalanced factorial design.

The initial design should emphasize:
- marginal effects;
- theoretically important interactions;
- decorrelation of naturally correlated factors;
- history manipulations;
- response mapping;
- task-family coverage.

Where variables are correlated in natural settings, deliberately generate discriminating combinations.

Examples:
- high success history + low current continuation value;
- low success history + high current continuation value;
- high prior investment + low controllability;
- positive progress + high continuation cost.

The design goal is **identifiability of causal determinants**, not ecological frequency.

### 7.3 Shared design across models

Generate one canonical semantic condition manifest.

All three models should evaluate the same semantic conditions wherever their validated interfaces permit.

Each model gets its own rendered prompt, but prompt provenance must point to the same semantic condition ID.

This allows direct cross-model comparison of the behavioral effect maps.

### 7.4 Initial broad exploration

Use a broad space-filling design comparable in spirit to the original discovery study.

Freeze:

```text
behavior_train
behavior_selection
behavior_test
```

before model fitting.

Group:
- both mappings of one semantic condition;
- repeated renderings;
- semantically linked contrast pairs.

No leakage across splits.

### 7.5 Behavioral causal-effect estimation

For every model × task × manipulable variable estimate:
- average causal contrast on `D`;
- uncertainty;
- standardized effect;
- direction consistency;
- response-mapping interaction;
- important preregistered interactions.

Create:

```text
behavior/<model>/variable_effects.csv
```

This is the empirical `B[t,v]` matrix.

Do not require every variable to matter in every task.

---

## 8. Stage 2 — Computational model discovery

### 8.1 Candidate model bank

At minimum include the existing core families:
- immediate state;
- dynamic re-evaluation;
- choice perseveration;
- outcome history;
- dual history;
- latent context;
- latent motivation;
- option termination;
- task-set reinstatement;
- meta-control.

Retain flexible baselines where computationally feasible:
- shared interactions;
- MLP;
- GRU or equivalent sequential baseline.

### 8.2 Cross-task sharing structures

Do not ask only:

> Which model wins?

For each computational family compare parameter-sharing structures such as:

- **M1 — fully shared:** one parameterization across tasks.
- **M2 — task-specific:** separate parameters per task.
- **M3 — hierarchical/random effects:** shared population structure with task-level deviations.
- **M4 — ontology-conditioned:** task parameters predicted by task properties / ontology.

Use the existing model-comparison infrastructure where valid.

### 8.3 Theory-survivor sets

Behavioral theory identity may remain unresolved.

Freeze a behavior-only survivor set before any neural analysis.

Do not force a unique winner when predictive differences are within the frozen practical-equivalence criterion.

For each model save:

```text
behavior/<model>/computational_models/
    model_comparison.csv
    parameter_sharing_comparison.csv
    survivor_set.json
    per_task_metrics.csv
    frozen_models/
```

---

## 9. Stage 2B — Active theory discrimination

### 9.1 Goal

Use automated experimental design to find conditions that distinguish surviving computational theories.

Preserve the original workflow:

```text
broad exploration
→ model formation
→ active discrimination
→ untouched validation
```

### 9.2 Candidate generation

Generate a large candidate condition pool from SweetPea/SweetBean.

Score candidates using a frozen acquisition function combining:
- coverage;
- predictive uncertainty;
- disagreement among surviving theories.

Do not use neural data in behavioral acquisition.

### 9.3 Active rounds

Iteratively:
1. select conditions;
2. query all three models;
3. refit/update computational models;
4. select next batch.

Keep an untouched final validation set that is never used for acquisition.

### 9.4 Outcome

The result may be:
- one theory resolved;
- several theories unresolved;
- theory resolution differs across models.

All are valid outcomes.

---

## 10. Stage 3 — Cross-task representational discovery

This stage asks:

> **What information is internally represented similarly across tasks?**

It does **not** assume that a shared representation causally controls persistence.

### 10.1 Representational targets

Include variables/quantities that meet at least one criterion:
- clear behavioral causal effect in one or more tasks;
- retained by a surviving computational model;
- theoretically central to the persistence ontology.

Possible targets:
- outcome history;
- action history;
- progress;
- success evidence;
- continuation value;
- disengagement value;
- cost;
- controllability;
- context state;
- integrated model prediction / persistence evidence.

Do not automatically treat the final continue/disengage logit as a cognitive variable.

### 10.2 Matched representational contrasts

For each valid task × variable cell generate matched contrasts where the target variable changes and nuisance factors are controlled.

For each semantic contrast collect both response mappings.

At the final pre-generation decision position, extract hidden states across a preregistered layer grid or all feasible layers.

Do not save full activation banks to Git.

Use scratch streaming and persist only compact sufficient artifacts.

### 10.3 Within-task representation

For each:

```text
model × task × variable × layer
```

estimate whether the variable is represented on held-out semantic conditions.

Possible methods:
- ridge decoding;
- low-rank supervised subspace;
- contrast-direction PCA/SVD;
- representation-level regression.

Use train/selection/test splits grouped by semantic contrast.

Primary outputs:
- held-out decoding R² / correlation;
- layer profile;
- mapping invariance;
- random/null comparison.

This is representation evidence only.

### 10.4 Cross-task representational transfer

Train representation model/subspace on source task `i`.

Without refitting, evaluate target variable `v` in target task `j`.

Store:

`R[i,j,v,l]`

Required output:

```text
representation/<model>/transfer_long.csv
```

Fields should include:

```text
source_task
target_task
variable
layer
metric
value
uncertainty
mapping_robustness
source_validity
target_validity
```

### 10.5 Shared/private representational decomposition

In addition to pairwise transfer, compare explicit sharing models:

- **R1 — one shared subspace**
- **R2 — shared + task-private**
- **R3 — aligned task-specific**
- **R4 — fully task-specific**

Compare on held-out tasks/conditions.

Do not assume R1.

### 10.6 Leave-one-task-out representational test

For variable `v`, learn a shared representation from all compatible tasks except target task `j`.

Evaluate on task `j`.

This directly estimates how much representational commonality exists without having seen the target task.

---

## 11. Stage 4 — Bottom-up persistence-effect geometry

This stage complements variable-specific representations.

It asks:

> Across many manipulations that increase or decrease persistence, is there a common neural effect geometry even when the manipulated variable differs?

For matched semantic pairs:

`Δh = h(high-persistence) - h(low-persistence)`

Construct task-specific and cross-task low-rank effect spaces.

Critically:
- train on some manipulation families;
- test on held-out manipulation families;
- preserve response mappings as paired observations.

This prevents a success-history direction, for example, from being mislabeled as a generic persistence representation.

Output:

```text
representation/<model>/persistence_effect_geometry/
```

Compare:
- task-specific effect spaces;
- shared effect spaces;
- held-out manipulation transfer;
- held-out task transfer;
- principal angles / subspace overlap.

This is still representational, not causal.

---

## 12. Stage 5 — Causal neural discovery

Only after Stages 1–4 have identified candidate shared information should the project intervene.

The causal stage should test specific hypotheses generated prospectively by the representational/computational analyses.

### 12.1 Variable-specific causal tests

For variable `v`, use matched source/base pairs corresponding to a controlled intervention on `v`.

The computational/behavioral model supplies the expected effect `ΔD_CF[v]`.

Search for or use a candidate `v`-related subspace and test whether neural intervention reproduces the expected effect.

Within-task evidence:

`C[i,i,v]`

Cross-task causal transfer:

`C[i,j,v]`

Only evaluate task pairs where `v` is semantically defined in both tasks.

### 12.2 Shared-subspace causal tests

If Stage 3 or 4 discovers a shared subspace across multiple tasks, freeze it before causal evaluation.

Test it on:
- held-out semantic conditions;
- held-out manipulation families;
- held-out task families.

A shared causal persistence mechanism is supported only if a subspace learned without target-task outcomes causally shifts target-task persistence in the predicted direction.

### 12.3 Causal search methods

Allowed methods include:
- DAS / interchange interventions;
- subspace patching;
- activation replacement along frozen low-rank bases.

The method must be selected before viewing causal outcomes.

Do not use simple probe directions as causal evidence.

### 12.4 Controls

Required where informative:
- matched random subspaces;
- output/readout direction;
- PCA/variance subspace;
- predictive probe direction;
- shuffled source/base where the target truly changes;
- response-mapping invariance.

If a control does not alter the target, mark:

`uninformative_control`

rather than pass/fail.

---

## 13. Stage 6 — Relate the levels

The central scientific analysis is how the levels relate.

Test questions such as:

### 13.1 Behavioral similarity → representational similarity

Do task pairs with similar behavioral variable effects have greater representational transfer?

### 13.2 Computational similarity → representational sharing

Do task pairs explained by similar computational models/parameters encode relevant variables similarly?

### 13.3 Representational sharing → causal transfer

Does shared information predict causal portability?

### 13.4 Shared behavior without shared mechanism

Identify cases where tasks show similar behavioral effects but weak representational or causal sharing.

These are scientifically important dissociations.

---

## 14. Cross-model synthesis

Once all three models are complete, compare:
- behavioral causal-effect topology;
- computational-model topology;
- representational-sharing topology;
- causal-transfer topology.

Ask:
1. Which variable effects replicate across architectures?
2. Which computational forms replicate?
3. Which representations are shared across tasks within each model?
4. Which shared representations replicate across models?
5. Which causal transfer patterns replicate?
6. Are differences primarily task-driven or architecture-driven?

Do not require identical neural coordinates across architectures.

Cross-model replication should focus on structural patterns.

---

## 15. Evidence hierarchy

Reports must distinguish:

- **Level 0 — behavioral association**
- **Level 1 — experimental behavioral causality**
- **Level 2 — computational explanation**
- **Level 3 — neural representation**
- **Level 4 — causal neural effect**
- **Level 5 — cross-task causal sharing**
- **Level 6 — circuit / necessity** — not required by this PRD

Never promote Level 3 evidence to Level 4/5 language.

---

## 16. Analysis outputs

For each model produce:

```text
cross_task_discovery/<model>/
    provenance.json

    interface/
        validation.json

    behavior/
        condition_manifest.parquet
        variable_effects.csv
        interactions.csv
        task_variable_matrix.csv

    computational/
        model_comparison.csv
        parameter_sharing_comparison.csv
        survivor_set.json
        active_discrimination.csv
        final_validation.csv

    representation/
        within_task.csv
        transfer_long.csv
        shared_private_models.csv
        loto_transfer.csv
        persistence_effect_geometry/

    causal/
        within_task.csv
        transfer_long.csv
        shared_subspace_tests.csv
        controls.csv

    synthesis/
        level_relationships.csv
        task_similarity.csv
        discovered_task_clusters.csv

    DISCOVERY_REPORT.md
```

Also produce:

```text
cross_task_discovery/cross_model/
    behavioral_comparison.csv
    computational_comparison.csv
    representational_comparison.csv
    causal_comparison.csv
    CROSS_MODEL_REPORT.md
```

---

## 17. Figures

Generate from frozen tables only.

Minimum useful figures:

1. task × variable behavioral causal-effect heatmap;
2. task × computational-model performance/sharing map;
3. cross-task representation-transfer maps for important variables;
4. shared vs private representational variance;
5. variable-specific causal-transfer maps;
6. behavioral/computational similarity vs representational transfer;
7. representational transfer vs causal transfer;
8. cross-model comparison of discovered structure.

Do not manually enter publication values.

---

## 18. Active-discovery principle

Preserve the automated-science logic:

```text
broad exploration
→ identify uncertainty / competing models
→ generate discriminating experiments
→ collect new behavior
→ update models
→ freeze hypotheses
→ targeted representation tests
→ targeted causal tests
```

Expensive neural interventions should be guided by discoveries from earlier stages.

Do not brute-force every task × variable × layer × theory combination if earlier evidence makes the test scientifically uninformative.

---

## 19. Reuse of prior work

Reuse:
- validated task generators;
- SweetPea/SweetBean design infrastructure;
- model adapters;
- interface-validation code;
- computational-model implementations after validation;
- metric implementations;
- provenance infrastructure;
- DAS/intervention infrastructure;
- artifact guards.

Do **not** use previous fitted behavioral models, neural bases, or mechanistic outcomes as scientific inputs to the new cross-model discovery run.

Prior Qwen/Gemma/Llama replications may be used only as:
- regression tests;
- historical comparison;
- compute planning.

---

## 20. Development and run strategy

### Phase A — Implement one-model end-to-end dry pipeline

Use Qwen for software integration.

No scientific interpretation until all stage boundaries and split rules are verified.

### Phase B — Run behavioral/computational discovery for all three models

These stages are cheaper and should complete before expensive neural work.

### Phase C — Freeze discovery outputs

Freeze:
- relevant variables;
- computational survivor sets;
- active-discrimination outcomes;
- representation targets;
- causal hypotheses.

This is a scientific freeze, not outcome tuning.

### Phase D — Run representational analyses

Representation is cheaper than broad causal intervention and should narrow causal targets.

### Phase E — Run causal tests

Only test preregistered hypotheses supported by earlier levels or specifically chosen as negative controls.

### Phase F — Cross-model synthesis

No model-specific post-hoc redesign.

---

## 21. TDD requirements

Use strict:

`RED → GREEN → REFACTOR`

Required test families:

### Design
- SweetPea factor balance;
- task × variable compatibility;
- impossible combinations rejected;
- semantic contrast pairing;
- mapping counterbalancing.

### Behavior
- semantic persistence logit correctness;
- response mapping reversal;
- split grouping;
- causal contrast recovery on synthetic data.

### Computational
- previously validated hand fixtures;
- synthetic parameter/model recovery;
- survivor-set logic;
- active-acquisition determinism;
- untouched validation protection.

### Representation
- no target leakage;
- source-trained decoder remains frozen on target;
- response mapping retained/grouped;
- random-label/null tests;
- layer identity.

### Causality
- counterfactual endpoint identity;
- intervention basis frozen before target evaluation;
- random-control correctness;
- no target-task refitting for cross-task causal tests.

### Provenance
- exact model revision;
- clean-worktree final runs;
- split hashes;
- design hashes;
- frozen hypothesis/config hashes.

---

## 22. Provenance

Every final run must record:
- Git commit;
- `git_dirty = false`;
- lockfile hash;
- model and tokenizer immutable revision;
- task-design hash;
- task-variable compatibility hash;
- behavioral split hash;
- computational-model spec hashes;
- survivor-set hash;
- active-acquisition config hash;
- representation-target manifest hash;
- neural split hash;
- subspace/controller hashes;
- endpoint/metric IDs;
- seeds.

No final result may silently inherit an unknown historical identity.

---

## 23. Compute and storage

Do not create permanent full activation banks.

Preferred pattern:
1. load model once;
2. stream prompts;
3. extract final decision-position activations;
4. update compact sufficient statistics / temporary scratch arrays;
5. fit compact decoders/subspaces;
6. save only compact scientific artifacts;
7. remove scratch activations.

Before causal runs, produce a compute estimate based on measured representational-stage throughput.

---

## 24. Stop conditions

Stop for owner review only if:
- task-variable ontology requires a scientific choice not encoded in the current project;
- measurement interface is invalid;
- computational implementation validation fails;
- semantic conditions cannot manipulate a target variable independently enough for causal interpretation;
- leakage is detected;
- endpoint definitions drift;
- provenance is incomplete;
- projected compute exceeds approved budget.

Do **not** stop because:
- behavioral theories remain unresolved;
- some variables have no effect;
- representations fail to transfer;
- no shared subspace is found;
- causal transfer is absent;
- models disagree.

Those are scientific outcomes.

---

## 25. Interpretation space

The pipeline must allow all of these outcomes.

### Outcome A — Strong universal structure
Behavioral variables, computational models, representations, and causal subspaces largely generalize across tasks.

### Outcome B — Shared computation, different coordinates
Behavioral/computational structure generalizes, but representations or causal subspaces do not.

### Outcome C — Mechanistic families
Subset clusters share behavioral variables, computational models, representations, and causal effects.

### Outcome D — Behavioral commonality only
Similar variables affect behavior, but internal representations/mechanisms differ.

### Outcome E — Broad fracture
Even behavioral/computational structure differs substantially by task.

None is considered a failed experiment.

---

## 26. Final research framing

The project should answer:

> **Across diverse persistence-like tasks, what is shared at the behavioral, computational, representational, and causal levels—and where does that commonality break down?**

The intended methodology is:

```text
formal experimental design
→ broad behavioral exploration
→ computational discovery
→ active discrimination
→ representational mapping
→ targeted causal testing
```

This PRD supersedes the earlier plan to immediately construct a 7×7 matrix of presumed task-specific persistence controllers.

The 7×7 transfer structure may still emerge as an output, but it should be indexed by the actual variable/representation discovered and should not presuppose that one universal controller exists.
