Yes. A clean repo is probably worth it now.

The current repo has done its job: it generated the persistence phenomenon, falsified several simple hypotheses, and exposed the need for a broader discovery framework. But it also contains several generations of task-specific pipelines, probe searches, contrast analyses, model zoos, and repair experiments. Carrying all of that forward risks making the new project harder to reason about and harder to present.

I would **freeze the current repo as provenance**, then selectively port only validated components into a new repo whose architecture mirrors the scientific method we now want to use.

# PRD — Automated Cognitive Discovery of LLM Persistence

**Working repo name:** `llm-cognitive-discovery`
**Working paper concept:** *Discovering the Computational Architecture of Persistence in Language Models*
**Primary model:** Qwen3.5-4B initially
**Primary venue target:** ICLR 2027
**Project type:** Behavioral computational discovery
**Status:** New clean repository
**Core infrastructure:** SweetPea + SweetBean + cognitive-model library + flexible discovery models

---

# 1. Scientific objective

Determine:

> **What computational organization governs whether an LLM continues or disengages from an ongoing course of action?**

And secondarily:

> **Which aspects of that organization resemble established computational accounts of persistence and motivational control in humans and animals?**

The project should not begin by assuming that persistence corresponds to:

* a unitary latent state;
* a shared neural direction;
* a common history kernel;
* a goal-continuity mechanism;
* or any other privileged theory.

Instead, use broad experimental coverage to discover which computational regularities actually generalize.

---

# 2. Scientific strategy

The central methodological principle is:

$$
\boxed{
\text{explore experimental space broadly before exploiting a favored theory}
}
$$

Pipeline:

$$
\text{Experimental ontology}
$$

$$
\downarrow
$$

$$
\text{Balanced design-space sampling}
$$

$$
\downarrow
$$

$$
\text{LLM behavioral experiments}
$$

$$
\downarrow
$$

$$
\text{Behavioral response surface}
$$

$$
\downarrow
$$

$$
\text{Cognitive model competition}
$$

$$
\downarrow
$$

$$
\text{Flexible residual discovery}
$$

$$
\downarrow
$$

$$
\text{Independent validation sample}.
$$

Only after this pipeline identifies a robust computation should mechanistic interpretability begin.

---

# 3. Starting empirical facts

The previous project supplies motivation but **not assumptions that the new pipeline must reproduce**.

Existing observations include:

* persistence is highly predictable behaviorally;
* recent history is often useful;
* finite history can explain substantial behavior;
* no convincing task-general persistence representation was recovered;
* strict cross-task representational transfer failed;
* exact history kernels differ substantially across tasks;
* matched sequential controls suggested that history can matter differently during ongoing goal pursuit;
* task-specific evidence such as cost, progress, alternatives, and reward matters differently across environments.

These become phenomena for the new framework to explain.

Do not hard-code them into the experimental or modeling architecture.

---

# 4. Primary hypotheses

Maintain a hypothesis bank rather than a single favored theory.

## H1 — Dynamic re-evaluation

Persistence is repeatedly recomputed from current prospective evidence:

$$
A_t =
V_{\text{continue},t}
-
V_{\text{disengage},t}
$$

$$
P(\text{continue}_t)=\sigma(A_t).
$$

History matters insofar as it changes beliefs about current/future value.

---

## H2 — Choice perseveration

Previous actions directly bias repeated behavior:

$$
K_t =
\lambda K_{t-1}+a_{t-1}
$$

$$
D_t=f(X_t)+\kappa K_t.
$$

---

## H3 — Outcome-history integration

Recent successes and failures are temporally integrated:

$$
R_t =
\lambda R_{t-1}+r_{t-1}.
$$

Persistence depends on:

$$
D_t=f(X_t,R_t).
$$

---

## H4 — Latent motivational state

A slowly evolving commitment/patience variable controls persistence:

$$
M_t =
\rho M_{t-1}+\beta X_t+\epsilon_t.
$$

---

## H5 — Latent-state / context inference

The agent infers which previous experiences are relevant to the current decision.

History effects therefore depend on inferred contextual/state membership.

---

## H6 — Hierarchical option termination

The current behavior is a temporally extended policy/option with a termination function:

$$
P(\text{terminate option }o\mid s)
=
\beta_o(s).
$$

---

## H7 — Task-set stability / reinstatement

Current context retrieves or maintains an existing control/task configuration.

Repeated context can therefore reinstate earlier policy settings.

---

## H8 — Meta-control / EVC

Continued engagement depends on the expected benefit of control relative to its costs.

---

## H9 — Generic sequential choice

There is nothing computationally specific about persistence. Ordinary history-sensitive sequential decision machinery explains it.

---

# 5. Crucial methodological rule

**The hypothesis bank must not determine the primary sampling distribution.**

Do not preferentially sample:

* conditions where H1 vs H2 disagree;
* conditions that maximize expected information gain;
* conditions predicted to falsify the current winner;
* conditions likely to exhibit known human effects.

Those methods may be tested later as sampling strategies.

The primary discovery dataset should prioritize **coverage of the legal experimental space**.

---

# 6. Experimental ontology

Define the behavioral experiment using computational dimensions rather than bespoke task-specific variables wherever possible.

## Universal dimensions

### Continuation value

$$
V_C
$$

Expected benefit from maintaining the current course.

Levels initially:

```text
low
medium
high
```

---

### Disengagement / alternative value

$$
V_D
$$

Value available from stopping, switching, or taking an outside option.

Levels:

```text
low
medium
high
```

---

### Continuation cost

$$
C
$$

Immediate cost of another step.

Levels:

```text
low
medium
high
```

---

### Progress evidence

$$
P
$$

Evidence that current effort is producing progress.

Levels:

```text
negative
neutral
positive
```

---

### Success evidence

$$
S
$$

Evidence concerning eventual success.

Levels:

```text
low
medium
high
```

---

### Uncertainty

$$
U
$$

How uncertain the agent should be about future success/value.

Levels:

```text
low
high
```

---

### Prior investment

$$
I
$$

Irrecoverable past effort/time.

Levels:

```text
low
high
```

when meaningful.

---

### Controllability

$$
K
$$

Degree to which the agent's actions causally influence outcomes.

Levels:

```text
low
high
```

when meaningful.

---

### Environmental stability

$$
E
$$

How predictive previous outcomes are of future outcomes.

Levels:

```text
stable
changing
```

---

### Goal/policy continuity

$$
G
$$

Whether the current decision belongs to the same ongoing objective/policy.

Levels:

```text
same
new
```

Do not privilege this variable in analysis.

---

# 7. History ontology

History should be represented independently of the current-state dimensions.

## Action history

Examples:

```text
continue, continue, continue
continue, switch, continue
switch, continue, switch
```

Use semantically normalized action histories.

---

## Outcome history

Use balanced patterns such as:

```text
+++ 
++-
+-+
-++
+--
-+-
--+
---
```

for three-step histories where the task supports binary success/failure.

Also support:

```text
neutral
mixed
```

for nonbinary contexts.

---

## History length

Initially support:

```text
0
1
3
5
```

Use longer sequences selectively.

---

# 8. Task families

The design space must be instantiated across qualitatively different forms of persistence.

Minimum ICLR battery:

1. **Reward pursuit / Bandit**
2. **Foraging / patch exploitation**
3. **Problem solving / solvability**
4. **Information gathering**
5. **Waiting**
6. **Effort expenditure / progressive ratio**
7. **Debugging / repair**

Optional if straightforward:

8. Search/research
9. Planning/replanning
10. Tool retry

Target:

$$
5-7
$$

high-quality families for the first full discovery dataset.

Do not delay the paper trying to reach 10.

---

# 9. Universal versus conditional factors

Not every factor should exist in every task.

For example:

| Dimension          | Bandit | Foraging | Solvability | Info sampling | Waiting | Effort | Debugging |
| ------------------ | -----: | -------: | ----------: | ------------: | ------: | -----: | --------: |
| continuation value |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      ✓ |         ✓ |
| outside option     |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      ✓ |         ✓ |
| cost               |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      ✓ |         ✓ |
| progress           |      – |        ✓ |           ✓ |             ✓ |       – |      ✓ |         ✓ |
| uncertainty        |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      – |         ✓ |
| prior investment   |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      ✓ |         ✓ |
| controllability    |      – |        ✓ |           ✓ |             – |       – |      ✓ |         ✓ |
| history            |      ✓ |        ✓ |           ✓ |             ✓ |       ✓ |      ✓ |         ✓ |

Missing constructs remain explicitly missing.

Never encode semantic absence as numeric zero.

---

# 10. SweetPea role

SweetPea is the **design compiler**.

It should define:

* experimental factors;
* levels;
* derived factors;
* constraints;
* legal crossings;
* counterbalancing;
* response-label mappings;
* trial ordering;
* sequential-history restrictions.

The design specification must be declarative.

Do not bury experimental constraints inside task-runner Python code.

---

# 11. Sampling strategy

Primary discovery sample:

$$
\boxed{\text{balanced coverage-oriented random sampling}}
$$

Use SweetPea to generate conditions satisfying the legal design constraints.

For continuous-like dimensions represented at three levels, prefer approximately uniform coverage.

If the Cartesian product is too large, use:

* balanced random sampling;
* partial factorial coverage;
* stratification;
* Latin-hypercube-like coverage where appropriate.

Do not exhaustively cross everything.

---

# 12. Core crossing

At minimum strongly balance:

```text
task_family
response_mapping
continuation_advantage
history_valence
```

where:

$$
\text{continuation advantage}
=
V_C - V_D - C
$$

is a derived experimental factor.

Other dimensions should receive stratified coverage.

---

# 13. Response label counterbalancing

Every semantic binary decision must be rendered under paired mappings.

Example:

```text
X = continue
Y = disengage
```

and:

```text
X = disengage
Y = continue
```

Store only semantic probabilities downstream:

$$
P(\text{continue})
$$

$$
P(\text{disengage}).
$$

Label mapping remains available as a nuisance-control variable.

---

# 14. SweetBean role

SweetBean is the **experiment renderer**.

Input:

```text
abstract condition specification
```

Output:

```text
concrete task interaction
```

SweetBean should handle:

* task instructions;
* sequence state;
* stimulus generation;
* response collection;
* history presentation;
* LLM participant interface.

Scientific variables live upstream.

Task rendering lives downstream.

---

# 15. Rendering invariance

Whenever two conditions differ only in factor \(X\):

$$
X_a \neq X_b
$$

all other semantic variables should remain matched.

Where feasible, use paired latent seeds.

This allows direct causal behavioral contrasts.

---

# 16. Prompt philosophy

Do not explicitly tell the model:

```text
This experiment measures persistence.
```

Do not ask it:

```text
How motivated are you?
```

Do not instruct:

```text
Compute the expected value of continuing.
```

unless explicit deliberation is itself an experimental manipulation.

Use natural task semantics.

---

# 17. Behavioral target

Primary dependent variable:

$$
D_t
=
\log
\frac{P(\text{continue}_t)}
{P(\text{disengage}_t)}.
$$

This continuous policy quantity should be primary when token logits are available.

Secondary:

$$
Y_t =
\text{sampled continue/disengage action}.
$$

For absorbing environments, also construct stopping hazard:

$$
h_t =
P(\text{stop at }t\mid\text{not previously stopped}).
$$

---

# 18. Why policy logit should be primary

Sampled actions introduce stochastic choice noise.

The model's own semantic probabilities provide a cleaner measurement of its decision policy.

Use sampled trajectories when history must evolve endogenously, but retain policy logits at every state.

---

# 19. Dataset sizes

### Pilot

Per task family:

```text
50–100 conditions
```

to validate rendering and behavioral range.

### Discovery dataset

Target:

$$
2{,}000-5{,}000
$$

unique semantic experimental conditions.

With two label mappings:

$$
4{,}000-10{,}000
$$

rendered observations.

Sequential states may increase the total substantially.

This is behavior-only and should be cheap relative to activation collection.

---

# 20. Dataset partitions

Before model comparison create fixed partitions.

## Discovery/train

Approximately:

$$
60\%.
$$

Used for model fitting.

---

## Interpolation test

Approximately:

$$
20\%.
$$

Random held-out conditions from the same broad design space.

---

## Structural/OOD test

Approximately:

$$
20\%.
$$

Contain deliberately held-out:

* factor combinations;
* parameter regions;
* task-family cells.

---

# 21. Held-out task-family evaluation

Separately perform LOTO:

$$
\text{train on }N-1\text{ families}
\rightarrow
\text{evaluate family }N.
$$

No target-family fitting in the strict zero-shot analysis.

This remains the strongest test of computational generality.

---

# 22. Cognitive model library

Implement the hypothesis bank as modular models.

Suggested models:

```text
intercept
immediate_state
dynamic_reevaluation
choice_perseveration
outcome_history
dual_history
latent_motivation
latent_context
option_termination
meta_control
task_set_reinstatement
```

Where a theory cannot be faithfully operationalized with available variables, document this rather than inventing a proxy silently.

---

# 23. Sharing assumptions

Each compatible cognitive model should support:

### Task-specific

$$
\theta_\tau.
$$

### Fully shared

$$
\theta.
$$

### Hierarchical

$$
\theta_\tau
=
\theta_G+\delta_\tau.
$$

This distinguishes:

$$
\text{same algorithm}
$$

from:

$$
\text{same exact parameters}.
$$

---

# 24. Flexible predictive models

The hypothesis bank must be tested against models capable of discovering missing structure.

Required:

### Regularized linear interaction model

Include main effects and prespecified second-order interactions.

---

### GAM

Useful for nonlinear monotonic/nonmonotonic response surfaces.

---

### MLP

Flexible nonlinear static predictor.

---

### GRU

Flexible sequential predictor.

Validate its training pipeline before interpreting recurrence.

---

### Optional tree-based model

Gradient boosting can be included as an additional nonlinear ceiling if convenient.

---

# 25. Flexible-ceiling validation

Before real-data comparisons, flexible models must reproduce synthetic targets generated from simpler models.

For example:

$$
y =
\sigma(
\beta^\top X+\gamma^\top H
).
$$

MLP/GRU should match the generating predictor within a predefined tolerance.

Also perform teacher distillation from fitted cognitive models.

No model is called a ceiling until this sanity check passes.

---

# 26. Primary model metric

For persistence logit:

* held-out \(R^2\);
* MSE;
* correlation;
* sign accuracy for paired effects only.

For sampled decisions/hazards:

* log loss;
* Brier score;
* calibration.

Primary cross-task summaries must use task-macro weighting.

---

# 27. Explainable-variance fraction

Define a cognitive-model performance fraction relative to null and flexible ceiling:

$$
F_m
=
\frac{
L_{\text{null}}-L_m
}{
L_{\text{null}}-L_{\text{flex}}
}.
$$

For \(R^2\)-style metrics use the corresponding explained-variance formulation.

This answers:

> How much of behavior captured by a flexible predictor is explained by the cognitive theory?

---

# 28. Residual discovery

After identifying the strongest cognitive model \(M^\*\):

$$
\epsilon_i
=
D_i-\hat D_i^{M^\*}.
$$

Fit residual structure using:

* regularized interaction model;
* GAM;
* feature-family ablations.

Search for systematic dependence on:

$$
H\times P
$$

$$
H\times G
$$

$$
C\times P
$$

$$
I\times G
$$

etc.

The goal is to identify explanatory structure missing from the established theory bank.

---

# 29. Rule for proposing new constructs

A new construct should not enter the theory because of one visually striking effect.

Require that it:

1. explains systematic residual variance;
2. appears across multiple task families;
3. predicts untouched observations;
4. improves over a simpler existing model.

This is how something like “goal continuity modulation” should earn theoretical status.

---

# 30. Independent validation stage

After the discovery analysis is complete:

1. freeze the candidate computational model;
2. ask SweetPea for a new independently generated sample;
3. exclude previous conditions;
4. run Qwen again;
5. evaluate predictions without refitting where possible.

Target:

```text
500–1,000 new semantic conditions
```

depending on cost.

This gives the paper:

$$
\boxed{\text{discovery}\rightarrow\text{novel prediction}\rightarrow\text{validation}}.
$$

---

# 31. Optional sampling-strategy study

Stretch goal only.

Using the fixed experiment grammar, compare equal-budget selection strategies:

```text
random / coverage
uncertainty
model disagreement
falsification
confirmation
```

For budgets:

```text
100
250
500
1000
```

conditions.

Evaluate every resulting theory on the same large reference test set.

Primary question:

> Which strategy most rapidly discovers a model that predicts the broader persistence response surface?

This directly connects to the automated-science research program but should not block the persistence paper.

---

# 32. Do not collect activations initially

Phase 1 is behavior only.

Do not:

* collect all-layer activation banks;
* train persistence probes;
* search for subspaces;
* steer;
* patch;
* localize heads.

Mechanistic analysis becomes a second-stage project after behavioral discovery identifies a computation worth targeting.

---

# 33. New repository architecture

Suggested:

```text
llm-cognitive-discovery/
│
├── README.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│   ├── discovery_v1.yaml
│   ├── tasks/
│   ├── models/
│   └── sampling/
│
├── src/
│   └── cognitive_discovery/
│       │
│       ├── ontology/
│       │   ├── factors.py
│       │   ├── histories.py
│       │   ├── constraints.py
│       │   └── task_schema.py
│       │
│       ├── design/
│       │   ├── sweetpea_design.py
│       │   ├── sampling.py
│       │   ├── counterbalance.py
│       │   └── manifests.py
│       │
│       ├── experiments/
│       │   ├── sweetbean/
│       │   ├── bandit.py
│       │   ├── foraging.py
│       │   ├── solvability.py
│       │   ├── information_sampling.py
│       │   ├── waiting.py
│       │   ├── effort.py
│       │   └── debugging.py
│       │
│       ├── participants/
│       │   ├── base.py
│       │   ├── qwen.py
│       │   └── token_mapping.py
│       │
│       ├── data/
│       │   ├── schema.py
│       │   ├── validation.py
│       │   └── splits.py
│       │
│       ├── models/
│       │   ├── cognitive/
│       │   ├── flexible/
│       │   └── sharing/
│       │
│       ├── analysis/
│       │   ├── response_surface.py
│       │   ├── model_comparison.py
│       │   ├── loto.py
│       │   ├── residual_discovery.py
│       │   └── validation.py
│       │
│       └── reporting/
│
├── tests/
│   ├── ontology/
│   ├── design/
│   ├── experiments/
│   ├── participants/
│   ├── models/
│   └── analysis/
│
├── scripts/
│   ├── generate_design.py
│   ├── run_experiments.py
│   ├── fit_models.py
│   ├── discover_residuals.py
│   └── validate_predictions.py
│
└── artifacts/
    └── .gitignore
```

---

# 34. Data schema

Every observation should include at minimum:

```text
design_id
condition_id
paired_condition_id

task_family
episode_id
step

factor_<name>
factor_available_<name>

history_actions
history_outcomes

semantic_continue_token
semantic_disengage_token
response_mapping

p_continue
p_disengage
persistence_logit

sampled_action

prompt_hash
environment_seed
model
model_revision

split
sampling_strategy
```

Store raw prompts separately if large.

---

# 35. Provenance

Every experiment must be reproducible from:

```text
config
SweetPea design seed
task renderer version
model revision
environment seed
```

Create a manifest containing:

```text
git commit
config hash
design hash
model identifier
dependency versions
timestamp
```

---

# 36. Migration policy from the old repo

The old repository should be **frozen**, not deleted.

Tag it:

```text
persistence-sprint-final
```

or equivalent.

Do not copy its directory tree wholesale.

Only port:

### Task logic

Validated semantic/environment logic for:

* Bandit;
* Foraging;
* Solvability;
* Information Sampling;
* Partial Reinforcement;

and any repaired tasks that are clearly useful.

### Utilities

* Qwen inference wrapper;
* semantic token/logit extraction;
* paired label counterbalancing;
* episode-safe splitting;
* task-macro metrics;
* validated synthetic model-recovery utilities.

### Models

Port cognitive models individually after tests confirm equivalence.

---

# 37. Do not port

Do not initially copy:

* old activation banks;
* persistence probe infrastructure;
* L21/L22 candidate code;
* matched-change-space code;
* steering pipelines;
* patching pipelines;
* unused model-zoo variants;
* obsolete artifacts;
* one-off analysis scripts.

Keep the old repo as the historical record.

---

# 38. Regression tests against old results

For components intentionally ported, run small regression datasets.

For example:

### Bandit

New implementation should reproduce old semantic persistence probabilities within tolerance on fixed seeds.

### Foraging

Same.

### Solvability

Same.

Do not require byte-identical prompt text if SweetBean changes rendering, unless exact equivalence is intended.

---

# 39. TDD policy

Mandatory:

$$
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
$$

Before implementation, write tests for:

### Design correctness

* required factor coverage;
* illegal combinations never sampled;
* response mappings balanced;
* sequential constraints respected.

### Renderer correctness

Changing one abstract factor changes only the intended semantic construct.

### Paired designs

Latent environmental state identical across paired nuisance/counterbalance conditions.

### Data integrity

No post-termination states where hazard applies.

### Future leakage

No realized future information in predictors.

### Model recovery

Known synthetic data-generating models recoverable.

### Flexible-ceiling recovery

Flexible models reproduce simpler teachers.

### Split integrity

No pair/episode leakage.

---

# 40. Pilot gates

Before full discovery collection, each task family must satisfy:

### Engineering

* parsing >99%;
* valid semantic tokens;
* reproducible seeds;
* counterbalancing functioning.

### Behavioral usability

Across sampled conditions:

$$
0.05 <
\overline{P(\text{continue})}
<
0.95
$$

and meaningful variance exists.

Do not require every specific manipulation to move behavior in a predicted direction.

That would turn scientific hypotheses into gate criteria.

---

# 41. Artifact layout

```text
artifacts/
  discovery_v1/
    design/
      ontology.json
      sweetpea_spec.json
      condition_manifest.parquet

    behavior/
      raw/
      standardized.parquet

    splits/
      discovery.json
      interpolation_test.json
      structural_test.json

    models/
      cognitive/
      flexible/

    analysis/
      response_surface/
      model_comparison/
      residual_discovery/
      loto/

    validation/
      independent_sample.parquet
      predictions.csv

    figures/

    report.md
    run_metadata.json
```

---

# 42. Primary analyses

## Analysis 1 — Response surface

Estimate marginal and interaction effects for the major experimental dimensions.

Answer:

> What actually moves persistence?

---

## Analysis 2 — Cognitive model comparison

Compare all grounded theories on held-out data.

Answer:

> Which existing computational account best compresses behavior?

---

## Analysis 3 — Flexible ceiling

Determine remaining unexplained structure.

Answer:

> Is the hypothesis bank sufficient?

---

## Analysis 4 — Held-out task generalization

Answer:

> What computational structure generalizes across forms of goal pursuit?

---

## Analysis 5 — Residual discovery

Answer:

> Which dimensions/interactions systematically violate the strongest existing theory?

---

## Analysis 6 — Independent validation

Answer:

> Do discovered regularities predict newly generated experiments?

---

# 43. Key figures

### Figure 1 — Experimental discovery framework

Diagram:

$$
\text{Ontology}
\rightarrow
\text{SweetPea}
\rightarrow
\text{SweetBean}
\rightarrow
\text{LLM}
\rightarrow
\text{Models}
\rightarrow
\text{Discovery}.
$$

---

### Figure 2 — Persistence response surface

Task families × experimental dimensions.

Show standardized behavioral effects.

---

### Figure 3 — Cognitive model tournament

Held-out performance of:

* dynamic re-evaluation;
* perseveration;
* latent context;
* option termination;
* etc.

Include flexible ceiling.

---

### Figure 4 — Cross-task generalization

LOTO performance.

---

### Figure 5 — Explainable variance

For each cognitive theory:

$$
F_m.
$$

---

### Figure 6 — Discovered residual structure

Show the strongest interaction or missing computational ingredient.

---

### Figure 7 — Independent validation

Predicted versus observed effects in untouched conditions.

---

# 44. Automated report

`report.md` must answer:

1. How many experimental conditions were sampled?
2. How evenly was the design space covered?
3. Which dimensions have the largest marginal effects on persistence?
4. Which interactions are strongest?
5. Which cognitive theory best explains held-out behavior?
6. How close is it to the flexible predictive ceiling?
7. Which theory generalizes best to unseen task families?
8. Does a latent motivational state improve prediction?
9. How much does recent history contribute?
10. Are action and outcome history distinct?
11. Is persistence distinguishable from generic sequential choice?
12. What systematic structure remains in model residuals?
13. Does the discovered structure generalize across task families?
14. Does it predict the independent validation dataset?
15. What computational hypothesis has earned mechanistic follow-up?

---

# 45. Primary success criterion

The project succeeds if it produces a defensible answer to:

> **Which computational principles explain persistence across a broadly sampled set of LLM goal-pursuit decisions?**

It does **not** require finding one universal mechanism.

Possible successful outcomes include:

### Shared dynamic re-evaluation

A common value-based architecture explains most flexible-model performance.

### Shared ingredients, task-specific integration

History, cost, progress, and alternatives recur, but their coefficients/functions vary substantially.

### Contextual history relevance

History matters depending on inferred context/task/goal structure.

### Generic sequential choice

Persistence adds little once sequential decision structure is controlled.

### No compact theory

Flexible models strongly outperform every cognitive theory.

That would itself motivate automated theory induction.

---

# 46. ICLR minimum viable scope

Given the deadline, I would define the **must-have** version as:

1. clean repo;
2. SweetPea ontology/design generator;
3. SweetBean or equivalent rendering for 5–7 task families;
4. 2k–5k broadly sampled semantic conditions;
5. cognitive hypothesis bank;
6. validated MLP/GRU flexible ceilings;
7. held-out-condition comparison;
8. LOTO task-family comparison;
9. residual discovery;
10. one independent validation sample.

Everything else is optional.

---

# 47. Stretch goals

Only after the minimum pipeline works:

* sampling-strategy comparison;
* another open-weight model;
* model-size comparison;
* human SweetBean deployment;
* activation collection;
* mechanistic intervention;
* automated closed-loop selection via AutoRA.

---

# 48. Paper-level thesis

The paper should not require the result to be “LLMs have human-like motivation.”

The more general contribution is:

> **A coverage-oriented computational cognitive discovery framework can be used to infer how language models regulate ongoing goal pursuit without assuming in advance which human cognitive construct they implement.**

Then the substantive persistence result becomes whatever the data support.

An ideal outcome would be:

> **LLM persistence is not governed by a unitary motivational state. Instead, behavior emerges from dynamic integration of recent experience, prospective progress, costs, and alternatives, with some computational principles generalizing across otherwise distinct goal-pursuit contexts.**

But the new repo should be explicitly designed so that it can conclude something else.

That is the main reason I think restarting clean is the right move: **the repository architecture itself can now embody the scientific-discovery strategy rather than the historical sequence of hypotheses that got us here.**
