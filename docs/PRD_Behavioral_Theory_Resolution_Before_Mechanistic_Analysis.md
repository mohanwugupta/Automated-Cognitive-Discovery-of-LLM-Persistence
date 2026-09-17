# PRD — Behavioral Theory Resolution Before Mechanistic Analysis

**Project:** Automated Cognitive Discovery of LLM Persistence  
**Stage:** Discovery Round 3 / behavioral theory resolution  
**Primary model:** Qwen3.5-4B  
**Infrastructure:** SweetPea + SweetBean  
**Status:** Exploratory  
**Purpose:** Complete the behavioral/computational theory before beginning mechanistic interpretation.

---

## 1. Scientific objective

The project has now established a reasonably strong behavioral architecture:

\[
D_t^{(\tau)}
=
f(X_t,H_t;\theta_\tau)
\]

where persistence depends on current task evidence and recent history, with a common functional form but substantial task-specific parameterization.

The best current models are `dual_history` and `latent_context`. The ontology-conditioned `dual_history / M4` model reaches task-macro \(R^2=.738\), but M2–M4 are effectively tied, so the current results support **shared form + task-dependent parameters** more strongly than they support the claim that the present task ontology predicts those parameter differences.

The final untouched behavioral test is strong:

\[
R^2_{\mathrm{macro}}=.763,
\]

with pooled \(R^2=.862\), RMSE \(=.491\), and calibration slope \(=.824\).

The immediate goal is therefore:

> **Resolve the remaining uncertainty about the computational architecture of persistence well enough to define a precise mechanistic target.**

This PRD covers five steps:

1. finish outstanding validity/audit work;
2. freeze the surviving cognitive theories;
3. introduce a contextual-history experiment family;
4. actively sample conditions that are informative about the surviving theories;
5. determine the narrowest behavioral theory justified by the resulting evidence.

No activation analysis is part of this PRD.

---

## 2. Current theory state

The following claims are reasonably supported:

### A. Invariant cross-task coefficients are inadequate

Fully shared M1 versions perform substantially worse than M2–M4.

### B. Task-specific parameterization is important

Task-specific, random-effects hierarchical, and ontology-conditioned variants perform similarly well for the leading architecture.

### C. Recent history matters

`dual_history`, `outcome_history`, and `latent_context` dominate immediate-state and latent-motivation models.

### D. Latent motivation is weak

A slow unitary motivational-state account is currently poorly supported.

### E. Dual history and latent context remain behaviorally unresolved

The paired bootstrap does not distinguish them.

### F. Active refinement appears useful

The existing active mixture improves final-test performance over continued coverage-only sampling across the tested budgets.

---

## 3. Main unresolved scientific question

Two broad interpretations survive.

### H-DH — Direct dual-history integration

Recent actions and outcomes directly enter the persistence decision:

\[
D_t
=
f_\tau(
X_t,
A_t,
O_t
).
\]

Here:

- \(A_t\) summarizes recent action history;
- \(O_t\) summarizes recent outcome history;
- task-specific parameters determine their influence.

History is behaviorally relevant in its own right.

### H-LC — Latent-context-dependent history

Recent experiences influence behavior according to whether they are inferred to belong to the current latent state/context:

\[
z_t
=
P(\text{latent context}\mid\text{current cues, history})
\]

and:

\[
D_t
=
f_\tau(
X_t,
H_t;
z_t
).
\]

History is therefore **selectively retrieved or weighted according to contextual relevance**.

---

## 4. What would distinguish them

The key contrast is not simply:

> Does history matter?

Both theories predict yes.

Instead:

> **Does the model use the most recent history indiscriminately, or selectively reinstate history associated with the currently inferred context?**

This motivates a new experiment family involving:

\[
A\rightarrow B\rightarrow A
\]

context reinstatement and explicit change points.

---

# WORKSTREAM A — Complete Outstanding Audits

## 5. Flexible-ceiling teacher-recovery audit

Current result:

\[
30/36
\]

matched flexible-teacher checks passed.

This must be resolved before using terms such as “flexible ceiling.”

### Required output

For each of the 36 checks report:

- teacher architecture;
- student architecture;
- sharing structure;
- teacher \(R^2\);
- student \(R^2\);
- \(\Delta R^2\);
- pass/fail;
- failure reason if identifiable.

---

## 6. Classify the six failures

Assign each failure to one of:

### Optimization failure
Student does not fit training teacher adequately.

### Generalization failure
Training teacher fit is good but held-out teacher prediction is worse.

### Feature mismatch
Student lacks information available to teacher.

### Sharing mismatch
Student architecture is not structurally equivalent.

### Numerical/data bug
Scaling, masks, splits, missingness, etc.

### Expected regularization difference
Student intentionally constrained.

Do not aggregate these categories.

---

## 7. Mandatory flexible-model sanity tests

For every flexible architecture that will appear in a paper-level comparison:

### Synthetic teacher

Generate:

\[
y=f_{\mathrm{teacher}}(X,H)
\]

on the real design matrix.

Require:

\[
R^2_{\mathrm{student,teacher}}\ge .98
\]

unless a documented reason makes that inappropriate.

### Actual model distillation

Train on frozen predictions from:

- dual history;
- latent context.

Require near-perfect recovery.

Until passed, do not call the flexible model an upper bound.

---

## 8. M2/M3/M4 uncertainty analysis

The current point estimates are nearly identical.

Quantify whether they are meaningfully distinguishable.

For each candidate architecture compute paired bootstrap distributions for:

\[
R^2_{M2}-R^2_{M3}
\]

\[
R^2_{M2}-R^2_{M4}
\]

\[
R^2_{M3}-R^2_{M4}.
\]

Bootstrap unit must preserve the appropriate semantic-condition/task structure.

Report:

- mean difference;
- median;
- 95% CI;
- probability each model wins.

---

## 9. Parameter variance decomposition

For the winning architecture:

\[
\theta_\tau
=
\mu+Bz_\tau+u_\tau.
\]

For each parameter \(j\), estimate:

\[
\operatorname{Var}(\theta_{\tau j})
\]

and partition approximately into:

\[
\operatorname{Var}(B_jz_\tau)
\]

and:

\[
\operatorname{Var}(u_{\tau j}).
\]

Define:

\[
Q_j
=
\frac{
\operatorname{Var}(B_jz_\tau)
}{
\operatorname{Var}(\theta_{\tau j})
}.
\]

This determines whether the ontology explains task variation.

---

## 10. Parameter sign consistency

For the leading dual-history architecture analyze at minimum:

- success evidence;
- progress;
- continuation cost;
- disengagement value;
- action-history kernel;
- outcome-history kernel.

For each parameter report:

\[
P(\beta_{\tau j}>0)
\]

or empirical sign consistency across task families.

Distinguish:

### Shared sign
Same qualitative role, different strength.

### Mixed sign
Role fundamentally changes with task.

### Near-zero
Construct has limited broad relevance.

This will help define what “shared architecture” actually means.

---

## 11. Information Sampling audit

Information Sampling remains structurally unusual and previously caused extreme LOTO failures.

Audit:

- definition of semantic `continue`;
- factor availability masks;
- persistence-logit distribution;
- factor distribution;
- scaling;
- response-token mapping;
- whether success/progress semantics match other tasks;
- whether information accumulation creates a qualitatively distinct state representation.

Run all primary hierarchy analyses:

1. with Information Sampling;
2. without Information Sampling.

Do not remove it from the primary dataset unless there is an implementation error.

---

## 12. Final-validation calibration audit

For the untouched validation set report task-macro primary metrics:

\[
R^2_{\mathrm{macro}}
\]

\[
RMSE_{\mathrm{macro}}
\]

\[
r_{\mathrm{macro}}.
\]

Also report:

\[
\hat y=\alpha+\beta y
\]

with current:

\[
\beta\approx .824.
\]

Generate:

- calibration by decile;
- per-task calibration;
- extreme-logit residuals.

No recalibration for the primary frozen-model metric.

---

# WORKSTREAM B — Freeze the Surviving Cognitive Models

## 13. Freeze criterion

After Workstream A, retain models that satisfy both:

### Predictive viability

Within:

\[
\Delta R^2 \le .03
\]

of the best cognitive model on task-macro held-out performance;

and

### Distinct theoretical interpretation

The model must make meaningfully different computational assumptions.

Expected surviving set:

- dual history;
- latent context;
- possibly outcome history.

Do not carry weak models forward merely because they existed in Round 1.

---

## 14. Frozen model package

For every surviving model save:

```text
model_spec.json
parameters.pt / parameters.csv
training_condition_hashes.json
prediction_function.py
task_parameter_table.csv
```

After freezing:

- no refitting on discriminating-test observations;
- no changing architecture based on Round-3 outcomes;
- no changing history kernels after seeing target data.

---

## 15. Freeze task-level parameterization

For each model preserve:

- M2 task-specific version;
- best hierarchical version;
- ontology-conditioned version if identifiable.

The model-discrimination experiment should compare theories rather than accidentally compare different amounts of parameter flexibility.

Primary theoretical comparison should use matched effective flexibility wherever possible.

---

# WORKSTREAM C — Add a Contextual-History Experiment Family

## 16. Purpose

Construct conditions in which:

\[
\text{recent history}
\]

and:

\[
\text{contextually relevant history}
\]

come apart.

This is where dual history and latent context should make meaningfully different predictions.

---

## 17. Core paradigm: A → B → A reinstatement

Each episode contains three phases:

\[
A_1
\rightarrow
B
\rightarrow
A_2.
\]

### Phase A1

Context A produces a characteristic history:

\[
H_A.
\]

### Phase B

Context B produces a conflicting history:

\[
H_B.
\]

### Phase A2

Return to A and obtain the critical persistence decision.

---

## 18. Example contrast

Condition 1:

\[
A_1: + + +
\]

\[
B: - - -
\]

\[
A_2: ?
\]

Condition 2:

\[
A_1: - - -
\]

\[
B: + + +
\]

\[
A_2: ?
\]

A pure recency-heavy model tends to favor:

\[
H_B.
\]

A latent-context model can reinstate:

\[
H_A
\]

when the context returns to A.

---

## 19. Context cues

Vary context evidence independently of numerical history.

Possible cues:

- environment label;
- latent payoff regime;
- visual/textual context description;
- causal rule;
- task-state cue.

Do not rely on only a name change such as “Context A.”

The cue must predict a real difference in underlying environment statistics.

---

## 20. Context reliability

Manipulate:

\[
P(z_t\mid cue).
\]

Suggested levels:

```text
low
medium
high
```

This makes context inference graded.

Latent-context theories should predict stronger reinstatement when context cues are more reliable.

---

## 21. Change-point manipulation

Add conditions where identical recent histories are followed by either:

### Same environment

> The conditions governing outcomes remain unchanged.

### Change point

> The environment has changed; earlier observations may no longer describe the current process.

Prefer operational environmental changes over explicit metacognitive instructions where possible.

---

## 22. History conflict factors

SweetPea should generate combinations of:

### A-history valence

```text
positive
negative
mixed
```

### B-history valence

```text
positive
negative
mixed
```

### Context return

```text
A
B
novel C
```

### Cue reliability

```text
low
medium
high
```

### Current prospective evidence

```text
weak
medium
strong
```

### Cost / alternative value

retain existing ontology dimensions.

---

## 23. Current-state matching

Critical pairs must hold fixed:

- current continuation value;
- disengagement value;
- continuation cost;
- immediate success evidence;
- progress;
- labels.

Only contextual history assignment should differ.

This is essential for identifying history-retrieval effects.

---

## 24. Cross-domain implementation

Do not make this one arbitrary abstract paradigm.

Implement contextual-history variants in at least **three** existing task families, preferably:

1. Bandit/reward pursuit
2. Foraging
3. Debugging or solvability

Stretch:

4. Waiting

This tests whether contextual reinstatement itself generalizes.

---

# WORKSTREAM D — Active Discriminating Sample

## 25. Candidate generation

Use SweetPea to generate a large legal candidate set from:

- original experiment grammar;
- new contextual-history grammar.

Target:

\[
N_{\mathrm{candidate}}\ge 20{,}000
\]

semantic conditions.

No LLM inference needed to generate candidates.

---

## 26. Frozen-model predictions

For every candidate condition \(x\), obtain predictions from:

\[
\hat D_{DH}(x)
\]

and:

\[
\hat D_{LC}(x).
\]

If outcome-history remains viable, include:

\[
\hat D_{OH}(x).
\]

---

## 27. Disagreement score

Define:

\[
D(x)
=
\operatorname{Var}_m[
\hat D_m(x)
]
\]

or pairwise absolute differences.

Primary:

\[
D_{DH,LC}(x)
=
|\hat D_{DH}(x)-\hat D_{LC}(x)|.
\]

---

## 28. Uncertainty score

Use bootstrap/model ensembles to estimate:

\[
U(x)
=
\frac{1}{M}
\sum_m
\operatorname{Var}
[
\hat D_m(x)
].
\]

This distinguishes useful disagreement from unstable extrapolation.

---

## 29. Coverage score

Retain:

\[
C(x)
\]

measuring underrepresentation in:

- marginal factors;
- factor pairs;
- task × factor combinations;
- contextual-history cells.

---

## 30. Active discrimination score

Use:

\[
S(x)
=
.50D(x)
+
.25U(x)
+
.25C(x).
\]

Normalize each component before combination.

The exact weights can be changed only before observing Round-3 outcomes.

---

## 31. Sampling allocation

Target approximately:

\[
1{,}200
\]

new semantic conditions.

Suggested allocation:

### 60% discriminating sample

Highest-scoring conditions from \(S(x)\).

### 20% coverage sample

Coverage-oriented random conditions from the expanded space.

### 20% pure random reference

Uniform legal sample.

This allows us to distinguish targeted gains from distribution shifts.

---

## 32. Balanced task/domain sampling

Do not allow one task family to dominate disagreement selection.

Within each task/context domain impose quotas.

For example:

\[
N_{\tau}\approx
\frac{N}{N_{\mathrm{domains}}}.
\]

---

## 33. Counterbalancing

All selected conditions retain:

- arbitrary response mapping;
- paired labels;
- deterministic latent environmental seed where applicable;
- semantic-condition hash.

As before, downstream analysis uses semantic persistence logits.

---

## 34. Untouched discrimination test

Before model updating, reserve approximately:

\[
25\%
\]

of Round-3 conditions as a model-discrimination test set.

Neither model may be refit on these conditions before comparison.

---

## 35. Primary model-discrimination metric

For each held-out condition calculate prediction error:

\[
e_m=
(D-\hat D_m)^2.
\]

Compare:

\[
\Delta e
=
e_{DH}-e_{LC}.
\]

Bootstrap over semantic conditions, stratified by task/domain.

Report:

- overall difference;
- per-task difference;
- contextual-history subset;
- high-disagreement subset;
- random-reference subset.

---

## 36. Model updating after frozen comparison

Only after the frozen comparison is complete:

1. add Round-3 training observations;
2. refit surviving models;
3. reevaluate on the untouched discrimination test;
4. reevaluate on a broad final coverage-random sample if budget permits.

This separates:

\[
\text{prediction}
\]

from:

\[
\text{post-hoc accommodation}.
\]

---

# WORKSTREAM E — Resolve/Narrow the Behavioral Theory

## 37. Outcome A — Direct dual-history integration wins

Evidence:

- dual history beats latent context on frozen predictions;
- especially in contextual reinstatement conditions;
- recent history dominates older context-matched history;
- context cues add little after explicit action/outcome history.

Conclusion:

> **Persistence is primarily governed by direct integration of recent action and outcome history with current evidence.**

Mechanistic target:

\[
A_t,\quad O_t
\]

and their integration into continuation choice.

---

## 38. Outcome B — Latent context wins

Evidence:

- context return selectively reinstates context-matched histories;
- effect scales with context-cue reliability;
- latent-context predictions outperform direct-history predictions;
- change points downweight history from obsolete contexts.

Conclusion:

> **Persistence depends on context-sensitive inference about which prior experiences are relevant to the current state.**

Mechanistic target:

\[
z_t
\]

and context-dependent history retrieval/weighting.

---

## 39. Outcome C — Observational equivalence persists

If targeted conditions still cannot reliably distinguish the models:

> **Dual-history and latent-context accounts remain observationally equivalent over the experimentally sampled domain.**

Do not invent a winner.

Mechanistic strategy should then target their common computational content:

\[
\text{outcome-history representation}
\]

and ask whether contextual information modulates it internally.

---

## 40. Outcome D — Third architecture emerges

If both surviving models fail badly on Round 3:

- inspect residual structure;
- fit flexible model;
- determine what newly sampled conditions expose.

Only then reopen the hypothesis bank.

Do not immediately create a new named construct from one interaction.

---

## 41. Behavioral-theory stopping rule

After Round 3, stop behavioral expansion if all are true:

1. one or two models explain most validated behavior;
2. untouched validation remains strong;
3. residual discovery finds no substantial broad structure;
4. remaining uncertainty concerns implementation rather than large behavioral prediction gaps.

The project should then move to mechanistic interpretation.

---

## 42. Mechanistic handoff specification

The output of this PRD must explicitly identify:

### Computational variable(s)

For example:

```text
outcome-history summary
action-history summary
latent-context probability
contextual history relevance
```

### Behavioral equation

For example:

\[
D_t
=
\beta_{\tau,X}X_t
+
\beta_{\tau,A}A_t
+
\beta_{\tau,O}O_t.
\]

### Task-specific coefficients

Save:

\[
\beta_{\tau,*}
\]

for each task.

### Intervention predictions

For each task, derive:

\[
\frac{\partial D_\tau}
{\partial O_t}
\]

etc.

These become quantitative predictions for steering/patching in the next PRD.

---

## 43. Why task-specific coefficients matter for mechanism

The mechanistic phase should eventually test:

\[
\boxed{
\Delta D_\tau^{predicted}
=
\beta_{\tau,O}
\Delta O
}
\]

against:

\[
\boxed{
\Delta D_\tau^{observed}
}
\]

after neural intervention.

Therefore this PRD must output well-estimated task-level coefficients with uncertainty.

This is the bridge between:

\[
\text{behavior}
\rightarrow
\text{computational theory}
\rightarrow
\text{causal implementation}.
\]

---

## 44. TDD requirements

All new work follows:

\[
\text{RED}
\rightarrow
\text{GREEN}
\rightarrow
\text{REFACTOR}.
\]

### Audit tests

- failed teacher checks correctly surfaced;
- no failed check silently omitted;
- M2/M3/M4 share identical splits;
- bootstrap preserves paired conditions.

### Context experiment tests

- A→B→A ordering correct;
- context-specific histories stored separately;
- current test state matched across critical contrasts;
- context-cue reliability generated correctly;
- no future information leakage.

### Sampling tests

- frozen models never see Round-3 outcomes during scoring;
- disagreement scores verified on synthetic examples;
- quotas respected;
- duplicates excluded;
- coverage/random subsets drawn independently.

### Theory comparison tests

Synthetic direct-history generator should favor dual history.

Synthetic latent-context generator should favor latent context.

When models are mathematically equivalent, pipeline must report equivalence rather than arbitrary winner.

---

## 45. Suggested repository additions

```text
src/cognitive_discovery/
    audits/
        teacher_failures.py
        hierarchy_uncertainty.py
        parameter_consistency.py
        information_sampling.py

    experiments/
        contextual_history/
            factors.py
            reinstatement.py
            change_point.py
            renderers.py

    sampling/
        theory_disagreement.py
        model_uncertainty.py
        discrimination_mixture.py

    theory_resolution/
        frozen_predictions.py
        paired_model_test.py
        equivalence.py
        theory_report.py
```

---

## 46. Required artifacts

```text
artifacts/theory_resolution_v1/
    audits/
        flexible_teacher_checks.csv
        hierarchy_bootstrap.csv
        parameter_variance.csv
        parameter_signs.csv
        information_sampling_audit.csv

    frozen_models/
        dual_history/
        latent_context/
        outcome_history/

    contextual_history/
        condition_manifest.parquet
        factor_coverage.csv

    active_sampling/
        candidate_pool.parquet
        candidate_predictions.parquet
        selected_conditions.parquet
        sampling_scores.csv

    discrimination/
        frozen_model_errors.csv
        model_difference_bootstrap.csv
        per_task_comparison.csv

    theory/
        final_model_comparison.csv
        final_parameters.csv
        mechanistic_targets.json

    figures/
    report.md
    run_metadata.json
```

---

## 47. Key figures

### Figure 1 — Parameter architecture

Shared form with task-specific weights.

### Figure 2 — Parameter consistency

Success, progress, costs, action history, outcome history across tasks.

### Figure 3 — Context reinstatement predictions

Dual-history versus latent-context predictions for:

\[
A\rightarrow B\rightarrow A.
\]

### Figure 4 — Frozen model discrimination

Observed behavior against each frozen theory in actively sampled conditions.

### Figure 5 — Context reliability effect

History reinstatement as a function of cue reliability.

### Figure 6 — Final theory comparison

Broad validation + discriminating conditions.

---

## 48. Automated report must answer

1. Which six teacher-recovery checks failed and why?
2. Are flexible ceilings now valid for the relevant comparison?
3. Are M2, M3, and M4 meaningfully different?
4. How much task-parameter variance is explained by the ontology?
5. Which computational parameters have stable signs across tasks?
6. Is Information Sampling structurally different or incorrectly mapped?
7. Which cognitive theories survive the freeze criterion?
8. Where in the expanded experiment space do those theories disagree?
9. Does targeted sampling distinguish dual-history from latent-context?
10. Does A→B→A context reinstatement occur?
11. Does context reliability modulate history reinstatement?
12. Does evidence favor direct history integration, latent-context inference, or observational equivalence?
13. What computational variable has earned mechanistic investigation?
14. What quantitative task-specific intervention effects does the behavioral model predict?

---

## 49. Success criterion

This PRD succeeds if it leaves us with a statement of the form:

> **Across diverse goal-pursuit tasks, persistence is governed by [specific computational architecture], parameterized differently by task. The remaining task-dependent parameters are [partly/not] predicted by task structure, and targeted active experiments [resolve/do not resolve] whether recent history acts directly or through contextual inference.**

Most importantly, the output must provide a **specific computational variable and quantitative behavioral model suitable for causal mechanistic testing**.

That is the gate to the next project stage:

\[
\boxed{
\text{behavioral discovery}
\rightarrow
\text{computational theory}
\rightarrow
\text{mechanistic implementation}
}
\]

and the next PRD should only begin once that target is earned.
