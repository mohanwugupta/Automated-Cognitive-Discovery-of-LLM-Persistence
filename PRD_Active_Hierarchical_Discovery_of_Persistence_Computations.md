# PRD — Active Hierarchical Discovery of Persistence Computations

**Project:** Automated Cognitive Discovery of LLM Persistence  
**Stage:** Discovery round 2  
**Primary model:** Qwen3.5-4B  
**Primary infrastructure:** SweetPea + SweetBean  
**Status:** Exploratory / sequential scientific discovery  
**Primary purpose:** Determine whether persistence across task families is governed by a common computational architecture whose parameters depend systematically on task structure.

---

## 1. Starting evidence

Discovery Round 1 sampled:

- **2,800 semantic conditions**
- **18,240 counterbalanced observations**
- **7 task families**
- exactly **400 semantic conditions per family**

Major results:

### Response surface

Persistence increases with:

- success evidence;
- positive progress;
- low disengagement value;
- high continuation value.

Persistence decreases with:

- low success evidence;
- negative progress;
- high disengagement value;
- high continuation cost.

### Cognitive models

Best interpolation model family:

\[
\texttt{dual\_history}
\]

with approximately:

\[
R^2=.789.
\]

`latent_context` and `outcome_history` are close competitors.

Latent motivational-state performance is substantially weaker:

\[
R^2=.183.
\]

Generic sequential choice is also weak:

\[
R^2=.089.
\]

### Strict task transfer

Exact parameter reuse across held-out task families performs poorly.

Information Sampling is especially catastrophic under LOTO.

Therefore:

\[
\boxed{\text{shared exact parameters}}
\]

is poorly supported.

This does **not** distinguish:

\[
\boxed{\text{shared architecture + task-specific parameters}}
\]

from:

\[
\boxed{\text{different algorithms}}.
\]

### Validation

Frozen-model independent validation:

\[
R^2=.772.
\]

---

## 2. Primary scientific question

> **Does one computational architecture explain persistence across task families when its parameters are allowed to adapt systematically to the structure of the task?**

Secondary:

> **Can the required task-specific parameters themselves be predicted from the ontology of the task?**

This defines three increasingly strong levels of computational generality.

---

## 3. Three levels of generality

### Level 1 — Shared exact parameters

\[
\theta_t=\theta.
\]

Already weakly supported.

### Level 2 — Shared architecture, free task parameters

\[
y=f(X,H;\theta_t).
\]

All tasks use the same mathematical form but independently fitted parameters.

This may already describe the current dual-history result.

### Level 3 — Task-conditioned shared architecture

\[
\theta_t=\mu+Bz_t+u_t.
\]

Task parameters are partially predictable from observable task structure.

This is the strongest hypothesis to test next.

---

## 4. Competing architecture families

Do not restrict this test to dual history.

Run the hierarchy over at least:

### A. Immediate-state evaluation

\[
D_t=f(X_t).
\]

### B. Choice perseveration

\[
D_t=f(X_t)+\kappa_AA_t.
\]

### C. Outcome history

\[
D_t=f(X_t)+\kappa_OO_t.
\]

### D. Dual history

\[
D_t=f(X_t)+\kappa_AA_t+\kappa_OO_t.
\]

### E. Dynamic re-evaluation

Current evidence/history used to update continuation versus disengagement prospects.

### F. Latent-context model

History influence weighted by inferred contextual relevance.

### G. Latent motivation

Retain as a falsification competitor.

---

## 5. Task ontology for hierarchical parameter prediction

Define a small task descriptor vector:

\[
z_t.
\]

Do **not** put all existing experimental factors into it.

These describe the *structure of the task*, not the current state.

Initial task descriptors should be limited to approximately 4–6 variables because there are only seven task families.

Candidates:

### Outcome dependence

Does current action influence later outcomes?

- low
- high

### Evidence accumulation

Does information accumulate meaningfully across steps?

- no
- yes

### Explicit progress

Does the task expose a meaningful progress signal?

- no
- yes

### Reward stationarity

Are recent outcomes predictive of future opportunities?

- changing
- stable

### Absorbing disengagement

Does disengagement terminate the current pursuit?

- no
- yes

### Effort accumulation

Does continuation incur cumulative effort/resource cost?

- low
- high

Freeze descriptor definitions before fitting the task-conditioned model.

---

## 6. Model hierarchy

For every candidate architecture fit four variants.

### M1 — Fully shared

\[
\theta_t=\mu.
\]

### M2 — Task-specific

\[
\theta_t
\]

independently estimated.

This is the maximum task-flexibility reference.

### M3 — Random-effects hierarchical

\[
\theta_t=\mu+u_t
\]

\[
u_t\sim\mathcal N(0,\Sigma).
\]

This tests whether partial pooling improves generalization.

### M4 — Ontology-conditioned hierarchical

\[
\theta_t=\mu+Bz_t+u_t.
\]

Primary new model.

---

## 7. Critical interpretation

These models answer different questions.

If:

\[
M2\gg M1
\]

then exact shared parameters are inappropriate.

Already expected.

If:

\[
M3\approx M2
\]

then the **same architecture with partially pooled task parameters** captures almost all task-specific performance.

If:

\[
M4>M3
\]

then task structure explains meaningful parameter variation.

That would be strong evidence for:

\[
\boxed{\text{shared architecture instantiated according to task structure}}
\]

rather than unrelated task algorithms.

---

## 8. Zero-shot task prediction under the hierarchical model

For held-out task \(t^\*\), ordinary hierarchical random effects cannot estimate:

\[
u_{t^\*}
\]

without target data.

Therefore strict zero-shot M3 predicts:

\[
\hat\theta_{t^\*}=\mu.
\]

This explains why vanilla hierarchical LOTO can still perform badly.

The ontology-conditioned model instead predicts:

\[
\boxed{\hat\theta_{t^\*}=\mu+Bz_{t^\*}}
\]

without observing target-task choices.

This is the fair test of whether **task structure predicts implementation parameters**.

---

## 9. Few-shot adaptation

Also evaluate:

\[
n=1,4,8,16,32,64
\]

target semantic conditions.

Update only:

\[
u_{t^\*}
\]

while keeping:

\[
\mu,B
\]

frozen.

Plot performance versus \(n\).

Interpretation:

### Fast adaptation

Same architecture, task-specific calibration.

### Slow adaptation

Substantial task-specific computation.

---

## 10. Fix the flexible ceiling

This is mandatory before comparing explainable fractions.

The current flexible models are fully shared while the winning cognitive models are hierarchical/task-specific.

That is not a valid ceiling comparison.

### 10.1 Train flexible models under matched sharing structures

Run:

#### Linear interactions

- fully shared
- task-specific
- hierarchical/task-embedding

#### MLP

- fully shared
- task-specific heads
- shared trunk + task-specific head
- task-conditioned task embedding

#### GRU

same variants where sequence data permit.

---

## 11. Preferred flexible hierarchical architecture

Use:

\[
h=g_\phi(X,H)
\]

as a shared representation.

Then:

\[
D_t=w_t^\top h+b_t.
\]

Task-specific head:

\[
w_t=w_0+Wz_t+u_t.
\]

This gives the flexible model the same structural advantage as the cognitive hierarchy.

---

## 12. Synthetic teacher recovery

Before calling any flexible model a ceiling:

Generate targets from:

- dual history;
- latent context;
- nonlinear interaction model.

using the **real Discovery Round 1 design matrix**.

Require flexible models to recover teacher predictions approximately:

\[
R^2_{\text{teacher}}\ge.98
\]

or:

\[
\Delta R^2\le .02
\]

relative to the teacher.

---

## 13. Real-model distillation

Train flexible predictors directly on:

\[
\hat D_{\text{dual-history}}
\]

from the fitted dual-history model.

If they cannot reproduce it, the flexible training pipeline remains invalid.

---

## 14. Recalculate explainable fraction

Only after fixing the ceiling compute:

\[
F_m=\frac{R^2_m-R^2_{\text{null}}}{R^2_{\text{flex}}-R^2_{\text{null}}}.
\]

If:

\[
F_m>1
\]

after matched architectures and proper validation, interpret this as simpler-model generalization advantage—not as >100% “explained variance.”

Cap nothing silently.

Report the raw numbers.

---

## 15. Audit Information Sampling

The LOTO collapse for Information Sampling is extreme enough to require diagnosis.

Do not immediately treat it as psychological evidence.

Audit:

### Factor semantics

Does:

\[
\text{success evidence}
\]

mean the same thing there as elsewhere?

### Target semantics

Is `continue`:

> gather more evidence

rather than:

> continue pursuing reward?

Document this explicitly.

### Feature availability

Compare masks across tasks.

### Factor distributions

Plot standardized distributions by task.

### Persistence-logit scale

Compare:

- mean;
- variance;
- range.

### Response mapping

Verify semantic token handling.

### Condition distribution

Check for task-exclusive combinations.

---

## 16. Information-Sampling sensitivity analysis

Run LOTO:

### with Information Sampling

and:

### without Information Sampling.

If the global conclusion changes substantially, report Information Sampling as a structurally distinct domain rather than allowing it to dominate macro generalization.

Do **not** discard it merely because it transfers poorly.

---

## 17. Residual-discovery repair

Current reporting identifies:

`history_outcome_kernel × goal_continuity`

as the strongest residual feature even though plotted held-out gains appear approximately zero or negative.

Change the rule.

A residual effect can be called **discovered structure** only if:

\[
\Delta R^2_{\text{heldout}}>0
\]

and bootstrap support excludes or strongly disfavors zero.

Otherwise report:

> **No tested residual interaction improved held-out prediction.**

---

## 18. Nested residual discovery

Use:

### Discovery subset

Select candidate residual terms.

### Residual validation subset

Evaluate them.

Never select and evaluate an interaction on the same observations.

Use stability selection across bootstrap folds.

---

## 19. Independent validation calibration

For the frozen model already evaluated at:

\[
R^2=.772
\]

report:

\[
\hat y=\alpha+\beta y.
\]

Specifically:

- calibration slope;
- intercept;
- RMSE;
- MAE;
- residual SD;
- performance by prediction decile;
- performance by task.

Do not recalibrate the existing validation predictions for the primary metric.

---

## 20. Active discovery round

Round 1 was deliberately coverage-oriented.

That makes a model-informed second round scientifically appropriate.

The goal is **not** simply to select maximum model disagreement.

Instead optimize three objectives:

\[
Score(x)=\lambda_C C(x)+\lambda_I I(x)+\lambda_D D(x).
\]

---

## 21. Component 1 — Coverage score

\[
C(x)
\]

is high when condition \(x\) lies in an underrepresented region of the design space.

Use:

- factor marginal coverage;
- pairwise-factor coverage;
- task × factor coverage.

This protects against narrow exploitation.

---

## 22. Component 2 — Parameter-information score

\[
I(x)
\]

measures how informative the condition is expected to be about hierarchical parameters.

Possible implementations:

- approximate Fisher information;
- ensemble variance over \(\theta_t\);
- posterior/pseudo-bootstrap parameter uncertainty;
- leverage score of hierarchical design matrix.

Prefer a simple robust implementation for ICLR.

Bootstrap ensemble variance is sufficient.

---

## 23. Component 3 — Architecture-disagreement score

\[
D(x)
\]

measures disagreement between the leading architecture families.

Initially:

\[
\text{dual history}
\]

versus:

\[
\text{latent context}.
\]

Use only as one part of the sampling policy.

Do not let disagreement dominate coverage.

---

## 24. Round-2 sampling mixture

Recommended allocation:

### 40% Coverage

Balanced random sampling from poorly covered regions.

### 40% Parameter information

Conditions expected to reduce uncertainty about:

\[
\theta_t,\mu,B.
\]

### 20% Model distinction

Conditions where leading cognitive architectures make meaningfully different predictions.

This explicitly implements:

\[
\boxed{\text{explore}\rightarrow\text{partially exploit}}
\]

rather than switching entirely to hypothesis-driven sampling.

---

## 25. Round-2 budget

Target approximately:

\[
1{,}400
\]

new semantic conditions:

\[
200\times7\text{ task families}.
\]

Counterbalance label mappings as before.

This should be sufficient for a meaningful active round without creating another huge dataset.

---

## 26. Sampling implementation with SweetPea

SweetPea continues to define the **legal experiment space**.

The active sampler does not directly generate prompts.

Pipeline:

\[
\text{candidate pool}
\]

generated from SweetPea constraints

\[
\downarrow
\]

score legal candidates

\[
\downarrow
\]

select conditions

\[
\downarrow
\]

SweetBean renders experiments.

Generate a candidate pool at least:

\[
10\times
\]

larger than the required sample.

---

## 27. Avoid duplicate active samples

Exclude:

- Discovery Round 1 conditions;
- independent-validation conditions;
- exact counterfactual duplicates unless deliberately selected.

Maintain a design hash.

---

## 28. Final untouched test set

After active sampling/model updating, generate one additional **coverage-random** test set.

Do not use active criteria.

Target:

\[
700-1{,}000
\]

semantic conditions.

This remains the final evaluation distribution.

Thus:

\[
\text{broad discovery}
\]

\[
\rightarrow
\]

\[
\text{active refinement}
\]

\[
\rightarrow
\]

\[
\text{broad independent test}.
\]

That is the clean scientific arc.

---

## 29. Compare sampling efficiency

This now gives us a small but valuable automated-science analysis.

Using the same Round-1 model state, simulate/select equal budgets from:

### Coverage-only

### Active mixture

Evaluate both on the untouched final test set.

Budgets:

\[
100,250,500,1000.
\]

Question:

> Does the active second-stage policy improve discovery efficiency after broad exploration?

This is much more theoretically motivated than comparing five selection algorithms from scratch.

---

## 30. Primary model-comparison metrics

For each architecture report:

### Interpolation

Held-out-condition macro \(R^2\).

### Task-specific ceiling

Per-task fitted performance.

### Hierarchical performance

Partial-pooling performance.

### Strict zero-shot

Held-out task with:

\[
\theta^*=\mu
\]

for random-effects models.

### Ontology zero-shot

Held-out task with:

\[
\theta^*=\mu+Bz^*.
\]

### Few-shot adaptation

Performance after \(n\) target samples.

---

## 31. Parameter variance decomposition

For each parameter \(j\):

\[
\operatorname{Var}(\theta_{tj})=\operatorname{Var}(B_jz_t)+\operatorname{Var}(u_{tj}).
\]

Report the fraction:

\[
Q_j=\frac{\operatorname{Var}(B_jz_t)}{\operatorname{Var}(\theta_{tj})}.
\]

Interpretation:

### High \(Q_j\)

Task structure predicts how this computation is instantiated.

### Low \(Q_j\)

Parameter variation remains idiosyncratic.

This is scientifically more informative than asking whether coefficients are identical.

---

## 32. Parameter visualization

For the winning architecture show task-specific estimates for:

- action-history weight;
- outcome-history weight;
- current-success evidence;
- progress;
- continuation cost;
- outside-option value.

Use partial-pooling intervals.

This figure may be central to the paper.

---

## 33. Test parameter sign consistency

Even if magnitudes vary, ask:

\[
P(\beta_{tj}>0)
\]

across tasks.

For example, perhaps all tasks show:

\[
\beta_{\text{success}}>0
\]

but magnitudes differ dramatically.

That is a weaker but meaningful type of computational generality.

---

## 34. Compare dual history versus latent context

Do not choose based solely on point-estimate \(R^2\).

Use paired condition-level or episode-level bootstrap differences:

\[
\Delta R^2=R^2_{\text{dual}}-R^2_{\text{context}}.
\]

Report:

- macro difference;
- CI;
- per-task difference;
- Round-2 prediction;
- final independent-test difference.

If indistinguishable:

> retain both as observationally equivalent candidate explanations.

Do not manufacture a winner.

---

## 35. Model recovery

Generate synthetic data under:

### Shared dual history

### Task-specific dual history

### Hierarchical dual history

### Ontology-conditioned dual history

### Latent context

### Latent motivation

Run the full comparison.

Ensure that:

\[
M1,M2,M3,M4
\]

can be distinguished at the current number of tasks/conditions.

This is especially important because there are only seven task families.

---

## 36. Complexity constraint

With only seven task families:

\[
B
\]

must remain small.

Use:

- strong ridge/shrinkage;
- maximum 4–6 task descriptors;
- no high-order ontology interactions initially.

If synthetic recovery says \(B\) cannot be identified, report that and fall back to random-effects hierarchy.

---

## 37. Main theoretical outcomes

### Outcome A — Shared architecture, task-conditioned parameters

Pattern:

\[
M4\approx M2
\]

and:

\[
M4>M3>M1.
\]

Ontology-conditioned zero-shot improves substantially.

Conclusion:

> **LLM persistence uses a common computational architecture whose parameters are systematically adapted to task structure.**

This would be a strong result.

### Outcome B — Shared architecture with idiosyncratic calibration

Pattern:

\[
M3\approx M2
\]

but:

\[
M4\approx M3.
\]

Conclusion:

> A shared computational form generalizes, but task structure does not yet predict its precise parameterization.

Still scientifically meaningful.

### Outcome C — Truly task-specific algorithms

Pattern:

\[
M2\gg M3,M4.
\]

No rapid few-shot adaptation.

Conclusion:

> The common cognitive-model fit masks substantial algorithmic heterogeneity.

### Outcome D — Latent-context architecture wins

Context model consistently outperforms dual-history on active and final samples.

Conclusion:

> History appears to affect persistence primarily through inference about the current latent context rather than direct history biases.

### Outcome E — Flexible models decisively outperform hypothesis bank

After ceiling correction:

\[
R^2_{\text{flex}}\gg R^2_{\text{cognitive}}.
\]

Conclusion:

> Existing cognitive theories leave substantial computational structure unexplained.

Proceed to stronger theory induction.

---

## 38. TDD requirements

Use:

\[
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
\]

Required new tests:

### Hierarchy

Synthetic shared parameters recover M1.

Synthetic random effects recover M3.

Synthetic ontology-conditioned parameters recover M4.

Held-out task never contributes outcomes to \(B\).

### Flexible ceiling

Teacher model recovered to tolerance.

Flexible task-specific model can reproduce task-specific linear teacher.

Flexible hierarchical model can reproduce hierarchical teacher.

### Active sampling

Coverage component selects underrepresented cells.

Information component selects high-uncertainty cells.

Disagreement component selects known synthetic divergences.

Mixture proportions correct within tolerance.

No previously observed condition selected accidentally.

### Information Sampling audit

Semantic orientation correct.

Feature availability masks correct.

Normalization reproducible.

### Residuals

Negative validation gains cannot be labeled discoveries.

Nested selection/evaluation split enforced.

---

## 39. Suggested repository additions

```text
src/cognitive_discovery/
    hierarchy/
        task_descriptors.py
        random_effects.py
        ontology_conditioned.py
        few_shot.py
        variance_decomposition.py

    sampling/
        candidate_pool.py
        coverage_score.py
        information_score.py
        disagreement_score.py
        active_mixture.py

    audits/
        flexible_ceiling.py
        teacher_distillation.py
        information_sampling.py
        calibration.py

    analysis/
        hierarchical_comparison.py
        parameter_structure.py
        active_efficiency.py
```

---

## 40. Required artifacts

```text
artifacts/discovery_v2/
    audits/
        flexible_ceiling.csv
        teacher_recovery.csv
        information_sampling.csv

    hierarchy/
        model_comparison.csv
        task_parameters.csv
        ontology_coefficients.csv
        variance_decomposition.csv
        loto.csv
        few_shot.csv

    active_sampling/
        candidate_manifest.parquet
        selected_conditions.parquet
        sampling_scores.parquet
        budget_curves.csv

    final_validation/
        predictions.csv
        calibration.csv
        per_task_metrics.csv

    residuals/
        candidate_terms.csv
        heldout_gains.csv

    figures/
    report.md
    run_metadata.json
```

---

## 41. Key figures

I would aim for five central figures.

### Figure 1 — Hierarchical architecture

Diagram:

\[
\text{task ontology }z_t\rightarrow\theta_t\rightarrow f(X,H;\theta_t).
\]

### Figure 2 — Shared form, different parameters

Task-specific coefficient estimates with partial pooling.

### Figure 3 — Model hierarchy comparison

Fully shared vs hierarchical vs ontology-conditioned vs task-specific.

### Figure 4 — Generalization curve

Zero-shot → few-shot adaptation.

### Figure 5 — Active discovery efficiency

Coverage-only versus active-refinement sampling on untouched conditions.

If we get those five working, the paper becomes considerably more coherent.

---

## 42. What we can already say — and what we cannot

I would currently say:

> **The first broad discovery round identifies recent action and outcome history as strong predictors of persistence across diverse task families, while a unitary latent motivational-state account performs poorly. However, exact cross-task parameter transfer is weak, suggesting that any generality lies at the level of computational architecture rather than invariant coefficients.**

I would **not** currently say:

> “Every task uses a different persistence algorithm.”

And I would also not say:

> “Dual history is the universal persistence mechanism.”

The next hierarchy is exactly what separates those possibilities.

The active-sampling idea also fits the scientific-discovery strategy much better now: **Round 1 explored broadly; Round 2 uses the resulting theory uncertainty to allocate samples more intelligently while retaining explicit exploration; then a broad random final sample tests whether the refined theory actually generalizes.**

That is a defensible explore → model → actively refine → independently validate scientific workflow, rather than either pure random sampling forever or immediately chasing whichever hypothesis happens to disagree most.
