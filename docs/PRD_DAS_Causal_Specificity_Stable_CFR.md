# PRD — Validate DAS Causal Specificity with Stable Counterfactual Recovery

**Project:** Counterfactual Causal Mechanistic Discovery for LLM Persistence  
**Model:** Qwen3.5-4B  
**Stage:** Level 5 causal specificity  
**Purpose:** Determine whether the DAS subspaces found in the current mechanistic run specifically implement outcome-history computations, rather than merely providing generic control over persistence.

The current run reaches **Evidence Level 4**: learned DAS subspaces reproduce frozen behavioral counterfactuals on held-out examples and appear to use shared neural coordinates across tasks. However, **Level 5 causal specificity and necessity fail**, so neither `outcome_history` nor `contextual_outcome_history` can yet be identified as the implemented computational variable.

A major technical issue is that the current **mean per-example CFR metric is numerically unstable** when the behavioral model predicts near-zero counterfactual effects. This PRD first repairs that metric, then reruns the causal-specificity analysis using matched comparisons.

---

## 1. Scientific question

The central question is no longer:

> Can we learn a neural subspace whose intervention changes persistence appropriately?

The existing Level-4 result says yes.

The new question is:

> **Does the learned subspace specifically implement the computational variable named by the behavioral model?**

For example, if a subspace is labeled:

\[
S_{O^\*},
\]

we need evidence that interchanging it behaves like:

\[
do(O^\*)
\]

rather than like:

\[
do(\text{persistence}),
\]

\[
do(\text{generic value}),
\]

or a generic learned controller.

---

## 2. Current evidence state

Freeze the current conclusions:

### Passed

- behavioral entry criteria;
- coarse causal localization;
- held-out causal representation / Level 4;
- shared-coordinate neural hypothesis currently favored.

### Failed / unresolved

- causal specificity;
- necessity;
- circuit implementation;
- discrimination between `outcome_history` and `contextual_outcome_history`.

Do not revise that conclusion until this PRD completes.

---

## 3. Core hypotheses

### H1 — Specific outcome-history implementation

There exists a neural subspace:

\[
S_O
\]

such that:

\[
do(S_O:O_b\leftarrow O_s)
\]

reproduces the behavioral counterfactual:

\[
do(O_b\leftarrow O_s).
\]

It should outperform unrelated-variable and generic-decision controls.

### H2 — Specific contextual-history implementation

There exists a neural subspace:

\[
S_{O^\*}
\]

such that:

\[
do(S_{O^\*}:O_b^\*\leftarrow O_s^\*)
\]

reproduces:

\[
do(O_b^\*\leftarrow O_s^\*).
\]

It should be especially effective on context-conflict examples where:

\[
O
\]

and:

\[
O^\*
\]

make different predictions.

### H3 — Shared downstream persistence-control subspace

DAS has learned a neural representation corresponding to a later quantity:

\[
S_D
\]

that controls persistence regardless of which high-level variable generated the counterfactual target.

Prediction:

\[
S_O
\approx
S_{O^\*}
\approx
S_D
\]

in causal function.

Cross-variable interventions will work almost as well as matched-variable interventions.

### H4 — Generic intervention/controller solution

The optimization procedure found a subspace that can manipulate persistence but does not correspond to a naturally used cognitive variable.

Prediction:

- shuffled or unrelated counterfactual targets perform similarly;
- random/control subspaces approach candidate performance;
- specificity vanishes on strict held-out conditions;
- intervention may be sufficient but not necessary.

---

## 4. First requirement — replace unstable CFR

The current per-example metric is:

\[
CFR_i
=
1-
\frac{
(\Delta D_i^{neural}-\Delta D_i^{CF})^2
}{
(\Delta D_i^{base}-\Delta D_i^{CF})^2+\epsilon
}.
\]

This is unstable whenever:

\[
\Delta D_i^{CF}\approx0.
\]

Tiny denominators can produce values on the order of:

\[
-10^6
\]

or:

\[
-10^7,
\]

which then dominate the mean.

Do **not** use mean per-example CFR as the primary statistic.

---

## 5. Primary CFR metric

Define **Global Counterfactual Recovery**:

\[
\boxed{
CFR_G
=
1-
\frac{
\sum_i
(\Delta D_i^{neural}-\Delta D_i^{CF})^2
}{
\sum_i
(\Delta D_i^{base}-\Delta D_i^{CF})^2
}
}
\]

If baseline change is zero:

\[
CFR_G
=
1-
\frac{
\sum_i
(\Delta D_i^{neural}-\Delta D_i^{CF})^2
}{
\sum_i
(\Delta D_i^{CF})^2
}.
\]

Interpretation:

### Perfect recovery

\[
CFR_G=1.
\]

### Equivalent to doing nothing

\[
CFR_G=0.
\]

### Worse than baseline

\[
CFR_G<0.
\]

This is analogous to an \(R^2\)-style aggregate error reduction.

---

## 6. Required companion metrics

Never report CFR alone.

For every counterfactual comparison report:

### Correlation

\[
r(
\Delta D^{CF},
\Delta D^{neural}
).
\]

### Calibration slope

Fit:

\[
\Delta D^{neural}
=
a+b\Delta D^{CF}.
\]

Report:

\[
b.
\]

### Intercept

\[
a.
\]

### RMSE

\[
RMSE
=
\sqrt{
\frac1N
\sum_i
(
\Delta D_i^{neural}
-
\Delta D_i^{CF}
)^2
}.
\]

### Sign accuracy

\[
P[
\operatorname{sign}(\Delta D^{neural})
=
\operatorname{sign}(\Delta D^{CF})
].
\]

These five metrics jointly define counterfactual performance.

---

## 7. Secondary diagnostic: thresholded CFR

For descriptive purposes only, calculate per-example CFR after excluding examples with negligible predicted counterfactual effects:

\[
|\Delta D^{CF}|<\delta.
\]

Set \(\delta\) before examining results.

Recommended:

\[
\delta=.10
\]

standardized persistence-logit units, or an equivalent preregistered value.

Report:

- fraction retained;
- median CFR;
- IQR.

Do not use this as the primary gate.

---

## 8. Recompute Level 3 localization

Rerun existing whole-state patching summaries using:

\[
CFR_G
\]

rather than mean per-example CFR.

For every:

\[
\text{layer}
\times
\text{variable}
\times
\text{behavioral theory}
\]

report:

- global CFR;
- \(r\);
- slope;
- RMSE;
- sign accuracy.

The purpose is to determine whether the apparently catastrophic negative Level-3 curves were purely metric artifacts.

---

## 9. Freeze DAS models

Do **not** retrain the current DAS subspaces for the primary reanalysis.

Freeze:

- layer;
- rank;
- learned rotation;
- learned subspace;
- behavioral model;
- source/base pair definitions;
- train/validation/test split.

First determine whether the current Level-4 result survives corrected evaluation.

Retraining is allowed only in a clearly labeled secondary analysis.

---

## 10. Re-evaluate Level 4

For every frozen candidate DAS model, evaluate on untouched test examples:

\[
\Delta D^{CF}
\]

versus:

\[
\Delta D^{DAS}.
\]

Primary gate:

\[
CFR_G>0
\]

and bootstrap CI excluding zero.

Also require:

\[
r>0
\]

with bootstrap support.

Do not require slope exactly 1.

Report calibration.

---

## 11. Bootstrap uncertainty

Use bootstrap resampling at the **semantic-condition / source-base-pair level**, not individual duplicated label mappings.

Recommended:

\[
B=2000.
\]

Produce 95% CIs for:

- \(CFR_G\);
- \(r\);
- slope;
- RMSE;
- sign accuracy.

Respect task clustering where relevant.

---

## 12. Level-5 specificity overview

For each target variable \(X\), compare:

\[
S_X
\]

against the following controls using **identical test examples** and identical metrics.

Controls:

1. matched random subspace;
2. shuffled source;
3. shuffled counterfactual target;
4. persistence-state subspace;
5. persistence-output subspace;
6. generic-value subspace;
7. task-ID subspace;
8. response-mapping subspace;
9. unrelated-variable DAS subspace.

No control may be evaluated on an easier/different pair set.

---

## 13. Random-subspace null

For each selected:

\[
(\text{layer},k)
\]

generate at least:

\[
N=500
\]

random \(k\)-dimensional orthonormal subspaces.

Use exactly the same source/base interchange operation.

For each calculate:

\[
CFR_G.
\]

Primary random-null statistic:

\[
p_{rand}
=
\frac{
1+\sum_j I(CFR_{G,j}^{rand}\ge CFR_G^{candidate})
}{
N+1
}.
\]

Gate:

\[
p_{rand}<.05.
\]

Also compare candidate correlation and RMSE against null distributions.

---

## 14. Shuffled-source control

Preserve the base state but randomly assign a source state from the same:

- task;
- response mapping;
- approximate current-state bin.

Break the relationship:

\[
X_s
\leftrightarrow
X_b.
\]

The learned subspace should lose counterfactual recovery.

Require:

\[
CFR_G^{candidate}
>
CFR_G^{shuffled-source}.
\]

Use paired bootstrap difference.

---

## 15. Shuffled-target control

Keep neural source/base pairs unchanged but permute:

\[
\Delta D^{CF}
\]

within task.

This tests whether the DAS model merely produces structured changes that happen to align with the overall persistence distribution.

Expected:

\[
CFR_G^{shuffled-target}\le0.
\]

---

## 16. Direct persistence control

Learn or use a neural subspace optimized directly for:

\[
D_t.
\]

This is a positive control for generic persistence manipulation.

Ask two separate questions.

### Control effectiveness

Can:

\[
S_D
\]

strongly manipulate persistence?

It probably should.

### Computational specificity

Does:

\[
S_D
\]

reproduce the detailed source-specific counterfactual predictions generated by:

\[
O
\]

or:

\[
O^\*?
\]

If yes, then successful DAS counterfactual recovery may reflect manipulation of a generic downstream decision state rather than the upstream computational variable.

---

## 17. Cross-variable specificity matrix

This is the most important new analysis.

Learn/freeze:

\[
S_O
\]

from raw outcome-history counterfactuals.

Learn/freeze:

\[
S_{O^\*}
\]

from contextual-history counterfactuals.

Optionally:

\[
S_A
\]

for action history.

Evaluate every subspace against every counterfactual target.

Construct matrix:

| Neural intervention | \(do(O)\) targets | \(do(O^\*)\) targets | \(do(A)\) targets |
|---|---:|---:|---:|
| \(S_O\) | ? | ? | ? |
| \(S_{O^\*}\) | ? | ? | ? |
| \(S_A\) | ? | ? | ? |
| \(S_D\) | ? | ? | ? |

Cells contain:

\[
CFR_G.
\]

---

## 18. Variable-specificity index

For target variable \(X\):

\[
VSI_X
=
CFR_G(S_X\rightarrow X)
-
\max_{Z\ne X}
CFR_G(S_X\rightarrow Z).
\]

Strong specificity requires:

\[
VSI_X>0
\]

with bootstrap CI excluding zero.

This is stricter than simply beating random subspaces.

---

## 19. Subspace interchangeability

Compute principal-angle / canonical-correlation similarity between learned subspaces:

\[
S_O,
\quad
S_{O^\*},
\quad
S_D.
\]

Report:

- principal angles;
- projection overlap;
- canonical correlations.

But treat geometry as secondary.

Two subspaces can have low geometric overlap and still implement the same functional quantity.

Functional cross-intervention results are primary.

---

## 20. Context-conflict test

The original behavioral theories remain unresolved.

Therefore construct a dedicated test set where raw and contextual history disagree.

Require conditions satisfying:

\[
|O_s-O_b|
\]

and:

\[
|O_s^\*-O_b^\*|
\]

that imply materially different behavioral counterfactuals.

Prefer:

\[
|\Delta D_O^{CF}
-
\Delta D_{O^\*}^{CF}|
\]

in the top quartile or preregistered upper range.

---

## 21. Theory-discriminating counterfactual test

For every context-conflict pair calculate:

\[
\Delta D_O^{CF}
\]

and:

\[
\Delta D_{O^\*}^{CF}.
\]

After neural intervention obtain:

\[
\Delta D^{neural}.
\]

Compare:

\[
MSE_O
=
(\Delta D^{neural}-\Delta D_O^{CF})^2
\]

versus:

\[
MSE_{O^\*}
=
(\Delta D^{neural}-\Delta D_{O^\*}^{CF})^2.
\]

Primary discriminating statistic:

\[
\Delta MSE
=
MSE_O-MSE_{O^\*}.
\]

Positive values favor contextual history.

Use paired bootstrap CIs.

---

## 22. Do not allow DAS to define the theory

For theory discrimination, the **same frozen neural intervention** must be scored against both theories.

Do not train:

\[
S_O
\]

against \(M_O\) and:

\[
S_{O^\*}
\]

against \(M_{O^\*}\)

and then compare their training objectives alone.

That comparison confounds theory with optimization flexibility.

Primary theory discrimination requires:

\[
\boxed{\text{same neural result, competing frozen theoretical predictions}.}
\]

---

## 23. Held-out task test

Where sample size permits:

\[
\text{train DAS on }N-1\text{ tasks}
\]

and evaluate on the held-out task with **no target-task neural refitting**.

Report:

\[
CFR_G^{LOTO}.
\]

Test separately:

### Shared-coordinate hypothesis

Same:

\[
S_X
\]

must transfer.

### Task-specific-coordinate hypothesis

Allow task-specific:

\[
S_{X,\tau}
\]

but freeze the same high-level counterfactual semantics.

This separates:

\[
\text{shared neural coordinates}
\]

from:

\[
\text{shared causal computation}.
\]

---

## 24. Information Sampling negative control

Retain Information Sampling as a quantitative negative control if the frozen behavioral model predicts near-zero history sensitivity.

For a valid history representation:

\[
do(O^\*)
\]

should successfully alter the neural representation while producing:

\[
\Delta D\approx0
\]

in Information Sampling.

This is stronger than requiring large steering everywhere.

---

## 25. Necessity test

Only after specificity is reassessed.

For candidate subspace \(S_X\), remove or neutralize its information.

Preferred intervention:

\[
h'
=
h-P_Sh+P_Sh_{\text{neutral}}.
\]

Neutral state can be:

- matched population mean;
- task-conditioned mean;
- source from matched \(X\approx0\) condition.

Do not use zeroing as the only necessity test.

---

## 26. Necessity prediction

If \(S_X\) implements \(X\), removing it should selectively reduce behavioral sensitivity to \(X\).

Estimate:

\[
D
=
\beta_X X+\cdots
\]

before intervention and:

\[
D'
=
\beta_X'X+\cdots
\]

after neutralization.

Define:

\[
NS_X
=
1-\frac{|\beta_X'|}{|\beta_X|}.
\]

Positive necessity means the model becomes less behaviorally sensitive to \(X\).

---

## 27. Necessity specificity

Ablating \(S_O\) should primarily reduce:

\[
\beta_O
\]

rather than uniformly flattening all behavioral effects.

Measure changes in coefficients for:

- current value;
- progress;
- disengagement value;
- action history;
- outcome history;
- contextual history.

A generic reduction in all sensitivity indicates nonspecific damage.

---

## 28. Level-5 pass criteria

A computational variable \(X\) reaches Level 5 only if all primary conditions hold:

### A. Counterfactual recovery

\[
CFR_G>0
\]

with 95% CI excluding zero.

### B. Random-subspace specificity

Candidate exceeds 95th percentile random null.

### C. Shuffled-source specificity

Candidate exceeds shuffled-source control.

### D. Shuffled-target specificity

True counterfactual mapping exceeds shuffled mapping.

### E. Generic-decision specificity

Candidate provides information beyond persistence/output control.

### F. Cross-variable specificity

\[
VSI_X>0
\]

with bootstrap support.

All must pass.

---

## 29. Necessity is a separate gate

Do not require necessity for Level 5 if keeping the existing evidence hierarchy.

Instead:

### Level 5A

Causal specificity.

### Level 5B

Necessity.

Only use strong wording like:

> “implements computational variable \(X\)”

when both are supported.

---

## 30. Interpretation outcomes

### Outcome A — Specific contextual-history implementation

Evidence:

\[
S_{O^\*}
\]

selectively reproduces contextual-history counterfactuals, beats all controls, performs better than raw history on conflict trials, and passes necessity.

Conclusion:

> **A shared neural subspace causally implements context-sensitive outcome-history integration.**

### Outcome B — Specific raw-history implementation

\[
S_O
\]

wins equivalent tests.

Conclusion:

> **Recent outcome history is causally implemented directly without evidence that the neural variable performs the proposed contextual transformation.**

### Outcome C — Shared downstream causal state

Both:

\[
S_O
\]

and:

\[
S_{O^\*}
\]

reproduce both target families similarly.

Persistence-control subspace does similarly.

Conclusion:

> **Different behavioral history variables converge onto a common downstream persistence-control representation.**

This would explain behavioral observational equivalence.

### Outcome D — DAS controller without variable identity

Candidate beats random but not:

- shuffled target;
- persistence control;
- unrelated-variable counterfactuals.

Conclusion:

> **DAS identified an effective persistence controller but not a specific implementation of the proposed cognitive variable.**

### Outcome E — Level-4 result disappears

Corrected CFR shows weak held-out counterfactual recovery.

Conclusion:

> **The previous Level-4 result was partly an artifact of unstable evaluation.**

Return to causal localization/search.

---

## 31. Required figures

### Figure 1 — Corrected coarse localization

\[
CFR_G
\]

across layers for each target/theory.

No \(10^7\)-scale axis.

### Figure 2 — Held-out counterfactual equivalence

\[
\Delta D^{CF}
\]

versus:

\[
\Delta D^{DAS}.
\]

Show identity line and fitted regression.

Annotate:

- \(r\);
- slope;
- \(CFR_G\);
- RMSE.

### Figure 3 — Random-subspace null

Histogram of random:

\[
CFR_G
\]

with candidate marked explicitly.

### Figure 4 — Specificity control comparison

For each control report:

\[
CFR_G
\]

with bootstrap CI.

Avoid plotting unstable mean per-example CFR.

### Figure 5 — Cross-variable specificity matrix

Heatmap:

\[
S_X
\times
do(Z)
\]

with global CFR.

### Figure 6 — Context-conflict discrimination

Plot neural effect against:

\[
\Delta D_O^{CF}
\]

and:

\[
\Delta D_{O^\*}^{CF}.
\]

### Figure 7 — Necessity

Behavioral coefficient before versus after subspace neutralization.

---

## 32. Automated report questions

The report must answer:

1. Does corrected global CFR reproduce the previous Level-4 conclusion?
2. Which layer/rank provides best untouched-test recovery?
3. Does candidate DAS beat matched random subspaces?
4. Does it beat shuffled sources?
5. Does it beat shuffled targets?
6. Does it outperform a direct persistence-control representation?
7. Does \(S_O\) specifically recover \(do(O)\)?
8. Does \(S_{O^\*}\) specifically recover \(do(O^\*)\)?
9. Are \(S_O\) and \(S_{O^\*}\) functionally interchangeable?
10. Which theory better predicts context-conflict interventions?
11. Does causal recovery transfer to held-out tasks?
12. Does neutralizing the subspace selectively reduce sensitivity to its proposed variable?
13. What is the highest justified evidence level?
14. Which mechanistic claim is justified?
15. Which claims remain unsupported?

---

## 33. TDD requirements

Maintain:

\[
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
\]

### CFR tests

Synthetic perfect recovery:

\[
CFR_G=1.
\]

No-op intervention:

\[
CFR_G=0
\]

when baseline effect is zero.

Wrong-way intervention must produce:

\[
CFR_G<0.
\]

Near-zero individual counterfactual effects must **not** cause global CFR explosion.

### Control tests

- shuffled targets genuinely change mapping;
- shuffled sources preserve task/mapping strata;
- random subspaces are orthonormal;
- candidate and controls use identical test rows.

### Cross-variable tests

- \(S_O\) and \(S_{O^\*}\) remain distinct artifact IDs;
- counterfactual target labels cannot leak into neural intervention;
- cross-variable evaluation performs no retraining.

### Necessity tests

- neutralization alters only target subspace;
- orthogonal residual remains unchanged;
- control coefficient estimates reproduce baseline before intervention.

---

## 34. Artifact structure

```text
artifacts/causal_specificity_v2/
    frozen_models/
        behavioral_hash.json
        das_manifest.json

    corrected_metrics/
        global_cfr.csv
        calibration.csv
        bootstrap_intervals.csv

    controls/
        random_subspaces.parquet
        shuffled_source.csv
        shuffled_target.csv
        persistence_control.csv
        generic_value_control.csv

    cross_variable/
        specificity_matrix.csv
        subspace_geometry.csv
        context_conflict.csv

    necessity/
        neutralization_results.parquet
        coefficient_changes.csv

    figures/
    gates.json
    report.md
    run_metadata.json
```

Continue the no-full-activation-bank policy.

---

## 35. Stop rule

Do **not** proceed to attention-head, MLP, or path-level circuit localization until:

\[
\boxed{\text{Level 5 specificity is resolved}.}
\]

If DAS reproduces counterfactuals but does not identify a unique high-level variable, that is the scientific result.

Do not force circuit interpretation around an ambiguous representation.

---

## 36. Scientific success criterion

The strongest result is:

\[
\boxed{
do(X)
\text{ in the cognitive model}
\approx
do(S_X)
\text{ in the network}
}
\]

such that this equivalence:

- holds on untouched examples;
- transfers across tasks where predicted;
- beats matched random and shuffled controls;
- is specific to \(X\);
- and selectively disappears when \(S_X\) is neutralized.

That would justify moving from:

> “DAS found a subspace that controls persistence”

to:

> **“We identified a neural implementation of a computational variable independently discovered from behavior.”**

That is the threshold to clear before circuit discovery.
