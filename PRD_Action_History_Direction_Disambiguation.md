# PRD — Is the Late Action-History Direction a Computational Variable or a Persistence Readout?

**Project:** Mechanistic implementation of LLM persistence  
**Model:** Qwen3.5-4B  
**Purpose:** Resolve the interpretation of the unusually clean layer-30 action-history steering result before proceeding with broader mechanistic work.

The current mechanistic run selected `action_history` at layer 30, measured at the final prompt token. Steering is highly monotonic across tasks (\(\bar\rho=-.964\)), but the preregistered quantitative causal test failed (\(r=.538\)).

The goal is to distinguish three explanations:

\[
\boxed{
\text{action-history computation}
}
\]

versus

\[
\boxed{
\text{late shared stay/switch decision variable}
}
\]

versus

\[
\boxed{
\text{trivial persistence/output geometry}
}
\]

---

## 1. Central scientific question

The existing result establishes that a layer-30 direction which predicts action history can be causally manipulated to alter persistence.

It does **not yet establish** that:

\[
d_{A,30}
\]

is the neural implementation of the behavioral action-history variable \(A_t\).

The key alternative is:

\[
A_t
\rightarrow
\text{downstream persistence state}
\rightarrow
D_t
\]

and the probe has recovered the downstream state because it remains correlated with \(A_t\).

This PRD asks:

> **Does the action-history representation causally influence persistence independently of an already-formed persistence decision?**

---

## 2. Competing hypotheses

### H1 — Upstream action-history implementation

The neural direction genuinely represents the behavioral variable:

\[
A_t.
\]

It contributes causally to the eventual persistence decision.

Predictions:

- action history is strongly represented at an earlier layer;
- action-history information survives controlling for current persistence;
- steering this residualized representation changes later persistence;
- task-specific steering effects track the frozen behavioral action-history coefficients.

### H2 — Late shared stay/switch variable

The late representation is no longer pure action history.

Instead:

\[
A_t,\ O_t,\ X_t,\ldots
\rightarrow
S_t
\rightarrow
D_t
\]

where \(S_t\) is a common downstream stay/switch decision variable.

The action-history probe recovers \(S_t\) because \(A_t\) contributes to it.

Predictions:

- early layers strongly encode \(A_t\);
- late layers increasingly overlap with current persistence;
- late steering remains causal even after removing simple output geometry;
- steering effects are relatively similar across tasks rather than tracking \(\beta_{\tau,A}\);
- early action-history steering may be substantially weaker.

### H3 — Persistence/readout artifact

The selected direction is principally:

\[
d_{A,30}\approx d_{\text{persistence/output}}.
\]

Figure 4 is therefore approximately:

\[
h_{30}
+
\alpha d_{\text{output}}
\rightarrow
D.
\]

Predictions:

- strong alignment between \(d_A\) and persistence/readout directions;
- removing the persistence component eliminates steering;
- earlier action-history directions do not causally influence persistence;
- matched action-history contrasts disappear after current-decision matching;
- random late directions aligned with output gradients show similar linear effects.

---

## 3. Freeze existing results

Do **not** refit or modify the existing layer-30 direction.

Preserve:

```text
original_action_history_L30
```

as the exact direction used in the current Figure 4.

The current intervention position also remains frozen:

```text
final_prompt_token
```

with zero-based transformer block numbering.

All existing behavioral coefficients are frozen before running this follow-up.

---

## 4. Layers

Primary comparison:

```text
Layer 8
Layer 30
```

Rationale:

- Layer 8 is an early/middle layer with very strong existing action-history decoding.
- Layer 30 is the late layer selected by the current mechanistic gate.

This creates the key comparison:

\[
\boxed{\text{early history representation}}
\quad\text{vs}\quad
\boxed{\text{late decision-associated representation}}.
\]

Optional diagnostic layers:

```text
4, 12, 20, 28, 31
```

Do not perform another unrestricted layer search.

Layers 8 and 30 are primary and frozen before outcome inspection.

---

## 5. Variables

For every mechanistic state retain:

### Behavioral action history

\[
A_t
\]

using the exact frozen behavioral definition.

### Current persistence logit

\[
D_t=
\log
\frac{P(\text{continue})}
{P(\text{disengage})}.
\]

### Behavioral controls

Retain:

- outcome history;
- context-relevant outcome history;
- success evidence;
- continuation value;
- disengagement value;
- continuation cost;
- progress;
- task identity;
- response mapping.

No variable definitions may change based on mechanistic results.

---

## 6. Direction A — Original action-history direction

At each primary layer fit:

\[
A_t
=
a_l+w_{A,l}^{\top}h_l+\epsilon.
\]

Define:

\[
d_{A,l}
=
\frac{w_{A,l}}
{\|w_{A,l}\|}.
\]

For L30, retain both:

1. the **exact original direction** used in Figure 4;
2. a clean reproduction fitted under this PRD.

They should agree closely.

A mismatch is an engineering failure and must be resolved before continuing.

---

## 7. Direction B — Persistence-state direction

Fit a separate direction predicting current semantic persistence:

\[
D_t
=
a_l+w_{D,l}^{\top}h_l+\epsilon.
\]

Define:

\[
d_{D,l}
=
\frac{w_{D,l}}
{\|w_{D,l}\|}.
\]

This answers whether:

\[
d_A
\]

and:

\[
d_D
\]

occupy the same neural direction.

---

## 8. Direction C — Direct causal output gradient

The strongest trivial-readout control should not depend on another probe.

For every example calculate:

\[
g_{l,t}
=
\nabla_{h_l}D_t.
\]

Average on training data:

\[
g_l
=
E_t[g_{l,t}]
\]

and normalize:

\[
d_{G,l}
=
\frac{g_l}{\|g_l\|}.
\]

This represents the local neural direction that most directly increases the persistence logit.

This is the cleanest positive control for:

> “What if any direction that points toward CONTINUE gives Figure 4?”

---

## 9. Direction D — Action history geometrically orthogonal to persistence

Construct the control subspace:

\[
C_l=
\operatorname{span}
(d_{D,l},d_{G,l}).
\]

Use QR decomposition to obtain an orthonormal basis \(Q_l\).

Then:

\[
d_{A\perp,l}
=
(I-Q_lQ_l^\top)d_{A,l}
\]

followed by normalization.

This direction contains the component of the original action-history vector that is geometrically independent of the persistence/readout subspace.

---

## 10. Direction E — Statistically residualized action history

Geometric orthogonalization is not enough because neural covariance is anisotropic.

Therefore also construct a stronger computational control.

On **training data only**, predict action history from the current decision and registered nuisance variables:

\[
A_t
=
f(
D_t,
X_t,
\text{task},
\text{mapping}
)
+
\epsilon_t.
\]

Define:

\[
A_t^{resid}=\epsilon_t.
\]

Then fit:

\[
A_t^{resid}
=
a_l+
w_{A|D,l}^{\top}h_l+\epsilon.
\]

Define:

\[
d_{A|D,l}
=
\frac{w_{A|D,l}}
{\|w_{A|D,l}\|}.
\]

This is the critical direction.

It asks:

> **Is there neural information about action history above and beyond the model's current persistence tendency?**

---

## 11. Direction similarity analysis

At L8 and L30 calculate:

\[
\cos(d_A,d_D)
\]

\[
\cos(d_A,d_G)
\]

\[
\cos(d_D,d_G).
\]

Also calculate projection correlations over held-out activations:

\[
r(
d_A^\top h,
d_D^\top h
)
\]

and:

\[
R^2(
d_A^\top h
\sim
D_t
).
\]

Do not rely on cosine alone.

---

## 12. Matched-state representational test

Create two complementary matched datasets.

### Contrast A — Different history, same decision

Find within-task pairs satisfying:

\[
|D_i-D_j|<\delta_D
\]

but:

\[
|A_i-A_j|>\delta_A.
\]

Match as closely as possible on:

- current evidence;
- progress;
- value;
- cost;
- task;
- response mapping.

Question:

> Does the candidate representation still distinguish action history when current persistence is held fixed?

Primary metrics:

\[
\Delta m_A
\]

and classification/sign accuracy.

### Contrast B — Same history, different decision

Find pairs satisfying:

\[
|A_i-A_j|<\delta_A
\]

but:

\[
|D_i-D_j|>\delta_D.
\]

A pure action-history direction should have relatively weak sensitivity here.

A persistence/readout direction should respond strongly.

---

## 13. Representational dissociation score

Define:

\[
RDS_l
=
\frac{
|\Delta m_l|_{\text{history contrast}}
}{
|\Delta m_l|_{\text{decision contrast}}+\epsilon
}.
\]

Compare:

- \(d_A\)
- \(d_{A\perp}\)
- \(d_{A|D}\)
- \(d_D\)
- \(d_G\)

at L8 and L30.

Expected:

### Pure history

\[
RDS\gg1
\]

### Pure decision

\[
RDS\ll1.
\]

---

## 14. Steering design

Run the same intervention grid for every direction:

\[
\alpha
\in
\{-2,-1,-.5,0,.5,1,2\}.
\]

Primary directions:

```text
A_raw
A_orthogonal
A_residualized
persistence_probe
output_gradient
```

at:

```text
L8
L30
```

Use all seven task families.

---

## 15. Critical fairness requirement: matched intervention norm

The current directions have different natural scales.

Therefore the **primary control comparison** must use matched residual-stream displacement:

\[
\|\Delta h\|_2
\]

or matched activation-standard-deviation units.

Do not compare arbitrary coefficient units across directions.

For every direction \(d\):

\[
h_l'
=
h_l+
\alpha\sigma_l d
\]

where \(\sigma_l\) is a frozen projection or activation scale.

This answers:

> At equal neural intervention strength, which directions causally influence persistence?

---

## 16. Computational-unit steering

For action-history directions only, run a secondary calibrated intervention.

Fit on training data:

\[
A_t=a_l+b_lm_l+\epsilon
\]

with:

\[
m_l=d_l^\top h_l.
\]

Then:

\[
\alpha=
\frac{\Delta A}{b_l}.
\]

Use:

\[
\Delta A
\in
\{-2,-1,-.5,0,.5,1,2\}
\]

frozen behavioral SD units.

Do this separately for:

- \(d_A\);
- \(d_{A\perp}\);
- \(d_{A|D}\).

Do **not** calibrate the persistence-gradient direction into action-history units.

---

## 17. Primary steering metrics

For each:

\[
\text{layer}\times
\text{direction}\times
\text{task}
\]

calculate:

### Linear slope

\[
\Delta D
=
a+s\alpha.
\]

### Linearity

\[
R^2_{\text{dose}}.
\]

### Monotonicity

\[
\rho_{\text{Spearman}}.
\]

### Sign consistency

Fraction of tasks sharing the same causal direction.

### Effect magnitude

\[
E|\Delta D|.
\]

All five must be shown.

Do not summarize steering only by absolute effect.

---

## 18. Random-direction null

At each layer generate at least:

\[
N=100
\]

random directions matched for norm.

For each random direction run the full seven-dose intervention on a fixed balanced subset.

Calculate null distributions for:

- slope magnitude;
- \(R^2_{\text{dose}}\);
- Spearman \(|\rho|\);
- seven-task sign consistency;
- mean absolute effect.

The main question is:

> **Is Figure 4's near-linear monotonicity itself unusual at layer 30?**

This directly tests whether late residual-stream perturbations generically produce smooth linear logit changes.

---

## 19. Behavioral-model prediction

The frozen behavioral dual-history model contains task-specific:

\[
\beta_{\tau,A}.
\]

If \(d_A\) genuinely implements \(A_t\), then for calibrated action-history steering:

\[
\Delta D_\tau^{pred}
=
\beta_{\tau,A}\Delta A.
\]

No intervention data may be used to estimate \(\beta_{\tau,A}\).

For every action-history direction compare:

\[
\Delta D^{pred}
\]

against:

\[
\Delta D^{obs}.
\]

Report:

- correlation;
- slope;
- intercept;
- RMSE;
- sign agreement.

---

## 20. Crucial task-heterogeneity test

This is arguably more diagnostic than whether steering merely changes persistence.

Estimate:

\[
s_{\tau,l}
\]

for every task.

Then compare:

\[
s_{\tau,l}
\]

to:

\[
\beta_{\tau,A}.
\]

A true implementation of the behavioral action-history term predicts that task sensitivity should vary with the behavioral coefficients.

A generic stay/switch direction instead predicts:

\[
s_{\tau,l}\approx c
\]

across tasks even where:

\[
\beta_{\tau,A}
\]

differs substantially.

---

## 21. Positive-control expectation

The persistence probe and output-gradient directions should causally alter persistence.

If they do not, the intervention implementation is broken.

However:

> **Strong steering along the persistence/output control is not scientific evidence for a history mechanism.**

It is a technical positive control.

---

## 22. Primary falsification test

The most decisive comparison is:

\[
\boxed{
d_A
\quad
vs
\quad
d_{A\perp}
\quad
vs
\quad
d_{A|D}
}
\]

at:

\[
\boxed{
L8
\quad
vs
\quad
L30
}.
\]

This produces six main action-history conditions:

| Layer | Raw action history | Output-orthogonal | Decision-residualized |
|---|---:|---:|---:|
| L8 | ✓ | ✓ | ✓ |
| L30 | ✓ | ✓ | ✓ |

Plus persistence/output positive controls.

---

## 23. Optional projection patching

Only run after steering.

Construct matched high-\(A\)/low-\(A\) source-target pairs with current persistence approximately matched.

Patch:

\[
h_t'
=
h_t+
d(d^\top h_s-d^\top h_t).
\]

Compare:

- raw \(d_A\);
- residualized \(d_{A|D}\);
- persistence direction;
- random direction.

Question:

> Can isolated action-history information causally transfer a later persistence effect when the initial decision state is matched?

---

## 24. Outcome classification

### Outcome A — Upstream action-history mechanism

Evidence:

- L8 strongly represents history after decision matching;
- L8 \(d_{A|D}\) causally alters persistence;
- effect survives output orthogonalization;
- random directions do not reproduce the effect;
- task-specific steering sensitivities track \(\beta_{\tau,A}\).

Interpretation:

> **Action history is an upstream computational input into persistence.**

This is the strongest result.

### Outcome B — Shared downstream stay/switch state

Evidence:

- L8 cleanly represents action history but has weak causal effect;
- L30 strongly steers persistence;
- L30 remains causal after explicit output/readout orthogonalization;
- L30 task slopes do not closely track \(\beta_{\tau,A}\);
- L30 behaves more similarly across tasks.

Interpretation:

\[
\boxed{
\text{task-specific/history computation}
\rightarrow
\text{shared late stay/switch state}
\rightarrow
\text{response}
}
\]

This would also be scientifically important.

### Outcome C — Output/readout artifact

Evidence:

- L30 strongly aligns with \(d_D\) or \(d_G\);
- L30 raw steering works;
- \(d_{A\perp}\) and \(d_{A|D}\) steering collapses;
- L8 history direction does not causally affect persistence;
- random/output-aligned late directions reproduce the clean dose response.

Interpretation:

> **Figure 4 primarily reflects manipulation of an already-formed persistence decision rather than the computation producing that decision.**

Do not call it an action-history mechanism.

### Outcome D — Mixed representation

Some residual action-history effect survives but much of the original L30 effect disappears.

Interpretation:

> **The selected late direction mixes genuine history information with downstream decision geometry.**

Quantify the surviving fraction rather than forcing a binary conclusion.

---

## 25. Primary gates

### Gate 1 — Representational independence

\(d_{A|D}\) must decode held-out residual action history above random directions.

### Gate 2 — Matched-state specificity

Action-history projections must separate:

\[
A^+\neq A^-
\]

when:

\[
D^+\approx D^-.
\]

### Gate 3 — Causal specificity

\(d_{A|D}\) or \(d_{A\perp}\) must produce dose-dependent persistence effects exceeding the random-direction null.

### Gate 4 — Computational correspondence

For the strong “implementation of \(A_t\)” claim:

\[
\Delta D_\tau^{obs}
\]

must meaningfully track:

\[
\beta_{\tau,A}\Delta A.
\]

If Gate 3 passes but Gate 4 fails, interpret the direction as a downstream causal state rather than direct implementation of the behavioral \(A_t\) term.

---

## 26. Stop rules

Do not proceed into head/MLP localization if:

- only the raw L30 direction steers;
- residualized/orthogonal action-history directions fail;
- random late directions have similar monotonicity;
- the result is explained by the persistence gradient.

If this occurs, record the output/readout explanation and return to upstream computational variables.

---

## 27. Required figures

### Figure 1 — Representation across depth

Show action-history decoding and overlap with current persistence for L8 and L30.

### Figure 2 — Direction geometry

Pairwise:

\[
\cos(d_A,d_D),
\quad
\cos(d_A,d_G),
\quad
\cos(d_D,d_G).
\]

L8 versus L30.

### Figure 3 — Matched-history/decision dissociation

Projection differences for:

- history-change / decision-matched pairs;
- decision-change / history-matched pairs.

### Figure 4 — Critical steering figure

Six action-history curves:

```text
L8 raw
L8 orthogonal
L8 residualized
L30 raw
L30 orthogonal
L30 residualized
```

across all seven tasks.

### Figure 5 — Control dose responses

Compare:

- candidate;
- persistence probe;
- persistence gradient;
- random-direction distribution.

### Figure 6 — Behavioral-model correspondence

Plot:

\[
\beta_{\tau,A}
\]

against observed task steering slopes.

### Figure 7 — Predicted versus observed intervention

\[
\beta_{\tau,A}\Delta A
\]

versus:

\[
\Delta D_\tau^{obs}.
\]

Separate panels for L8 and L30.

---

## 28. Automated report questions

The final report must answer, in order:

1. Does L8 encode action history?
2. Does L30 encode action history?
3. How much does each representation overlap with current persistence?
4. Does action history remain decodable after current persistence is removed?
5. Does the action-history direction distinguish history when the decision is matched?
6. Is the L30 action-history direction aligned with the persistence probe?
7. Is it aligned with the direct persistence gradient?
8. Is Figure 4's monotonicity unusual relative to random directions?
9. Does raw L8 steering affect persistence?
10. Does residualized L8 steering affect persistence?
11. Does raw L30 steering replicate the previous result?
12. Does L30 steering survive output orthogonalization?
13. Does L30 steering survive statistical residualization?
14. Do task-specific causal slopes track frozen \(\beta_{\tau,A}\)?
15. Which of H1, H2, H3, or mixed H4 is best supported?
16. What mechanistic claim is justified?

---

## 29. TDD requirements

Maintain:

\[
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
\]

New tests:

### Direction tests

- \(d_{A\perp}\) has dot product \(<10^{-6}\) with the control subspace.
- QR projection is invariant to basis ordering.
- direction normalization is exact within tolerance.

### Residualization tests

- held-out outcomes never enter residualization.
- target residuals have near-zero training correlation with persistence predictors.
- synthetic action-history signal independent of decision is recovered.

### Gradient tests

- finite-difference perturbation agrees with persistence-logit gradient locally.
- gradient sign convention is tested.

### Steering tests

- zero dose exactly reproduces baseline.
- equal-norm interventions truly have equal norm.
- dose sign reversal reverses projection displacement.
- current response-label mapping is handled semantically.

### Matching tests

- history-matched and decision-matched calipers are enforced.
- no pair crosses task families.
- mapping is balanced.

---

## 30. Artifact strategy

Continue the existing no-large-activation policy.

Suggested output:

```text
artifacts/action_history_disambiguation_v1/
    directions/
        L8_action_raw.safetensors
        L8_action_orthogonal.safetensors
        L8_action_residualized.safetensors
        L8_persistence.safetensors
        L8_gradient.safetensors

        L30_action_raw.safetensors
        L30_action_orthogonal.safetensors
        L30_action_residualized.safetensors
        L30_persistence.safetensors
        L30_gradient.safetensors

    representation/
        direction_similarity.csv
        matched_state_results.csv
        residual_decoding.csv

    steering/
        dose_response.csv
        random_null.csv
        behavioral_correspondence.csv

    patching/
        matched_projection_patch.csv

    figures/
    gates.json
    report.md
    run_metadata.json
```

No full activations.

---

## 31. Success criterion

The strongest possible result is:

\[
\boxed{
\text{history information at an upstream layer}
\rightarrow
\text{causal persistence change}
}
\]

that survives controlling for the already-formed persistence decision **and** whose task-specific causal magnitude is predicted by the independently fitted behavioral action-history coefficients.

That would connect:

\[
\text{behavioral computational model}
\rightarrow
\text{internal representation}
\rightarrow
\text{causal implementation}.
\]

If instead only L30 survives and its causal effects are largely task-invariant, the more appropriate conclusion is:

> **Action history is represented upstream, but the clean cross-task steering direction corresponds to a later shared stay/switch decision state rather than the behavioral action-history variable itself.**

That outcome would still be valuable—it would tell us where the common cross-task mechanism enters the computation.
