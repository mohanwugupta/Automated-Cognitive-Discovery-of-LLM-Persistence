# PRD — Mechanistic Implementation of Context-Sensitive Outcome-History Integration

**Project:** Automated Cognitive Discovery of LLM Persistence  
**Stage:** Mechanistic interpretation / causal implementation  
**Primary model:** Qwen3.5-4B  
**Status:** Exploratory mechanistic follow-up  
**Primary objective:** Identify and causally manipulate internal representations of the computational variables discovered behaviorally, while keeping storage and repository size minimal.

---

## 1. Scientific objective

The behavioral discovery pipeline has narrowed persistence to a family of models in which current task evidence is combined with recent history, with substantial task-specific parameterization.

The strongest common computational ingredient is **outcome-history integration**, with an additional unresolved question about whether recent history is used directly or weighted according to **contextual relevance**.

The mechanistic phase asks:

> **How does the model internally represent and use recent outcome history when deciding whether to continue or disengage?**

And, more specifically:

> **Does the model transform raw recent outcome history into a context-relevant history representation before that information influences persistence?**

The desired scientific chain is:

\[
\boxed{
\text{Behavior}
\rightarrow
\text{Computational model}
\rightarrow
\text{Internal representation}
\rightarrow
\text{Causal intervention}
}
\]

The strongest possible endpoint is a quantitatively predictive intervention:

\[
\Delta D_\tau^{\text{predicted}}
\approx
\Delta D_\tau^{\text{observed}}
\]

where the predicted task-specific steering effect comes from the independently fitted behavioral model.

---

## 2. Scope

This PRD should implement:

1. frozen computational targets;
2. a small matched mechanistic dataset;
3. streaming all-layer representational analysis;
4. cross-task and specificity validation;
5. quantitative steering;
6. direction-specific activation patching / causal mediation;
7. optional contextual-history discrimination.

This PRD should **not initially** implement:

- full activation-bank collection;
- SAE training;
- broad circuit discovery;
- head-by-head localization;
- path patching;
- DAS;
- neuron-level interpretability;
- large `.pt` artifact dumps.

Those become follow-ups only if the causal tests succeed.

---

## 3. Core computational targets

Freeze all computational targets before collecting activations.

### Target A — Raw outcome history

Let:

\[
O_t
\]

represent the recent outcome-history quantity used by the direct-history model.

This should be computed entirely from observed prior outcomes according to the frozen behavioral-model specification.

### Target B — Action history

Let:

\[
A_t
\]

represent the corresponding recent-action / perseveration quantity.

This is primarily a control and secondary mechanistic target.

### Target C — Context-relevant outcome history

Let:

\[
O_t^\*
\]

represent outcome history after weighting/selecting history according to the current inferred context.

Conceptually:

\[
O_t^\*
=
\sum_{i<t}
w_{it}^{context}r_i.
\]

The exact definition must come from the frozen latent-context behavioral model.

Do not redefine this quantity based on neural results.

### Target D — Current-state evidence

Retain frozen current-state variables as nuisance/control quantities:

- success evidence;
- progress evidence;
- continuation cost;
- disengagement value;
- continuation value where available.

---

## 4. Behavioral equation to preserve

The mechanistic tests should use the frozen behavioral model.

A simplified target form is:

\[
D_t^{(\tau)}
=
\alpha_\tau
+
\beta_{\tau,X}^{\top}X_t
+
\beta_{\tau,A}A_t
+
\beta_{\tau,O^\*}O_t^\*.
\]

Here:

\[
D_t^{(\tau)}
=
\log
\frac{P(\text{continue})}
{P(\text{disengage})}.
\]

Task-specific coefficients must come from the behavioral phase and must not be re-estimated from activation data.

---

## 5. Primary mechanistic hypotheses

### H1 — Outcome-history representation

The transformer contains an internal quantity that tracks:

\[
O_t
\]

or:

\[
O_t^\*.
\]

Prediction:

A linear or low-dimensional representation should decode outcome history across held-out conditions and tasks.

### H2 — Context transformation

The network transforms:

\[
O_t
\]

into:

\[
O_t^\*
\]

when contextual information indicates that some recent observations are no longer relevant.

Prediction:

Earlier layers may reflect raw recent history more strongly, while later layers increasingly reflect context-relevant history.

### H3 — Causal outcome-history integration

Perturbing the identified representation should systematically change persistence.

Prediction:

\[
\Delta O^\*>0
\Rightarrow
\Delta D>0
\]

for tasks with positive behavioral outcome-history coefficients.

### H4 — Quantitative task-specific causal effects

The magnitude of the causal effect should be predicted by:

\[
\beta_{\tau,O^\*}.
\]

For a calibrated intervention:

\[
\boxed{
\Delta D_\tau^{pred}
=
\beta_{\tau,O^\*}
\Delta O^\*
}.
\]

This is the strongest mechanistic test in the project.

---

## 6. Storage philosophy

The project must operate under:

\[
\boxed{\text{stream activations, save summaries}}
\]

rather than:

\[
\boxed{\text{save full activation banks}}.
\]

No full hidden-state tensors should be committed to Git.

Prefer not to save them at all.

---

## 7. Repository size constraints

Git-tracked artifacts should generally remain under:

```text
10 MB per file
```

unless explicitly allowlisted.

Add CI/pre-commit protection that rejects large tracked files.

Recommended:

```text
scripts/check_artifact_sizes.py
```

Failure condition:

```text
tracked file > 10 MB
```

unless included in:

```text
config/artifact_allowlist.txt
```

---

## 8. Scratch-storage architecture

All temporary activations go to external scratch storage.

Environment variable:

```bash
MECH_SCRATCH=/scratch/$USER/persistence_mech
```

Expected structure:

```text
$MECH_SCRATCH/
    temporary_activations/
    temporary_batches/
    model_cache/
```

These paths must be excluded from Git.

Temporary tensors should be deleted automatically after sufficient statistics are extracted.

---

## 9. Mechanistic dataset

Do **not** use the entire behavioral dataset initially.

Construct a compact matched dataset specifically for mechanistic identification.

Target:

\[
1{,}000-2{,}000
\]

matched semantic contrasts total.

Use approximately:

\[
3-4
\]

task families for discovery.

Reserve remaining task families for strict transfer.

Preferred discovery tasks:

- Bandit;
- Foraging;
- Solvability;
- Debugging.

Potential held-out tests:

- Waiting;
- Effort;
- Information Sampling.

---

## 10. Contrast family A — Outcome-history manipulation

Construct matched pairs:

\[
x^+
\]

versus:

\[
x^-,
\]

where:

\[
O_t^+>O_t^-
\]

while holding current decision state constant.

Match:

- success evidence;
- progress;
- continuation value;
- continuation cost;
- disengagement value;
- task/environment state;
- response labels;
- step number;
- context where applicable.

Only prior outcome history should change.

---

## 11. Contrast family B — Context-relevant history

Use contextual A→B→A conditions.

Hold the raw sequence of outcomes constant while manipulating which historical outcomes belong to the current context.

Construct contrasts where:

\[
O_t
\]

is identical or closely matched but:

\[
O_t^\*
\]

differs.

This is the primary contrast for distinguishing direct history from context-sensitive history.

---

## 12. Contrast family C — Raw-history control

Construct examples where raw recency changes but context-relevant history does not.

This creates the reverse dissociation:

\[
\Delta O_t\neq0
\]

while:

\[
\Delta O_t^\*\approx0.
\]

Together with Contrast B, this creates a behavioral/neural double dissociation.

---

## 13. Contrast family D — Current-value controls

Manipulate:

- success evidence;
- continuation value;
- disengagement value;

while matching outcome history.

Purpose:

Ensure candidate directions do not simply encode generic value or persistence.

---

## 14. Activation position

Use one fixed pre-generation decision position.

Primary:

> hidden state at the final prompt token immediately before the model generates the response token.

This avoids output-token contamination as much as possible while keeping the position identical across tasks.

Do not search over arbitrary token positions initially.

Secondary token-position analyses only if needed.

---

## 15. Layer coverage

Initial scan:

```text
all transformer layers
```

but process activations sequentially/streamingly.

Do not store:

\[
N\times L\times d
\]

hidden-state tensors.

---

## 16. Streaming representation scan

For each layer \(l\):

1. run inference on one batch;
2. extract only the designated hidden state;
3. compute necessary statistics;
4. update streaming sufficient statistics;
5. discard full hidden vectors.

For paired contrasts calculate:

\[
\Delta h_l
=
h_l^+
-
h_l^-.
\]

Maintain only:

- running mean \(\Delta h_l\);
- covariance / regression sufficient statistics if required;
- projection scores;
- sample counts.

---

## 17. Mean-difference directions

For each computational variable:

\[
d_l^{O}
=
\frac{
E[\Delta h_l^{O}]
}{
\|E[\Delta h_l^{O}]\|
}.
\]

Similarly:

\[
d_l^{O^\*}
\]

and:

\[
d_l^{A}.
\]

These provide simple, interpretable first-pass directions.

---

## 18. Linear probe alternative

Also fit ridge models:

\[
\hat O_t
=
w_l^\top h_l+b
\]

and:

\[
\hat O_t^\*
=
v_l^\top h_l+c.
\]

Fit one layer at a time.

Use train-only normalization.

Raw hidden states may be cached temporarily during that layer's fit and deleted immediately afterward.

---

## 19. Low-dimensional extension

Only if 1D decoding is inadequate, test ranks:

```text
1
2
4
```

Do not immediately search higher ranks.

Use frozen rank based on validation.

---

## 20. Primary representation metrics

For every layer/target report:

- held-out \(R^2\);
- Pearson \(r\);
- MSE;
- contrast sign accuracy;
- task-macro performance;
- LOTO-task performance.

Primary criterion should emphasize continuous prediction rather than sign accuracy.

---

## 21. Cross-task representation validation

Use strict LOTO:

\[
\text{train on }N-1\text{ tasks}
\rightarrow
\text{test task }N.
\]

No target-task:

- normalization;
- rotation;
- calibration;
- refitting.

This is important because previous persistence directions failed precisely under strict transfer.

---

## 22. Representation specificity controls

Candidate computational directions must be tested against:

### Persistence logit

Is it merely final decision geometry?

### Generic value

Does it simply encode good/bad prospects?

### Current success evidence

Is it just current-state value?

### Terminality

Is it merely STOP versus CONTINUE geometry?

### Arbitrary binary choice

Does it encode generic binary response structure?

### Task identity

Can task ID explain the direction?

### Response labels

Verify X/Y counterbalancing does not determine projections.

---

## 23. Direct-history versus contextual-history comparison

At every layer compare:

\[
R^2(O_t)
\]

versus:

\[
R^2(O_t^\*).
\]

Also evaluate residual prediction:

\[
O_t^\*
\sim
O_t
\]

versus:

\[
O_t^\*
\sim
O_t + h_l.
\]

Question:

> Does the neural state contain information about contextual history relevance beyond raw recent outcome history?

---

## 24. Contextual transformation index

Define:

\[
CTI_l
=
R^2_l(O^\*)
-
R^2_l(O).
\]

This should be interpreted carefully because target variances can differ.

Also report matched partial-\(R^2\):

\[
\Delta R^2_l
=
R^2(O^\*\mid O,h_l)
-
R^2(O^\*\mid O).
\]

This is the stronger measure.

---

## 25. Candidate layer selection

Select at most:

\[
2-4
\]

candidate layers.

Selection criteria:

1. strong held-out decoding;
2. cross-task transfer;
3. specificity;
4. contextual discrimination where relevant;
5. not excessively output-adjacent if an earlier comparable layer exists.

Avoid choosing purely by maximum \(R^2\).

---

## 26. Random-direction controls

For each selected layer generate:

\[
N=100
\]

matched random directions or subspaces.

Compare:

- computational-variable decoding;
- steering effect;
- specificity.

A candidate must exceed appropriate random baselines.

---

## 27. Direction calibration

Before steering, calibrate neural projection onto the behavioral computational variable.

Let:

\[
m_l=d_l^\top h_l.
\]

Fit on training data:

\[
O_t^\*
=
a_l+b_lm_l+\epsilon.
\]

Freeze:

\[
a_l,b_l.
\]

The calibration must be evaluated on held-out data.

---

## 28. Steering intervention

At selected layer \(l\):

\[
h_l'
=
h_l+\alpha d_l.
\]

Use a range of intervention magnitudes expressed in **computational units**, not arbitrary activation units.

For example choose desired:

\[
\Delta O^\*
\in
\{-2,-1,-.5,.5,1,2\}
\]

standardized units.

Then:

\[
\alpha
=
\frac{\Delta O^\*}{b_l}.
\]

This creates comparable interventions across layers.

---

## 29. Primary steering prediction

The behavioral model independently predicts:

\[
\Delta D_\tau^{pred}
=
\beta_{\tau,O^\*}
\Delta O^\*.
\]

No steering data may be used to estimate:

\[
\beta_{\tau,O^\*}.
\]

---

## 30. Steering evaluation

For every task and intervention magnitude report:

\[
\Delta D_\tau^{obs}
=
D_\tau^{steered}
-
D_\tau^{base}.
\]

Compare predicted and observed using:

- correlation across task × dose cells;
- slope;
- intercept;
- RMSE;
- sign agreement.

The most important figure is:

\[
\Delta D^{pred}
\]

versus:

\[
\Delta D^{obs}.
\]

---

## 31. Strong quantitative success criterion

A particularly strong result would be:

\[
r(
\Delta D^{pred},
\Delta D^{obs}
)
\]

substantially positive across tasks and doses.

Do not require slope exactly 1.

Report calibration.

---

## 32. Cross-task qualitative prediction

The frozen behavioral model predicts different sensitivities across tasks.

Therefore the same standardized neural intervention should not produce identical behavioral effects.

The ordering should approximately reflect:

\[
\beta_{\tau,O^\*}.
\]

This is stronger evidence than simply demonstrating that steering changes continuation probability.

---

## 33. Steering specificity controls

Run:

### Negative direction

\[
-\alpha d_l.
\]

Effects should reverse.

### Random direction

Magnitude-matched random vectors.

### Orthogonal direction

Direction orthogonalized to \(d_l\).

### Generic value direction

If available.

### Persistence-output direction

Compare against a direct CONTINUE/STOP direction to distinguish computational steering from output steering.

---

## 34. Dose response

Require approximate monotonicity:

\[
\alpha
\rightarrow
\Delta D.
\]

Do not interpret only one hand-picked steering magnitude.

Plot full dose-response curves.

---

## 35. Direction-specific activation patching

After steering validation, implement subspace patching.

Given source \(s\) and target \(t\):

\[
m_s=d_l^\top h_s
\]

\[
m_t=d_l^\top h_t.
\]

Patch only this component:

\[
h_t'
=
h_t
+
d_l(m_s-m_t).
\]

Leave orthogonal components unchanged.

---

## 36. Why subspace patching is preferred

Whole residual-stream patching changes:

- task state;
- language content;
- current evidence;
- context;
- response tendencies.

Projection patching isolates the candidate computational mediator more directly.

---

## 37. Mediation contrast

For matched positive/negative relevant-history pairs define total behavioral effect:

\[
TE
=
D(x^+)-D(x^-).
\]

Projection-mediated intervention effect:

\[
ME_l
=
D(
x^-;
m_l\leftarrow m_l^+
)
-
D(x^-).
\]

Secondary normalized statistic:

\[
PM_l
=
\frac{ME_l}{TE}.
\]

Do not call \(PM_l\) a formal causal mediation proportion without appropriate assumptions.

Use:

> projection-mediated intervention fraction.

---

## 38. Contextual mediation test

For A→B→A examples:

Hold raw outcome sequence fixed.

Patch:

\[
m_l^{O^\*}
\]

from a condition reinstating A into a matched condition where B/recent history dominates.

Prediction:

Behavior should shift toward the persistence policy predicted from A-associated outcome history.

---

## 39. Direct versus contextual mediation

Compare patches for:

### Raw outcome-history direction

\[
d^O_l
\]

and:

### Context-relevant direction

\[
d^{O^\*}_l.
\]

If latent-context theory is correct:

\[
ME(d^{O^\*})
>
ME(d^O)
\]

especially on context-conflict examples.

If dual-history is sufficient:

\[
ME(d^O)
\]

should account for most of the effect.

---

## 40. Mechanistic theory outcomes

### Outcome A — Direct history implementation

Evidence:

- raw outcome history is strongly represented;
- context-relevant target adds little;
- steering raw-history direction changes persistence quantitatively;
- raw-history patch mediates behavioral effects.

Conclusion:

> **The model directly integrates recent outcome history into persistence decisions.**

### Outcome B — Context-transformed history

Evidence:

- context-relevant history is better represented than raw history at later layers;
- context relevance explains variance beyond recency;
- steering \(O^\*\) predicts behavior;
- context-relevant patching outperforms raw-history patching on A→B→A contrasts.

Conclusion:

> **The model transforms recent experience according to contextual relevance before using it to regulate persistence.**

### Outcome C — Both representations exist sequentially

Possible pattern:

\[
O_t
\]

strong early,

then:

\[
O_t^\*
\]

strong later.

Conclusion:

> **The network first represents recent outcomes and then performs a context-sensitive transformation before policy selection.**

This would be an especially strong mechanistic result.

### Outcome D — Representation without causal role

Strong decoding but steering/patching fails.

Conclusion:

> Computational variable is represented but the identified linear representation is not sufficient evidence of causal implementation.

Do not claim mechanism.

### Outcome E — No robust cross-task representation

Behavioral computation predicts behavior, but no shared neural variable generalizes.

Conclusion:

> A shared computational architecture may be implemented in task-specific neural coordinates.

This remains informative.

---

## 41. Efficient artifact strategy

Commit only compact artifacts.

Suggested:

```text
artifacts/mechanistic_v1/
    manifests/
        mechanistic_conditions.parquet
        split_manifest.json

    directions/
        outcome_history.safetensors
        contextual_history.safetensors
        action_history.safetensors
        metadata.json

    representation/
        layer_metrics.csv
        loto_metrics.csv
        specificity.csv
        random_controls.csv

    calibration/
        direction_calibration.csv

    steering/
        steering_results.parquet
        predicted_vs_observed.csv
        dose_response.csv

    patching/
        projection_patching.parquet
        mediation_summary.csv

    figures/
    report.md
    run_metadata.json
```

Target repository artifact footprint:

\[
<100\text{ MB}
\]

preferably substantially less.

---

## 42. Do not commit

Never commit:

```text
*.pt
*.npy
*.npz
```

containing full activations unless explicitly allowlisted.

Also exclude:

```text
temporary_activations/
model_cache/
temporary_batches/
```

---

## 43. Direction file sizes

For a model hidden size \(d\), storing a direction per layer requires only:

\[
L\times d
\]

values.

For approximately 32 layers and a few thousand hidden dimensions, this is only a small artifact.

Store directions in:

```text
safetensors
```

with accompanying metadata.

---

## 44. Optional projection cache

If repeated analyses need scalar projections, save:

```text
example_id
task
layer
target
projection
```

rather than hidden vectors.

A projection dataset across all layers should remain small.

---

## 45. Reproducibility manifest

Every mechanistic run records:

- model ID/revision;
- git commit;
- behavioral-model hash;
- mechanistic condition hash;
- activation position;
- layer index convention;
- direction fitting split;
- steering calibration;
- random seed.

---

## 46. Regression requirements

Before running new analysis:

1. reproduce frozen behavioral computational targets;
2. reproduce task-specific coefficients;
3. verify prompt/rendering hashes;
4. verify persistence logit extraction;
5. verify semantic label counterbalancing.

Mechanistic code may not silently alter behavioral preprocessing.

---

## 47. TDD requirements

All implementation follows:

\[
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
\]

### Streaming tests

- streamed mean equals full-matrix mean on tiny synthetic set;
- streamed regression equals batch regression;
- no hidden-state tensor persists after layer completion.

### Pairing tests

- matched histories differ only on intended variable;
- contextual pairs preserve raw outcome history where required;
- current-state factors match exactly.

### Layer tests

- layer numbering validated;
- activation hook position consistent;
- batch padding cannot alter extracted state.

### Probe tests

- train/test split leakage impossible;
- train-only normalization;
- synthetic linear target recovered.

### Steering tests

- zero \(\alpha\) reproduces baseline exactly;
- reversing direction reverses calibrated projection change;
- intervention changes only intended residual-stream state.

### Patching tests

- source=target gives zero intervention;
- projection replacement mathematically exact;
- orthogonal residual unchanged within tolerance.

---

## 48. Suggested code structure

```text
src/cognitive_discovery/
    mechanistic/
        targets/
            behavioral_targets.py
            history_targets.py
            contextual_targets.py

        dataset/
            matched_conditions.py
            contrasts.py
            split.py

        activations/
            hooks.py
            streaming_stats.py
            layer_scan.py

        representations/
            mean_difference.py
            ridge_probe.py
            low_rank.py
            loto.py
            specificity.py

        calibration/
            projection_to_computation.py

        steering/
            intervene.py
            dose_response.py
            quantitative_predictions.py
            controls.py

        patching/
            projection_patch.py
            mediation.py
            contextual_patch.py

        reporting/
            build_report.py
```

---

## 49. Configuration

Create:

```text
configs/mechanistic_v1.yaml
```

Suggested:

```yaml
protocol_version: mechanistic_v1

activation:
  token_position: final_prompt_token
  layers: all
  save_full_activations: false

scratch:
  env_var: MECH_SCRATCH
  delete_temporary_activations: true

targets:
  - outcome_history
  - contextual_outcome_history
  - action_history

representations:
  ranks: [1, 2, 4]
  random_controls: 100

discovery_tasks:
  - bandit
  - foraging
  - solvability
  - debugging

heldout_tasks:
  - waiting
  - effort
  - information_sampling

steering:
  computational_doses:
    - -2.0
    - -1.0
    - -0.5
    - 0.0
    - 0.5
    - 1.0
    - 2.0

patching:
  projection_only: true

artifact_limits:
  max_tracked_file_mb: 10
```

---

## 50. Required figures

### Figure 1 — Computational variables across depth

Decode:

\[
O_t
\]

and:

\[
O_t^\*
\]

across layers.

### Figure 2 — Raw → context-relevant transformation

Plot:

\[
R^2(O)
\]

and:

\[
R^2(O^\*)
\]

plus partial contextual relevance across depth.

### Figure 3 — Cross-task representation

LOTO performance for candidate computational variables.

### Figure 4 — Steering dose response

\[
\Delta O^\*
\rightarrow
\Delta D
\]

by task.

### Figure 5 — Predicted versus observed causal effects

Central mechanistic figure:

\[
\Delta D_\tau^{pred}
\]

versus:

\[
\Delta D_\tau^{observed}.
\]

### Figure 6 — Projection-mediated effects

Compare:

- raw outcome-history patch;
- context-relevant history patch;
- random control.

---

## 51. Automated report must answer

1. Is raw outcome history linearly represented?
2. Is context-relevant outcome history represented?
3. At what layers do these quantities emerge?
4. Does contextual relevance explain neural variance beyond raw history?
5. Do these representations transfer across tasks?
6. Are they more specific than generic value, terminality, or arbitrary-choice controls?
7. Which layers satisfy the representation gate?
8. Does steering alter the computational projection by the calibrated amount?
9. Does steering change persistence in the predicted direction?
10. Do task-specific steering effects correlate with frozen behavioral coefficients?
11. Is the dose response monotonic?
12. Does projection patching mediate outcome-history effects?
13. Does context-relevant patching outperform raw-history patching on context-conflict examples?
14. Which mechanistic theory is best supported?
15. What mechanistic claims are justified and which remain unsupported?

---

## 52. Mechanistic gates

### Gate 1 — Representation

Proceed to steering only if a candidate target:

- decodes above matched controls;
- generalizes to held-out examples;
- demonstrates reasonable cross-task transfer or clearly documented task-specificity;
- is not explained by output geometry.

### Gate 2 — Computational alignment

Proceed to causal interpretation only if:

- actual history manipulation moves neural projection appropriately;
- calibrated neural projection predicts the behavioral computational variable;
- task/value controls do not explain the effect.

### Gate 3 — Steering causality

Strong evidence requires:

- bidirectional steering;
- dose response;
- random-direction specificity;
- predicted cross-task effect ordering.

### Gate 4 — Mediation

Projection patching should causally transfer at least part of the matched behavioral effect.

Only then describe the representation as part of the causal implementation.

---

## 53. Stop rules

Stop deeper mechanistic work if:

- decoding fails cross-task and within-task effects are clearly output-adjacent;
- random directions perform comparably;
- steering changes persistence but not in relation to the computational calibration;
- patching fails while broad steering succeeds nonspecifically.

Do not escalate automatically into circuit localization.

---

## 54. Optional next phase

Only after a successful computational steering/patching result should the project consider:

- head localization;
- MLP localization;
- causal tracing;
- path patching;
- DAS;
- SAE feature analysis.

The question then becomes:

> **Which transformer components construct and transmit the validated computational variable?**

That is outside this PRD.

---

## 55. Paper-level success criterion

The strongest result would be:

> **Behavioral discovery identified context-sensitive outcome-history integration as a central determinant of LLM persistence. We recovered an internal representation of this computational quantity and causally manipulated it. Critically, the magnitude of the resulting change in persistence across tasks was predicted by independently estimated behavioral-model parameters.**

Formally:

\[
\boxed{
\text{experiment}
\rightarrow
\text{computational model}
\rightarrow
\text{internal variable}
\rightarrow
\text{quantitatively predicted intervention}
}
\]

This is the main scientific endpoint of the project.

Even a weaker outcome remains useful if it clearly identifies where the bridge between behavioral computation and neural implementation breaks.
