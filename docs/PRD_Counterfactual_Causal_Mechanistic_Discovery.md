# PRD — Counterfactual Causal Mechanistic Discovery for Cognitive Variables in LLMs

**Project:** Computational Cognitive Discovery for AI Systems  
**Use case:** Persistence / disengagement as the first application  
**Primary model:** Qwen3.5-4B  
**Purpose:** Replace probe-first mechanistic interpretation with a workflow in which an independently validated computational theory generates counterfactual predictions that constrain neural causal discovery.

---

## 1. Motivation

The previous mechanistic workflow was approximately:

\[
\text{computational variable}
\rightarrow
\text{decode from activations}
\rightarrow
\text{use decoder weights as direction}
\rightarrow
\text{steer}
\rightarrow
\text{interpret as mechanism}.
\]

This pipeline produced two important failures.

In the future-reward analysis, future reward was decodable but the identified representation did not explain or causally regulate persistence in the way expected from the proposed computation.

In the action-history analysis, action history was strongly decodable and remained identifiable independently of the current persistence decision, but steering the corresponding 1-D probe direction failed causal-specificity controls and failed to reproduce the task-specific effects predicted by the behavioral model.

These failures demonstrate a general methodological problem:

\[
\boxed{\text{decodability} \neq \text{causal implementation}}
\]

and:

\[
\boxed{\text{probe direction} \neq \text{natural intervention on the computational variable}.}
\]

A predictive probe identifies information that can be extracted from a neural state. It does not necessarily identify the representation that downstream model computations use.

The new workflow therefore makes **counterfactual causal alignment** the central criterion for mechanistic success.

---

## 2. Core scientific principle

A candidate neural representation should count as an implementation of a computational variable \(X\) only if interventions on that representation reproduce the effects predicted by interventions on \(X\) in an independently specified computational model.

Given a high-level model:

\[
Y=f(X,Z),
\]

the behavioral theory predicts:

\[
do(X=x') \Rightarrow Y^{CF}=f(x',Z).
\]

The mechanistic objective is to identify a neural intervention:

\[
do(H_X=h')
\]

such that:

\[
Y_{\text{LLM}}^{CF} \approx Y^{CF}.
\]

Therefore:

\[
\boxed{\text{mechanistic success}=\text{counterfactual equivalence}}
\]

rather than merely:

\[
\text{mechanistic success}=\text{linear decoding + steering}.
\]

---

## 3. New discovery workflow

The full pipeline becomes:

\[
\boxed{\text{Experimental grammar}}
\rightarrow
\boxed{\text{Broad behavioral sampling}}
\rightarrow
\boxed{\text{Computational theory discovery}}
\rightarrow
\boxed{\text{Behavioral counterfactual predictions}}
\rightarrow
\boxed{\text{Causal neural localization}}
\rightarrow
\boxed{\text{Causal representation search}}
\rightarrow
\boxed{\text{Held-out counterfactual validation}}
\rightarrow
\boxed{\text{Circuit localization}}
\rightarrow
\boxed{\text{Mechanistic theory}}.
\]

Decoding remains part of the workflow, but it is a **diagnostic**, not the definition of a mechanism.

---

## 4. Entry criteria

Mechanistic analysis should not begin simply because a behavioral variable correlates with model behavior.

At least one computational theory must satisfy:

1. strong held-out behavioral prediction;
2. improvement over simple baselines;
3. clear operational definitions of latent/computational variables;
4. successful synthetic recovery;
5. meaningful counterfactual predictions;
6. robustness across relevant task families.

If several behavioral theories remain viable, retain all of them.

Mechanistic experiments may then be used to discriminate between theories if they make different neural counterfactual predictions.

---

## 5. Stage 1 — Freeze the computational theory

Before neural analysis, freeze:

- model architecture;
- fitted parameters;
- variable definitions;
- normalization;
- task-specific coefficients;
- model version/hash;
- dataset version;
- behavioral predictions.

For persistence, a generic theory may take the form:

\[
D_t=f_\tau(X_t,A_t,O_t,O_t^*),
\]

where:

- \(D_t\): persistence logit;
- \(X_t\): current-state evidence;
- \(A_t\): action-history quantity;
- \(O_t\): raw outcome history;
- \(O_t^*\): context-sensitive outcome history;
- \(\tau\): task.

No neural result may change these definitions.

---

## 6. Stage 2 — Generate computational counterfactuals

The behavioral model must generate explicit counterfactual predictions before neural interventions.

For target variable \(X\), construct matched:

- **base state** \(b\);
- **source state** \(s\).

Example:

\[
b=(X_b,Z_b)
\]

and:

\[
s=(X_s,Z_s).
\]

Construct pairs such that:

\[
Z_s\approx Z_b
\]

while:

\[
X_s\neq X_b.
\]

The behavioral model predicts:

\[
Y^{CF}_{b\leftarrow s}=f(X_s,Z_b).
\]

The predicted causal effect is:

\[
\Delta Y^{CF}=f(X_s,Z_b)-f(X_b,Z_b).
\]

This value must be stored before neural intervention.

---

## 7. Continuous rather than binary counterfactuals

Whenever possible, use continuous counterfactual predictions.

For example, if:

\[
D_t=\beta_A A_t+\cdots
\]

then:

\[
\Delta D^{CF}=\beta_A(A_s-A_b).
\]

This is substantially stronger than merely predicting:

```text
CONTINUE → STOP
```

because the theory predicts **how much** behavior should change.

The primary neural test should therefore be:

\[
\boxed{\Delta D^{CF}\quad\text{vs}\quad\Delta D^{neural}}
\]

on data not used to discover the neural intervention.

---

## 8. Stage 3 — Diagnostic representation scan

Representation analysis is still useful.

At every layer, test whether relevant computational quantities can be recovered from activations.

Candidate methods:

- linear ridge probes;
- low-rank linear probes;
- nonlinear diagnostic probes if needed;
- RSA / representational similarity;
- matched-contrast analyses.

Report:

\[
R^2,\quad r,\quad \text{LOTO},\quad \text{matched-state specificity}.
\]

But explicitly label these results:

> **information-access results**

rather than:

> **mechanistic results**.

---

## 9. What decoding may be used for

Decoding can inform:

- which layers to investigate;
- whether a variable is available at all;
- whether representation changes across depth;
- whether task-general information exists;
- whether raw versus transformed variables appear at different stages.

Decoding may **not** justify:

- claiming causal use;
- treating probe weights as the natural neural variable;
- interpreting successful probe steering as causal implementation without additional controls.

---

## 10. Stage 4 — Coarse causal localization using activation patching

Before searching for a special direction or subspace, determine whether a layer actually contains causally relevant information.

For matched source/base pairs, patch the complete residual-stream activation:

\[
h_l(b)\leftarrow h_l(s).
\]

Measure:

\[
\Delta D_l^{patch}.
\]

Compare with the computational prediction:

\[
\Delta D^{CF}.
\]

This identifies layers at which source-state information can causally transfer the predicted behavioral effect.

---

## 11. Why whole-state patching comes first

Whole-state patching is deliberately coarse.

It cannot prove that variable \(X\) is the mediator because it transfers many differences simultaneously.

Its role is localization:

> **Is the relevant causal information present and usable at this layer?**

If whole-state patching at a layer cannot transfer the behavioral counterfactual under strong matching, there is little reason to search extensively for a tiny causal subspace there.

---

## 12. Coarse localization metrics

For each layer report:

\[
r(\Delta D^{patch},\Delta D^{CF})
\]

along with:

- slope;
- intercept;
- RMSE;
- sign accuracy;
- fraction of behavioral counterfactual recovered.

Define:

\[
CFR_l=1-\frac{(\Delta D_l^{patch}-\Delta D^{CF})^2}{(\Delta D^{base}-\Delta D^{CF})^2+\epsilon}.
\]

This can serve as a normalized counterfactual-recovery score.

---

## 13. Stage 5 — Causal representation search

Once candidate layers are localized, search for a **minimal neural representation whose intervention reproduces the computational counterfactual**.

The optimization target is no longer:

\[
\hat X=w^\top h.
\]

It is:

\[
\boxed{\text{intervene on neural representation}\rightarrow Y^{CF}.}
\]

---

## 14. Primary method — Distributed Alignment Search

Use Distributed Alignment Search / Boundless DAS-style interventions as the primary next method.

Learn an alignment:

\[
z=Rh
\]

and candidate subspace \(S\).

Given source \(s\) and base \(b\):

\[
z_b'=z_b
\]

except:

\[
z'_{b,S}=z_{s,S}.
\]

Transform back:

\[
h_b'=R^{-1}z_b'.
\]

Run the remainder of the network and obtain:

\[
D^{DAS}.
\]

Optimize \(R\) and/or subspace boundaries such that:

\[
D^{DAS}\approx D^{CF}.
\]

---

## 15. DAS objective

Primary continuous loss:

\[
\mathcal L_{CF}=\frac{1}{N}\sum_i(D_i^{DAS}-D_i^{CF})^2.
\]

Optional likelihood version:

\[
\mathcal L=-\sum_i\log P_{\text{LLM}}(Y_i^{CF}\mid do(S_i)).
\]

Prefer continuous semantic logits where available.

---

## 16. Search dimensionality

Start with:

\[
k\in\{1,2,4,8\}.
\]

Do not jump immediately to highly expressive interventions.

The scientific preference should be:

\[
\boxed{\text{smallest causal representation that passes validation}}.
\]

Compare performance against:

- \(k=1\);
- \(k=2\);
- \(k=4\);
- \(k=8\).

Only increase dimensionality when smaller models demonstrably fail.

---

## 17. Simplicity regularization

Avoid learning arbitrary controllers.

Possible objective:

\[
\mathcal L=\mathcal L_{CF}+\lambda_k k+\lambda_\Delta\|h'-h\|^2.
\]

The intervention should:

- use a small subspace;
- make minimal activation changes;
- generalize;
- transfer across counterfactual pairs.

---

## 18. Critical train/test separation

Split counterfactual pairs into:

```text
train
validation
test
held-out task
```

DAS/alignment parameters may use only training pairs.

Hyperparameters may use validation.

All scientific claims use test and held-out-task results.

No source/base test pair may influence the learned neural alignment.

---

## 19. Counterfactual generalization is the primary metric

A causal representation should generalize to new:

- histories;
- factor combinations;
- prompts;
- response mappings;
- tasks where appropriate.

The strongest test is:

\[
\text{train neural alignment on }N-1\text{ tasks}\rightarrow\text{intervene on held-out task}.
\]

However, failure of shared coordinates does not automatically falsify shared computation.

Also test task-specific alignments under a shared causal objective.

---

## 20. Hierarchical neural implementation

Allow three hypotheses:

### Neural H1 — Shared coordinates

\[
S_\tau=S.
\]

One neural subspace implements the variable across tasks.

### Neural H2 — Shared computational role, task-specific coordinates

\[
S_\tau\neq S_{\tau'}
\]

but:

\[
do(S_\tau:X\leftarrow x')
\]

produces the same high-level counterfactual semantics across tasks.

### Neural H3 — Task-specific computation

Neither coordinates nor counterfactual causal structure transfer.

These correspond directly to the hierarchical logic already used behaviorally.

---

## 21. Stage 6 — Causal specificity controls

Every learned causal representation must beat:

### Random subspaces

Matched:

- layer;
- dimensionality;
- intervention norm.

### Shuffled source states

Interchange the wrong \(X\) value.

### Shuffled counterfactual targets

Break:

\[
X_s\leftrightarrow D^{CF}.
\]

### Persistence/output subspace

Positive control representing direct response manipulation.

### Generic-value subspace

Test whether the representation is merely general good/bad state.

### Task-ID subspace

Ensure effect is not task routing.

### Response-mapping control

Ensure effects are semantic rather than token-specific.

---

## 22. Counterfactual specificity test

A candidate \(X\)-representation should reproduce:

\[
do(X)
\]

counterfactuals.

It should **not** automatically reproduce:

\[
do(Z)
\]

for unrelated computational variable \(Z\).

Define:

\[
CS_X=\text{CFR}_{X\text{-counterfactual}}-\max_Z\text{CFR}_{Z\text{-counterfactual}}.
\]

Positive specificity is required for mechanistic identification.

---

## 23. Necessity as well as sufficiency

Interchange interventions demonstrate a form of sufficiency.

Also test necessity when possible.

Possible intervention:

\[
h'=h-P_Sh,
\]

or replace \(S\) with a matched neutral/reference value.

Ask whether removing the representation selectively reduces sensitivity to the corresponding computational variable.

The strongest case has:

\[
\boxed{\text{sufficiency + necessity + specificity}}.
\]

---

## 24. Behavioral-model correspondence gate

The key criterion is not merely:

\[
\Delta D^{neural}\neq0.
\]

It is:

\[
\boxed{\Delta D^{neural}\approx\Delta D^{CF}.}
\]

Report:

- correlation;
- regression slope;
- intercept;
- RMSE;
- sign agreement;
- calibration;
- task-wise performance.

A neural intervention that strongly affects persistence but does not match the behavioral theory should be treated as a **behavior-control intervention**, not evidence for the computational mechanism.

---

## 25. Theory discrimination

If two behavioral theories survive, use neural counterfactuals to distinguish them.

Suppose \(M_1\) predicts \(D_1^{CF}\) and \(M_2\) predicts \(D_2^{CF}\).

Select conditions where:

\[
|D_1^{CF}-D_2^{CF}|
\]

is large.

Then ask whether the best validated causal neural intervention follows \(M_1\) or \(M_2\).

This makes mechanistic analysis an additional source of scientific evidence rather than merely a post-hoc illustration.

---

## 26. Stage 7 — Only then consider nonlinear representations

Failure of a 1-D direction does **not** imply a nonlinear mechanism.

Test in order:

\[
1D\rightarrow2D\rightarrow4D\rightarrow8D.
\]

Only if small linear subspaces fail should the project consider nonlinear representation mappings.

Possible follow-ups:

- nonlinear DAS/alignment functions;
- sparse autoencoders;
- intervention autoencoders;
- learned nonlinear bottlenecks;
- manifold methods.

---

## 27. Nonlinear-method guardrail

A sufficiently expressive nonlinear transformation can manufacture a controller.

Therefore nonlinear methods require stronger validation.

At minimum:

1. strict held-out counterfactual pairs;
2. held-out tasks;
3. shuffled-target null;
4. dimensionality/bottleneck regularization;
5. minimal-intervention penalty;
6. independently frozen behavioral model;
7. comparison to equally expressive random/null models.

Do not interpret training-set counterfactual success as mechanism.

---

## 28. SAEs are secondary, not primary

Sparse autoencoders may later be used to characterize a validated causal subspace.

For example, \(S_X\) may overlap with several sparse features.

Then ask whether those features:

- carry the computational variable;
- mediate the validated counterfactual;
- participate in downstream circuitry.

Do not begin with:

\[
\text{SAE feature correlates with }X\rightarrow\text{steer SAE feature}\rightarrow\text{claim mechanism}.
\]

That repeats the same methodological mistake in a different basis.

---

## 29. Stage 8 — Circuit localization

Only after a causal representation passes counterfactual validation should analysis move down to components.

Questions then become:

> Which components construct \(S_X\)?

and:

> Which components read \(S_X\) to affect behavior?

Methods may include:

- attention-head patching;
- MLP patching;
- path patching;
- causal tracing;
- attribution patching for candidate generation;
- component ablation.

---

## 30. Upstream circuit discovery

Given validated representation \(S_X\) at layer \(l\):

Patch or ablate candidate upstream components and ask whether they change:

\[
P_S h_l.
\]

Primary target:

\[
\Delta S_X.
\]

Secondary target:

\[
\Delta D.
\]

This separates components that **construct the variable** from components that simply alter the final behavior.

---

## 31. Downstream circuit discovery

Patch the validated \(S_X\) representation and determine which later:

- attention heads;
- MLPs;
- residual paths

transmit its causal effect to persistence.

Use path-specific interventions where feasible.

The desired explanation becomes:

\[
\boxed{\text{history evidence}\rightarrow\text{component A}\rightarrow S_X\rightarrow\text{component B}\rightarrow D}
\]

rather than merely:

\[
d_X\rightarrow D.
\]

---

## 32. Evidence hierarchy

Use the following levels in reports and papers.

### Level 0 — Behavioral association

Variable correlates with behavior.

### Level 1 — Computational model

Variable improves held-out behavioral prediction.

### Level 2 — Information representation

Variable can be decoded from activations.

### Level 3 — Coarse causal localization

Activation patching transfers the expected effect.

### Level 4 — Causal representation

A small neural intervention reproduces frozen computational counterfactuals.

### Level 5 — Causal specificity

Intervention passes random, shuffled, decision, value, and task controls.

### Level 6 — Circuit implementation

Specific components construct and transmit the causal representation.

Only Levels 4–6 should be described as strong evidence of mechanistic implementation.

---

## 33. Claims vocabulary

Use:

> “Information about \(X\) is linearly decodable.”

for Level 2.

Use:

> “\(X\)-related information at layer \(l\) has causal influence.”

for coarse patching.

Use:

> “A neural subspace is causally aligned with computational variable \(X\).”

only after counterfactual validation.

Use:

> “The model implements computational variable \(X\) through subspace \(S\).”

only after specificity and appropriate necessity/sufficiency tests.

Avoid:

> “We found the \(X\) direction”

based solely on probing.

---

## 34. Primary persistence application

The first application of the new workflow should focus on the strongest remaining behavioral computational variables rather than the previously identified 1-D probe directions.

Candidate variables:

\[
O_t
\]

raw outcome history,

\[
O_t^*
\]

context-relevant outcome history,

and where scientifically justified:

\[
A_t.
\]

The behavioral model must determine which variables are primary.

The representation search does not get to redefine the theory based on whichever variable is easiest to decode.

---

## 35. Persistence counterfactual example

Suppose a Bandit base state has:

\[
O_b^*=0.2
\]

and a matched source state has:

\[
O_s^*=1.1.
\]

The frozen behavioral model predicts:

\[
D_b=f_\tau(X_b,0.2)
\]

and:

\[
D^{CF}=f_\tau(X_b,1.1).
\]

Therefore:

\[
\Delta D^{CF}=D^{CF}-D_b.
\]

The mechanistic search asks:

> Which internal intervention transfers the source's \(O^*\) representation into the base while producing approximately \(\Delta D^{CF}\)?

That is the target—not whether an \(O^*\) probe has high \(R^2\).

---

## 36. Context-sensitive history test

For A→B→A conditions, construct pairs with identical or nearly identical raw history:

\[
O_t^{(1)}\approx O_t^{(2)}
\]

but different context-sensitive history:

\[
O_t^{*(1)}\neq O_t^{*(2)}.
\]

Dual-history and latent-context theories should generate counterfactual predictions.

Search for the neural intervention that reproduces those predictions.

This may directly help resolve the remaining behavioral theory ambiguity.

---

## 37. Negative-control variable

Information Sampling currently provides a useful weak-sensitivity condition for outcome-history effects.

If the frozen behavioral theory predicts:

\[
\beta_{\text{InfoSampling},O^*}\approx0,
\]

then a valid \(O^*\) intervention should produce:

\[
\Delta D_{\text{InfoSampling}}\approx0
\]

even if the neural intervention successfully changes the underlying representation.

This is a powerful mechanistic negative control.

---

## 38. Active mechanistic experiment selection

Once the computational model is frozen, candidate source/base pairs can be actively selected.

Prefer conditions with:

1. large predicted counterfactual effect;
2. strong disagreement between surviving theories;
3. good nuisance-variable matching;
4. coverage across tasks and histories;
5. uncertainty in current neural causal model.

This can later be formalized through Bayesian experimental design.

---

## 39. Mechanistic active-sampling objective

A candidate condition \(x\) can receive information value based on uncertainty over:

- behavioral theory \(M\);
- neural implementation \(S\);
- causal parameters.

Conceptually:

\[
x^*=\arg\max_x I(Y_x;M,S,\theta\mid\mathcal D).
\]

Do not use active sampling until broad behavioral coverage has established a credible theory bank.

---

## 40. Data splits

Recommended:

```text
behavioral_discovery
behavioral_validation

mech_pair_train
mech_pair_validation
mech_pair_test
mech_task_holdout
```

The behavioral-validation data must remain untouched by mechanistic optimization.

The mechanistic test split must remain untouched by representation search.

---

## 41. Storage philosophy

Continue:

\[
\boxed{\text{stream activations, save causal summaries}}
\]

rather than saving large activation banks.

Store:

- source/base IDs;
- frozen computational variables;
- predicted counterfactuals;
- learned alignment matrices;
- subspace masks;
- scalar projections;
- intervention results;
- layer/component metrics.

Do not store full activation tensors unless temporarily required.

---

## 42. Suggested artifact structure

```text
artifacts/causal_mech_v1/
    behavioral_handoff/
        model_hash.json
        frozen_coefficients.csv
        variable_definitions.json

    counterfactuals/
        pair_manifest.parquet
        predicted_effects.parquet

    localization/
        whole_state_patching.parquet
        layer_summary.csv

    representations/
        diagnostic_decoding.csv
        das_rank1.safetensors
        das_rank2.safetensors
        das_rank4.safetensors
        das_rank8.safetensors

    validation/
        counterfactual_recovery.parquet
        heldout_task_results.csv
        specificity_controls.csv
        random_subspace_null.csv

    circuits/
        component_patching.parquet
        path_patching.parquet

    figures/
    gates.json
    report.md
    run_metadata.json
```

---

## 43. TDD requirements

Maintain:

\[
\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}.
\]

### Counterfactual tests

- source/base pairing changes only intended variable within tolerance;
- behavioral counterfactual recomputation is deterministic;
- frozen behavioral model cannot be updated during mechanistic analysis.

### Patching tests

- source=base patch produces zero change;
- whole-state replacement is exact;
- specified token/layer is patched only once.

### DAS tests

- identity alignment + empty subspace reproduces baseline;
- full subspace replacement reproduces whole-state patch;
- shuffled source assignment degrades synthetic recovery;
- synthetic distributed teacher is recoverable.

### Split tests

- no test pair participates in representation learning;
- no held-out task normalization leakage;
- no behavioral-validation rows are used during mechanistic optimization.

### Intervention tests

- target subspace alone changes under interchange;
- orthogonal complement remains unchanged within tolerance;
- intervention norm logged.

### Counterfactual metric tests

- perfect intervention gives CFR=1;
- baseline/no-op gives expected baseline score;
- reversed counterfactual scores poorly.

---

## 44. Synthetic validation

Before real-model claims, construct synthetic systems with known:

### Case A

1-D causal variable.

### Case B

4-D distributed linear representation.

### Case C

task-specific rotations of a shared computational variable.

### Case D

decodable but non-causal correlated variable.

### Case E

downstream decision variable correlated with upstream computational variable.

The pipeline should distinguish these cases.

This is essential given the previous probe-steering failures.

---

## 45. Mandatory null result reporting

The pipeline should explicitly support scientifically useful negative conclusions.

Examples:

> “The variable is behaviorally predictive but no causally aligned neural representation was identified.”

> “The variable is decodable but not causally localized.”

> “A causal implementation exists but uses task-specific neural coordinates.”

> “The behavioral theories remain observationally and mechanistically equivalent under the tested interventions.”

These are acceptable outcomes.

---

## 46. Stop conditions

Stop escalation if:

- whole-state patching cannot transfer predicted counterfactual effects;
- DAS performs no better than matched random subspaces;
- shuffled-target DAS performs similarly;
- held-out counterfactual recovery collapses;
- causal intervention affects behavior but not according to the computational theory.

Do not respond by merely increasing model flexibility.

Return to the computational theory or experimental design.

---

## 47. When to revisit behavioral theory

Mechanistic failure should trigger behavioral reconsideration when:

\[
\text{behavioral model predicts a strong intervention}
\]

but:

\[
\text{no plausible neural intervention produces it}
\]

across multiple layers and methods.

Possible explanations include:

- wrong computational variable;
- wrong temporal abstraction;
- omitted latent state;
- wrong causal interpretation of fitted coefficients;
- behavioral model capturing prediction rather than computation.

Thus mechanism can falsify computational theory rather than merely decorate it.

---

## 48. Core methodological distinction

The project should explicitly distinguish:

\[
\boxed{\text{representation discovery}}
\]

from:

\[
\boxed{\text{mechanism discovery}}.
\]

Representation discovery asks:

> What information is present?

Mechanism discovery asks:

> What internal intervention reproduces the causal structure of the computational theory?

Both are valuable.

Only the second answers the scientific question of implementation.

---

## 49. Desired scientific endpoint

The strongest possible persistence result is no longer:

> “We found a direction encoding persistence-related variable \(X\), and steering it changes persistence.”

Instead:

> **“A computational model independently identified \(X\) as a determinant of persistence and predicted quantitative counterfactual effects of changing \(X\). We identified a neural subspace whose targeted interchange interventions reproduced those predictions on held-out conditions and tasks, passed causal-specificity controls, and traced the components that construct and transmit that representation.”**

Formally:

\[
\boxed{\text{Behavior}\rightarrow\text{Computational theory}\rightarrow\text{Counterfactual predictions}\rightarrow\text{Causal neural abstraction}\rightarrow\text{Circuit}}
\]

---

## 50. Broader methodological goal

This workflow should ultimately become a reusable component of the broader computational-cognitive discovery framework.

For any proposed model variable \(X\):

\[
\text{Cognitive theory}
\]

should specify:

\[
do(X)
\]

and predict its consequences.

Mechanistic discovery then becomes a constrained search for neural interventions satisfying those same counterfactual relationships.

The resulting paradigm is:

\[
\boxed{\text{discover computations from behavior}+\text{discover implementations from counterfactual equivalence}}
\]

rather than:

\[
\boxed{\text{discover correlates in activations and assign them cognitive labels}.}
\]

That should be the governing methodological principle for the next phase of the project.
