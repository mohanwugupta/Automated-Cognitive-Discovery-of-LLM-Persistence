# PRD — Identifying the Causal Abstraction Level of Persistence Computation

**Project:** Computational Cognitive Discovery for AI Systems  
**Application:** Persistence / disengagement in Qwen3.5-4B  
**Stage:** Mechanistic theory refinement after successful Level-4 DAS discovery  
**Primary objective:** Determine **which level of the behavioral computational decomposition corresponds to the causal neural state identified by DAS**.

The previous analysis established a strong but deliberately limited result. A low-dimensional DAS subspace reproduces frozen behavioral counterfactuals on untouched test examples, substantially beats matched random subspaces, and transfers across held-out tasks. However, it does not satisfy variable-specificity criteria for either raw outcome history or contextual outcome history. The current justified interpretation is therefore an effective persistence controller without established variable identity.

This PRD tests whether that apparent failure of one-to-one correspondence instead reflects a **different causal abstraction level** inside the network.

---

## 1. Scientific motivation

The broader scientific program is:

\[
\boxed{
\text{behavioral experiments}
\rightarrow
\text{computational theory}
\rightarrow
\text{mechanistic discovery}
}
\]

The goal is not necessarily to prove that every term in a behavioral model has a one-to-one neural representation.

Instead, computational models provide candidate decompositions of the computation and generate controlled counterfactuals.

Mechanistic analysis can then determine whether the model implements:

1. the original computational variable;
2. a transformed version of that variable;
3. an integrated downstream quantity;
4. or a more generic decision state.

The current DAS results strongly support correspondence at the level of **causal counterfactual structure**, but not yet at the level of individual variable identity.

This PRD asks:

> **At what computational abstraction level does the identified causal neural subspace correspond to the behavioral theory?**

---

## 2. Core causal hierarchy

Represent the candidate persistence computation as:

\[
\text{raw evidence/history}
\rightarrow
\text{context-sensitive transformation}
\rightarrow
\text{integrated decision contribution}
\rightarrow
\text{persistence decision}.
\]

Define four candidate abstraction levels.

### Level A — Raw history

\[
O_t
\]

Recent outcome history independent of contextual relevance.

### Level B — Context-transformed history

\[
O_t^\*
\]

History after weighting or selecting prior outcomes according to current context.

### Level C — Integrated history contribution

Define:

\[
H_t
=
\beta_{A,\tau}A_t
+
\beta_{O,\tau}O_t
\]

for dual-history models, or the analogous term for contextual-history models.

More generally:

\[
H_t
=
f_\tau(X_t,H_t^{history})
-
f_\tau(X_t,H_t^{history}=0).
\]

Interpretation:

> **How much does history, in total, currently contribute to persistence?**

This is a downstream quantity that may collapse several upstream variables.

### Level D — Integrated persistence evidence

Define:

\[
E_t
=
f_\tau(
X_t,
A_t,
O_t,
O_t^\*
).
\]

This is the full pre-decision evidence state predicted by the behavioral model.

Depending on model parameterization it may be equivalent or closely related to:

\[
D_t.
\]

Importantly, distinguish this from the literal output logit direction in the network.

---

## 3. Competing mechanistic hypotheses

### H1 — Raw-history implementation

The causal neural representation corresponds specifically to:

\[
O_t.
\]

Predictions:

- responds to changes in raw history even when contextual interpretation and downstream contribution are matched;
- does not respond when raw history is fixed but contextual relevance changes;
- raw-history counterfactuals outperform integrated-decision explanations.

### H2 — Context-transformed implementation

The causal representation corresponds to:

\[
O_t^\*.
\]

Predictions:

- distinguishes identical raw histories assigned different contextual relevance;
- remains stable across physically different histories with equivalent context-weighted meaning;
- contextual-history interventions outperform raw-history predictions.

### H3 — Integrated history contribution

The neural state represents something closer to:

\[
H_t.
\]

Different combinations of:

- action history;
- outcome history;
- contextual weighting

that yield the same:

\[
H_t
\]

should produce similar neural states and interchange effects.

Prediction:

\[
\boxed{
\text{upstream variables differ}
\quad\text{but}\quad
H_t\text{ matched}
\Rightarrow
S_H\text{ matched}.
}
\]

### H4 — General persistence evidence

The discovered subspace reflects a downstream decision quantity:

\[
E_t.
\]

Different causal routes producing the same persistence evidence should converge on the same neural representation.

For example:

\[
\text{better history}
\]

and:

\[
\text{lower continuation cost}
\]

may yield equivalent neural states if they produce the same:

\[
E_t.
\]

### H5 — Generic controller / learned intervention artifact

The DAS subspace does not correspond cleanly to any naturally used computational variable.

Prediction:

- interventions work even where abstraction-level predictions disagree;
- shuffled target mappings remain competitive;
- naturally occurring neural states do not organize according to candidate computational equivalence classes;
- source states can be replaced by artificial controllers without preserving variable semantics.

---

## 4. Central methodological change

Do **not** train separate DAS models for every theory and simply compare which achieves the highest counterfactual recovery.

That risks allowing optimization to manufacture a successful controller for each theory.

Instead:

\[
\boxed{
\text{design behavioral conditions that force the candidate abstraction levels to disagree}
}
\]

and evaluate **frozen neural representations and interventions** on those conditions.

The main evidence comes from **causal dissociations**.

---

## 5. Experimental strategy

Construct a dedicated **mechanistic identifiability dataset**.

The dataset should contain conditions selected specifically to orthogonalize:

\[
O_t,
\quad
O_t^\*,
\quad
H_t,
\quad
E_t.
\]

The goal is not broad behavioral coverage.

The goal is:

\[
\boxed{
\text{maximum disagreement among candidate abstraction levels}.
}
\]

---

## 6. Contrast Family 1 — Same raw history, different contextual interpretation

Construct:

\[
O_1=O_2
\]

but:

\[
O_1^\*\neq O_2^\*.
\]

Hold as closely as possible:

- current state;
- continuation cost;
- progress;
- response mapping;
- task;
- current reward/value.

This directly distinguishes:

\[
O
\]

from:

\[
O^\*.
\]

### Predictions

Raw-history representation:

\[
\Delta S_O\approx0.
\]

Contextual representation:

\[
\Delta S_{O^\*}\neq0.
\]

Integrated-history representation:

change proportional to:

\[
\Delta H.
\]

---

## 7. Contrast Family 2 — Different raw histories, same contextual history

Construct:

\[
O_1\neq O_2
\]

but:

\[
O_1^\*\approx O_2^\*.
\]

This provides the reverse dissociation.

### Predictions

Raw-history:

\[
\Delta S_O\neq0.
\]

Context-transformed:

\[
\Delta S_{O^\*}\approx0.
\]

---

## 8. Contrast Family 3 — Different upstream history, same integrated history contribution

Construct conditions where:

\[
(A_1,O_1,O_1^\*)
\neq
(A_2,O_2,O_2^\*)
\]

but:

\[
H_1\approx H_2.
\]

For example:

\[
\beta_AA_1+\beta_OO_1
\approx
\beta_AA_2+\beta_OO_2.
\]

This is critical.

If the network represents raw computational ingredients, neural states should still differ.

If the network has already integrated those ingredients, the causal representation should converge:

\[
S_{H,1}\approx S_{H,2}.
\]

---

## 9. Contrast Family 4 — Same history contribution, different total persistence evidence

Hold:

\[
H_1\approx H_2
\]

while changing another decision input such as:

- continuation cost;
- success evidence;
- disengagement value;
- progress.

Thus:

\[
E_1\neq E_2.
\]

This distinguishes:

\[
\boxed{\text{integrated history}}
\]

from:

\[
\boxed{\text{general persistence evidence}}.
\]

---

## 10. Contrast Family 5 — Different causal routes, same total persistence evidence

This is perhaps the strongest test of a downstream decision state.

Construct conditions such that:

\[
E_1\approx E_2
\]

but through very different causes.

Example:

### Condition A

Strong positive history:

\[
H_A\gg0
\]

but high continuation cost.

### Condition B

Neutral history:

\[
H_B\approx0
\]

but low continuation cost.

Choose parameters such that:

\[
D_A\approx D_B.
\]

If the neural representation reflects general persistence evidence:

\[
S_A\approx S_B.
\]

If it reflects history specifically:

\[
S_A\neq S_B.
\]

---

## 11. Contrast Family 6 — Same predicted behavioral effect, different manipulated variable

Construct counterfactual interventions with the same predicted:

\[
\Delta D^{CF}.
\]

For example:

\[
do(O): \Delta D=+1
\]

and:

\[
do(Cost): \Delta D=+1.
\]

The final behavioral consequence is identical, but the upstream intervention is different.

This is the cleanest distinction between:

\[
\text{history-specific representation}
\]

and:

\[
\text{generic downstream decision controller}.
\]

---

## 12. Candidate variable table

For every semantic condition calculate and freeze:

| Variable | Meaning |
|---|---|
| \(O\) | raw outcome history |
| \(O^\*\) | context-relevant outcome history |
| \(A\) | action history |
| \(H\) | integrated history contribution |
| \(E\) | total persistence evidence |
| \(D\) | observed/frozen persistence logit |
| \(C\) | continuation cost |
| \(V_{stop}\) | disengagement value |
| \(P\) | progress |
| \(S\) | success evidence |

No neural optimization may alter these definitions.

---

## 13. Dataset size

Target:

\[
2{,}000-4{,}000
\]

semantic conditions.

Prioritize balanced dissociation cells over raw size.

Each major contrast family should contain at least:

\[
200
\]

high-quality matched contrasts where feasible.

Include several task families rather than one task only.

---

## 14. Experimental generation

Use the existing experimental grammar to generate a large legal candidate pool.

Recommended:

\[
N_{\text{candidate}}\ge 50{,}000.
\]

Score candidates for abstraction-level disagreement.

---

## 15. Identifiability acquisition score

For candidate pair \(x_i,x_j\), compute standardized differences:

\[
\Delta O,
\quad
\Delta O^\*,
\quad
\Delta H,
\quad
\Delta E.
\]

Prefer pairs where one candidate variable changes while others remain matched.

For example:

\[
Score_{O^\*|O}
=
|\Delta O^\*|
-
\lambda_O|\Delta O|
-
\lambda_H|\Delta H|.
\]

Similarly define scores for other dissociations.

Select approximately equal numbers from each contrast family.

---

## 16. Optional Bayesian active selection

After an initial balanced identifiability sample, use Bayesian experimental design to select additional conditions expected to discriminate among:

\[
H1-H5.
\]

The posterior should be over abstraction-level hypotheses, not merely behavioral parameters.

Conceptually:

\[
x^\*
=
\arg\max_x
I(
Y_x;
M_{\text{abstraction}}
\mid
\mathcal D
).
\]

Reserve at least:

\[
20\%
\]

of the dataset as coverage/random validation.

---

## 17. Freeze current DAS representation

The first analysis must use the **existing frozen Level-4 DAS subspaces**.

Do not retrain them initially.

This asks:

> What abstraction level does the already discovered causal subspace naturally correspond to?

The current best candidate is the frozen rank-2 layer-28 outcome-history/dual-history DAS solution with corrected test:

\[
CFR_G=.968.
\]

---

## 18. Natural-state representational equivalence

Before intervention, ask whether natural activations respect computational equivalence classes.

For frozen DAS projection:

\[
z_S=P_Sh.
\]

For each contrast family calculate:

\[
\|\Delta z_S\|.
\]

Then ask which candidate computational difference best predicts neural distance:

\[
\Delta O,
\quad
\Delta O^\*,
\quad
\Delta H,
\quad
\Delta E.
\]

---

## 19. Representational model comparison

Fit on training data:

\[
\|\Delta z_S\|
\sim
|\Delta O|
\]

versus:

\[
\|\Delta z_S\|
\sim
|\Delta O^\*|
\]

versus:

\[
\|\Delta z_S\|
\sim
|\Delta H|
\]

versus:

\[
\|\Delta z_S\|
\sim
|\Delta E|.
\]

Evaluate on held-out pairs.

Primary metric:

\[
R^2_{\text{pair}}.
\]

This is diagnostic evidence only.

Causal intervention remains primary.

---

## 20. Causal interchange tests

For each contrast pair, perform the same frozen DAS interchange:

\[
z_{base,S}
\leftarrow
z_{source,S}.
\]

Measure:

\[
\Delta D^{neural}.
\]

Every candidate abstraction level makes a distinct counterfactual prediction.

Calculate:

\[
\Delta D_O^{CF},
\]

\[
\Delta D_{O^\*}^{CF},
\]

\[
\Delta D_H^{CF},
\]

\[
\Delta D_E^{CF}.
\]

---

## 21. Same neural intervention, competing theories

This is a critical rule.

For a given:

\[
\Delta D^{neural},
\]

score the **same intervention** against every candidate abstraction.

Do not retrain DAS separately.

Calculate:

\[
MSE_M
=
\frac1N
\sum_i
(
\Delta D_i^{neural}
-
\Delta D_{i,M}^{CF}
)^2
\]

for:

\[
M\in\{O,O^\*,H,E\}.
\]

---

## 22. Primary abstraction-level statistic

For candidate abstraction \(M\), calculate:

\[
CFR_G(M)
=
1-
\frac{
\sum_i
(
\Delta D_i^{neural}
-
\Delta D_{i,M}^{CF}
)^2
}{
\sum_i
(
\Delta D_i^{base}
-
\Delta D_{i,M}^{CF}
)^2
}.
\]

Compare candidate abstraction levels using paired bootstrap differences.

---

## 23. Contrast-specific predictions

Create an explicit prediction matrix.

| Contrast | Raw \(O\) | Context \(O^\*\) | History contribution \(H\) | Persistence evidence \(E\) |
|---|---:|---:|---:|---:|
| Same O, different O* | no change | change | maybe | maybe |
| Different O, same O* | change | no change | maybe | maybe |
| Different histories, same H | change | possible change | no change | no change if E matched |
| Same H, different E | possible | possible | no change | change |
| Different causes, same E | likely change | likely change | likely change | no change |

This table must be frozen before analysis.

---

## 24. Counterfactual factorial ANOVA/regression

Use the engineered contrasts to estimate:

\[
\Delta z_S
=
\gamma_O\Delta O
+
\gamma_{O^\*}\Delta O^\*
+
\gamma_H\Delta H
+
\gamma_E\Delta E
+
\epsilon.
\]

Because the experimental design deliberately reduces collinearity, coefficients become more interpretable.

Report:

- standardized coefficients;
- partial \(R^2\);
- bootstrap CIs.

This supplements the direct theory comparison.

---

## 25. Causal convergence test

For same-\(E\), different-cause conditions, ask whether neural projections converge:

\[
E_i\approx E_j
\]

but:

\[
O_i,O_j,H_i,H_j
\]

differ strongly.

Define:

\[
CE
=
1-
\frac{
d(S_i,S_j)_{\text{same }E}
}{
d(S_i,S_j)_{\text{random matched}}
}.
\]

Large positive convergence supports a downstream integrated decision state.

---

## 26. Causal divergence test

For conditions with the same upstream history but different total persistence evidence:

\[
O_i\approx O_j,
\quad
H_i\approx H_j,
\]

but:

\[
E_i\neq E_j,
\]

ask whether:

\[
S_i\neq S_j.
\]

This directly tests whether the subspace sits downstream of history integration.

---

## 27. Shuffled-target problem

The previous best DAS candidate did **not** beat the shuffled-target control despite very high test CFR.

The new dissociation dataset must therefore deliberately increase the variance and uniqueness of counterfactual predictions.

Require:

\[
SD(\Delta D^{CF})
\]

substantially greater than in the existing mechanistic test set.

Avoid test sets where almost all effects lie in:

\[
+.8\text{ to }+1.5.
\]

Include:

- negative effects;
- near-zero effects;
- moderate effects;
- large positive effects.

---

## 28. Balanced causal effect distribution

Target approximately:

```text
20% strong negative
20% moderate negative
20% near zero
20% moderate positive
20% strong positive
```

within each relevant theory where feasible.

This makes shuffled-target controls genuinely destructive.

---

## 29. New shuffled-target gate

Within task and approximate effect-magnitude strata, permute source-target relationships.

Require:

\[
CFR_G^{true}
>
CFR_G^{shuffle}
\]

with paired-bootstrap 95% CI excluding zero.

If this still fails on the high-identifiability dataset, interpret the neural state as generic controller rather than variable-specific computation.

---

## 30. Compare against persistence-output state

Continue to include:

\[
S_D
\]

and direct output-gradient controls.

But now evaluate them on the same factorial dissociation dataset.

A generic persistence controller should:

- track \(E\);
- ignore which upstream variable produced \(E\).

This provides a benchmark for H4.

---

## 31. Train a downstream-integrated DAS candidate only secondarily

After evaluating the existing frozen subspace, optionally train new DAS models directly on:

\[
H
\]

and:

\[
E.
\]

Call them:

\[
S_H
\]

and:

\[
S_E.
\]

These are secondary analyses.

They answer:

> Can we explicitly learn cleaner neural abstractions at the downstream levels suggested by the frozen-subspace analysis?

---

## 32. Cross-abstraction matrix

Construct:

| Neural subspace | \(do(O)\) | \(do(O^\*)\) | \(do(H)\) | \(do(E)\) |
|---|---:|---:|---:|---:|
| Frozen existing DAS | ? | ? | ? | ? |
| \(S_O\) | ? | ? | ? | ? |
| \(S_{O^\*}\) | ? | ? | ? | ? |
| \(S_H\) | ? | ? | ? | ? |
| \(S_E\) | ? | ? | ? | ? |
| Persistence control | ? | ? | ? | ? |

Cells contain stable global CFR.

---

## 33. Task-general versus task-specific abstraction

For the winning abstraction level, test:

### Shared coordinates

\[
S_{\tau}=S.
\]

### Task-specific coordinates

\[
S_\tau.
\]

But require the same high-level variable semantics.

Possible result:

\[
\boxed{
\text{shared causal abstraction}
+
\text{task-specific neural realization}.
}
\]

Do not equate failed coordinate transfer with failed computational generality.

---

## 34. Depth analysis

Repeat the abstraction-level analysis at a small set of preselected layers.

Recommended:

\[
L16,L20,L24,L28,L30.
\]

Do not run another unrestricted fishing expedition.

The corrected localization suggests causal history information rises sharply in this part of the network.

---

## 35. Transformation-through-depth hypothesis

One scientifically strong outcome would be:

\[
L16:
O
\]

\[
\downarrow
\]

\[
L20:
O^\*
\]

\[
\downarrow
\]

\[
L24:
H
\]

\[
\downarrow
\]

\[
L28:
E.
\]

Do not assume this sequence.

Test it.

This would provide an actual **computational trajectory** rather than one isolated “direction.”

---

## 36. Layerwise abstraction score

For each layer \(l\) and abstraction \(M\), compute held-out:

\[
CFR_G(l,M)
\]

and:

\[
R^2_{\text{natural}}(l,M).
\]

Plot:

\[
\text{layer}
\times
\text{abstraction level}.
\]

Look for systematic transitions rather than isolated maxima.

---

## 37. Theory refinement criterion

Mechanistic evidence may revise the interpretation of the cognitive model.

For example:

Behavioral fit may use:

\[
O+A
\]

as separate regressors because this gives good prediction.

Mechanistically, the network may represent:

\[
H=\beta_OO+\beta_AA.
\]

Then the refined computational theory becomes:

\[
O,A
\rightarrow
H
\rightarrow
D.
\]

This is a scientific refinement, not a failure of the original behavioral approach.

---

## 38. Evidence hierarchy for this PRD

### Evidence A

Computational variables predict behavior.

### Evidence B

A neural subspace reproduces their downstream counterfactuals.

Already established at Level 4.

### Evidence C

Factorial dissociations identify the abstraction level represented.

Primary goal of this PRD.

### Evidence D

The abstraction transforms systematically through layers.

Strong mechanistic result.

### Evidence E

Specific components construct and transmit those transformations.

Future circuit stage.

---

## 39. Decision rules

### Outcome A — Raw variable correspondence

Dissociation tests favor:

\[
O.
\]

Conclusion:

> **The neural state corresponds closely to raw outcome-history information.**

### Outcome B — Contextual variable correspondence

Tests favor:

\[
O^\*.
\]

Conclusion:

> **The model transforms raw history according to contextual relevance before the identified causal stage.**

### Outcome C — Integrated history correspondence

Tests favor:

\[
H.
\]

Conclusion:

> **The neural state does not preserve the behavioral model's individual history terms as the relevant causal variable; it represents their integrated contribution to persistence.**

### Outcome D — General decision-evidence correspondence

Tests favor:

\[
E.
\]

Conclusion:

> **History and other task evidence converge on a shared persistence-evidence state before the final response.**

### Outcome E — No candidate abstraction passes

Shuffled targets or generic controls remain competitive.

Conclusion:

> **The current cognitive decomposition does not uniquely characterize the discovered neural controller.**

Return to computational-theory discovery.

---

## 40. Required figures

### Figure 1 — Experimental identifiability

Correlation matrix among:

\[
O,O^\*,H,E.
\]

Show that new dataset successfully reduces collinearity.

### Figure 2 — Causal dissociation grid

Plot neural-state differences for all six contrast families.

### Figure 3 — Frozen DAS versus abstraction levels

Same neural intervention scored against:

\[
O,O^\*,H,E.
\]

### Figure 4 — Cross-abstraction CFR matrix

Neural subspace × computational intervention.

### Figure 5 — Same-effect/different-cause convergence

Show neural distances for conditions matched on \(E\) but generated by different causal routes.

### Figure 6 — Abstraction through depth

Heatmap:

\[
\text{layer}
\times
\{O,O^\*,H,E\}
\]

with held-out CFR.

### Figure 7 — Shuffled-target validation

True mapping versus shuffled-target CFR on the high-identifiability test set.

---

## 41. Automated report questions

The report must answer:

1. Did the new dataset successfully decorrelate \(O,O^\*,H,E\)?
2. Does the existing frozen DAS subspace distinguish same-\(O\)/different-\(O^\*\) conditions?
3. Does it distinguish different-\(O\)/same-\(O^\*\) conditions?
4. Does it collapse physically different histories with equal \(H\)?
5. Does it collapse different causal routes with equal \(E\)?
6. Which abstraction best predicts natural-state neural geometry?
7. Which abstraction best predicts frozen DAS interventions?
8. Does the best abstraction beat shuffled targets?
9. Does it beat persistence/output controls?
10. Does abstraction level change systematically through depth?
11. Is the same abstraction shared across tasks?
12. Are neural coordinates shared across tasks?
13. Which computational decomposition is best supported?
14. How should the behavioral theory be refined?
15. What mechanistic claim is justified?

---

## 42. TDD requirements

Maintain:

\[
\text{RED}
\rightarrow
\text{GREEN}
\rightarrow
\text{REFACTOR}.
\]

### Matching tests

For each contrast family, verify required equalities/inequalities.

Example:

```text
same_raw_different_context:
    abs(delta_O) < tolerance
    abs(delta_O_star) > minimum
```

### Identifiability tests

Synthetic candidate pool must recover intentionally constructed:

- same-\(O\)/different-\(O^\*\);
- same-\(H\)/different-\(E\);
- same-\(E\)/different-cause

pairs.

### Counterfactual tests

Every abstraction's prediction must be generated before neural intervention.

No neural results may update computational targets.

### Frozen-subspace tests

Existing DAS rotation and rank must reproduce prior hashes exactly.

The primary analysis performs no retraining.

### Shuffle tests

Shuffled-target mapping must preserve marginal distributions while breaking example-level correspondence.

### Synthetic recovery

Generate known systems implementing:

1. raw history;
2. contextual history;
3. integrated history;
4. general persistence evidence.

The pipeline should identify the correct abstraction in each synthetic system.

---

## 43. Artifact structure

```text
artifacts/abstraction_discovery_v1/
    behavioral/
        frozen_models.json
        abstraction_variables.parquet

    design/
        candidate_pool.parquet
        contrast_manifest.parquet
        identifiability_metrics.csv

    frozen_das/
        manifest.json
        natural_geometry.csv
        interchange_results.parquet

    abstraction_tests/
        same_raw_context.csv
        same_context_raw.csv
        same_history_contribution.csv
        same_total_evidence.csv
        same_effect_different_cause.csv

    depth/
        abstraction_by_layer.csv

    controls/
        shuffled_target.csv
        persistence_control.csv
        random_subspaces.csv

    figures/
    gates.json
    report.md
    run_metadata.json
```

No full activation bank.

---

## 44. Stop rule

Do not proceed to circuit localization merely because one subspace has high counterfactual recovery.

Proceed only once the project can make a defensible statement of the form:

> **The causal neural state corresponds most closely to computational abstraction \(M\), and targeted dissociation experiments rule out relevant neighboring abstraction levels.**

If no abstraction wins, that itself is a result.

---

## 45. Scientific endpoint

The strongest possible result is not necessarily:

\[
\boxed{
\text{behavioral variable}
=
\text{neural variable}.
}
\]

It is:

\[
\boxed{
\text{behavioral cognitive models}
\rightarrow
\text{candidate causal decompositions}
\rightarrow
\text{targeted neural dissociations}
\rightarrow
\text{actual computational decomposition}.
}
\]

For example, the final result might be:

\[
O_t
\rightarrow
O_t^\*
\rightarrow
H_t
\rightarrow
E_t
\rightarrow
D_t,
\]

with different stages appearing at different transformer depths.

Or the model may collapse several of these stages.

Either outcome advances the original research program.

The governing claim should therefore be:

> **Cognitive models are used not merely to search for neural correlates of their variables, but to generate experimentally discriminable hypotheses about the causal computational decomposition implemented by the model.**

That is the hypothesis this PRD is designed to test.
