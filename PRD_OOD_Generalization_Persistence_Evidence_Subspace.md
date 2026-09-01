# PRD — Out-of-Distribution Generalization of the Persistence-Evidence Subspace

**Project:** Computational Cognitive Discovery for AI Systems  
**Model:** Qwen 4B model used in the persistence experiments  
**Stage:** Final out-of-distribution causal validation  
**Primary question:** Does the frozen causal subspace discovered from structured persistence tasks regulate a qualitatively different form of persistence: **how long the model voluntarily continues free-form generation before terminating?**

---

## 1. Scientific motivation

The persistence project began by using cognitive models to characterize why an LLM continues versus disengages across structured sequential tasks.

The behavioral and mechanistic results now support a conservative working interpretation:

\[
\text{cognitive variables}
\rightarrow
\boxed{E}
\rightarrow
\text{continue / stop}
\]

where \(E\) is a downstream amalgamation of the computational variables identified through behavioral modeling rather than a clean neural representation of any individual cognitive variable.

The important remaining question is whether this causal state is merely shared across the structured task battery or reflects a substantially more general mechanism governing continued engagement.

This experiment deliberately moves **outside the discovery task space**.

The model will receive open-ended prompts such as:

> Write about anything you want.

There is:

- no explicit reward;
- no bandit;
- no patch;
- no solvability judgment;
- no externally defined effort requirement;
- no explicit CONTINUE/STOP labels.

The model itself determines when generation ends by emitting its EOS token.

The central hypothesis is:

\[
\boxed{
\text{If }E\text{ is a task-general persistence-control state, manipulating it should alter voluntary generation duration.}
}
\]

---

## 2. Primary prediction

At generation step \(t\), free-form generation implicitly contains the decision:

\[
\text{produce another token}
\quad\text{vs.}\quad
\text{emit EOS}.
\]

Let positive intervention correspond to increasing the previously frozen persistence-evidence state.

Then preregister:

\[
\alpha<0
\Rightarrow
\text{higher termination hazard}
\Rightarrow
\text{shorter generation}
\]

and

\[
\alpha>0
\Rightarrow
\text{lower termination hazard}
\Rightarrow
\text{longer generation}.
\]

The strongest result is a monotonic dose response:

\[
-E
<
\text{baseline}
<
+E
\]

in generation length.

---

## 3. Hard constraint: no rediscovery

This experiment is a **generalization test**, not another mechanistic search.

Before collecting any free-generation results, freeze:

- intervention subspace;
- layer;
- rank;
- orientation;
- steering scale;
- prompt set;
- decoding parameters;
- primary outcomes;
- exclusion rules;
- statistical analyses.

Do **not**:

- search layers;
- retrain DAS;
- optimize a new steering vector on free-generation data;
- choose doses based on observed length effects;
- select favorable prompts;
- redefine the persistence direction after seeing OOD results.

The primary test must use the previously discovered causal representation unchanged except for deriving a signed axis **within the already frozen causal subspace**, as described below.

---

## 4. Frozen neural object

Use the already validated causal subspace:

\[
S_E
\subset \mathbb{R}^{d}
\]

with its previously selected:

- transformer layer;
- rank;
- DAS rotation/subspace;
- activation position conventions.

For the current primary candidate this is expected to be approximately:

\[
L=28,\qquad k=2.
\]

The implementation must load this object from existing artifacts and hash-verify it.

No training occurs in this experiment.

---

## 5. Define the signed \(E\) axis inside the frozen subspace

DAS provides a rank-\(k\) subspace, not necessarily a single signed steering vector.

We therefore need a fixed mapping from DAS coordinates to increasing versus decreasing \(E\).

Let:

\[
z=P_Sh
\]

be the activation projected into the frozen causal subspace.

Using **only the original structured persistence dataset**, fit:

\[
E_i
=
\beta_0+w_E^\top z_i+\epsilon_i.
\]

Define:

\[
d_E
=
P_S^\top
\frac{w_E}{\|w_E\|}.
\]

Orient its sign so that:

\[
d_E^\top h
\uparrow
\]

corresponds to greater frozen behavioral-model \(E\).

### Important

This regression only chooses an orientation **inside the previously causally validated subspace**.

It is not permissible to fit:

\[
E\sim h
\]

over the full residual stream and call that the OOD steering direction.

That would recreate the earlier probe-steering problem.

---

## 6. Freeze steering scale before OOD testing

Define intervention in standardized \(E\)-subspace units.

Recommended dose set:

\[
\alpha\in\{-2,-1,0,+1,+2\}.
\]

Intervention:

\[
h'_{l,t}
=
h_{l,t}
+
\alpha\sigma_E d_E
\]

where \(\sigma_E\) is determined entirely from the original structured-task activation distribution.

Alternative if the existing DAS coordinates have a natural standardized scale:

\[
z'_t=z_t+\alpha d_E.
\]

Either approach is acceptable, but the exact definition must be frozen before free-generation inference.

---

## 7. Primary OOD task

Prompt:

> **Write about anything you want.**

The model should have complete freedom over:

- topic;
- structure;
- content;
- length;
- stopping time.

No requested minimum or maximum length should appear in the prompt.

---

## 8. Prompt-family replication

Do not make the entire claim depend on one wording.

Freeze a small open-ended prompt family, for example:

1. `Write about anything you want.`
2. `Write whatever you would like.`
3. `Choose any topic and write about it.`
4. `Say whatever comes to mind.`
5. `Write freely about a subject of your choice.`
6. `Tell me about anything you feel like discussing.`

These prompts should remain intentionally underspecified.

Do not include phrases such as:

- “be concise”;
- “give a detailed answer”;
- “keep writing”;
- “write at least...”;
- “in a few sentences.”

---

## 9. Remove application-level generation limits

### Goal

Allow the model's own EOS decision to determine generation length whenever technically possible.

There are two different limits:

### Removable

Application/API generation limits such as:

```python
max_new_tokens=512
```

or:

```python
max_length=1024
```

These should not determine ordinary stopping.

### Not removable

The model has a finite architectural context window.

Therefore truly unlimited generation is impossible:

\[
L_{\text{prompt}}+L_{\text{generated}}
\le
L_{\text{context}}.
\]

The experiment should remove **artificial output caps**, not pretend the context window is infinite.

---

## 10. Recommended decoding implementation: custom autoregressive loop

Do not rely on a library's low default `max_length`.

Implement a custom token-by-token decoding loop:

```text
encode prompt

while True:
    run model on current sequence
    intervene at frozen layer
    obtain next-token logits
    sample next token
    append token

    if token == EOS:
        termination_reason = "eos"
        break

    if context_capacity_exhausted:
        termination_reason = "context_limit"
        break
```

This makes the effective generation limit:

\[
L_{\max}
=
L_{\text{context}}
-
L_{\text{prompt}}
-
L_{\text{safety margin}}.
\]

Use a small safety margin only if required by the serving stack.

### Do not silently use

```python
max_new_tokens=2048
```

as the primary stop rule if the model can support substantially longer generation.

---

## 11. Detect model context capacity programmatically

Do not hardcode the context size unless the exact model configuration requires it.

Inspect the loaded model/tokenizer configuration for the appropriate maximum sequence length.

Conceptually:

```python
context_limit = resolve_context_limit(
    model.config,
    tokenizer.model_max_length
)
```

Then:

```python
available_tokens = context_limit - prompt_tokens - safety_margin
```

The run stops at context exhaustion only if EOS has not already occurred.

If the runtime/model implementation supports extending context via an officially supported configuration, that can be used **only if it is the same inference configuration used to discover the original DAS representation or is validated not to alter the relevant representation**.

Do not introduce an unvalidated RoPE/context-extension modification solely for this experiment.

---

## 12. Context-limit termination is censoring

If the model reaches its context boundary without EOS:

```text
termination_reason = context_limit
```

Do not treat:

\[
L=L_{\max}
\]

as the true stopping time.

Instead record:

\[
T\ge L_{\max}.
\]

These observations are **right-censored** in the primary length analysis.

---

## 13. Decoding configuration

Use a fixed stochastic decoding configuration.

Recommended starting point:

```text
do_sample: true
temperature: 1.0
top_p: 1.0
top_k: disabled
repetition_penalty: 1.0
```

The purpose is to avoid artificial truncation or continuation biases.

However, if the original model's canonical/default sampling policy differs, use a preregistered fixed configuration.

Do not tune decoding separately by intervention condition.

---

## 14. Random seeds

For each prompt, use matched random seeds across all intervention doses.

Example:

```text
prompt_01 × seed_001:
    alpha=-2
    alpha=-1
    alpha=0
    alpha=+1
    alpha=+2
```

The same sampling RNG initialization should be used across doses where technically feasible.

This substantially reduces stochastic variance.

---

## 15. Sample size

Recommended initial primary experiment:

\[
6\text{ prompts}
\times
100\text{ seeds}
\times
5\text{ doses}
=
3000\text{ generations}.
\]

If unconstrained outputs become extremely long and compute becomes prohibitive:

\[
50\text{ seeds/prompt}
\]

is an acceptable first preregistered stage.

Do not stop early because results look favorable.

---

## 16. Primary intervention regime: continuous

At **every autoregressive generation step**, intervene at the frozen layer:

\[
h'_{l,t}
=
h_{l,t}+\alpha d_E.
\]

Only modify the activation corresponding to the current final-token position used to predict the next token.

This most closely maps onto the structured persistence experiments:

\[
\text{each generation step}
=
\text{another opportunity to continue or terminate}.
\]

---

## 17. Secondary intervention regime: single pulse

Secondary experiment:

Apply intervention only on the first token-generation step:

\[
t=1.
\]

Then allow the model to run normally.

This distinguishes two possibilities.

### Reconstructed decision state

If continuous steering works but single-pulse steering does not:

\[
E_t
\]

is likely reconstructed at each decision.

### Persistent state induction

If a single pulse causes a sustained change in generation length:

\[
E
\]

may induce a longer-lived downstream state.

This analysis is secondary and must not replace the primary continuous-intervention result.

---

## 18. Primary dependent variable: EOS hazard

At every generated token \(t\), log:

\[
p_t(\mathrm{EOS}).
\]

Primary token-level DV:

\[
h_t=P(\mathrm{EOS}\mid\text{generation survived to }t).
\]

Also define continuation evidence:

\[
D_t^{gen}
=
\log
\frac{
1-p_t(\mathrm{EOS})
}{
p_t(\mathrm{EOS})
}.
\]

Primary mechanistic prediction:

\[
\alpha\uparrow
\Rightarrow
p_t(EOS)\downarrow.
\]

---

## 19. Primary behavioral DV: voluntary generation length

Record:

\[
T=
\text{number of generated tokens before EOS}.
\]

Also record word count for interpretability, but token count is primary.

Required variables:

```text
prompt_id
seed
alpha
generated_tokens
generated_words
eos_emitted
termination_reason
context_censored
```

---

## 20. Survival analysis

Because some generations may reach the context boundary, use survival analysis as the primary length statistic.

Event:

\[
\text{EOS emission}.
\]

Context exhaustion:

\[
\text{right censoring}.
\]

Recommended Cox model:

\[
h(t)
=
h_0(t)
\exp(
\beta_\alpha\alpha
+
u_{\text{prompt}}
).
\]

Primary directional hypothesis:

\[
\boxed{\beta_\alpha<0}.
\]

Increasing \(E\) should reduce EOS hazard.

---

## 21. Dose-response test

Test both:

### Linear trend

\[
T\sim\alpha
\]

or corresponding survival coefficient.

### Monotonic ordering

Require median length approximately:

\[
T_{-2}
<
T_{-1}
<
T_0
<
T_{+1}
<
T_{+2}.
\]

Perfect ordering is not required for every prompt/seed, but aggregate direction should be monotonic.

---

## 22. Immediate EOS-logit effect

At every generation step, save both:

\[
p^{base}(EOS)
\]

and:

\[
p^{steered}(EOS)
\]

where technically feasible.

Define:

\[
\Delta \logit(EOS)
=
\logit p^{steered}(EOS)
-
\logit p^{base}(EOS).
\]

Prediction:

\[
\alpha>0
\Rightarrow
\Delta\logit(EOS)<0.
\]

This gives a very clean proximal causal measure.

---

## 23. Critical control 1 — random matched subspace

Generate matched random rank-\(k\) subspaces at the same layer.

For example:

\[
N_{\text{random}}=100.
\]

Match neural intervention norm to the true \(E\) intervention.

For the expensive full-length generation test, it is not necessary to run all 100 random directions at all 3000 conditions.

Recommended two-stage approach:

### Neural control

Evaluate all random directions on a fixed set of generation states for immediate EOS-logit effects.

### Behavioral control

Select a preregistered random subset, e.g. 10 directions, for complete free-generation runs.

The true \(E\) intervention should exceed the random distribution.

---

## 24. Critical control 2 — direct EOS intervention

Construct a positive-control intervention that directly suppresses or enhances EOS output probability.

Purpose:

\[
\text{direct EOS hacking}
\]

is a benchmark for what trivial termination manipulation looks like.

Compare:

- length;
- EOS hazard;
- repetition;
- coherence;
- semantic quality.

A particularly useful finding would be:

\[
E\text{-steering}
\rightarrow
\text{longer coherent continuation}
\]

while:

\[
EOS\text{-suppression}
\rightarrow
\text{greater degeneration}.
\]

Do not require this difference for the primary generalization claim, but report it.

---

## 25. Critical control 3 — zero intervention

\[
\alpha=0
\]

must reproduce unmodified model behavior exactly, within expected stochastic reproducibility.

A hook that numerically modifies activations at \(\alpha=0\) is a test failure.

---

## 26. Output quality guardrails

Longer generation alone could result from pathological EOS suppression.

Measure:

### Repetition

- repeated 4-gram fraction;
- repeated 8-gram fraction;
- longest repeated substring where practical.

### Lexical diversity

\[
\frac{\text{unique tokens}}{\text{tokens}}
\]

with length-adjusted reporting.

### Degeneration

Detect:

- repeated sentences;
- endless punctuation;
- phrase loops;
- obvious template loops.

### Coherence

Optional blinded model judge:

> Is this continuation coherent and meaningfully progressing, rather than merely repeating or degenerating?

Judge must not know intervention condition.

---

## 27. Important secondary DV: semantic continuation

A more stringent definition of persistence is:

\[
\text{continued meaningful generation}
\]

rather than:

\[
\text{continued token production}.
\]

Optionally estimate:

\[
T_{\text{coherent}}
=
\text{tokens generated before detected degeneration or EOS}.
\]

Keep raw EOS-based length primary because this secondary measure introduces judgment/modeling assumptions.

---

## 28. Topic/content analysis

Because prompts are open ended, steering could accidentally alter topic instead of persistence.

Record embeddings or coarse topic classifications for generated texts.

Ask whether intervention systematically changes:

- topic choice;
- genre;
- affect;
- formatting.

This is secondary.

The central result should not be attributable solely to:

\[
+E\rightarrow\text{choose inherently verbose topics}.
\]

One useful analysis is conditional generation after the first sentence/topic has already been established.

---

## 29. Secondary OOD prompt set with fixed topic

To distinguish topic selection from persistence, include a secondary battery:

> `Write about the ocean.`

> `Write about mathematics.`

> `Write about trees.`

> `Write about music.`

These still contain no requested length.

Prediction should generalize:

\[
+E\rightarrow\text{longer generation}
\]

even when topic is fixed.

This battery remains OOD with respect to the structured persistence tasks.

---

## 30. Stronger continuation-specific variant

A further secondary experiment can give the model an initial paragraph and say only:

> `Continue.`

This removes topic-selection effects almost entirely.

Again:

\[
+E
\rightarrow
\text{lower EOS hazard / longer continuation}.
\]

This is useful but not the primary free-choice experiment.

---

## 31. Hardware/runtime guardrail

Uncapped generation can become expensive.

Do not introduce a hidden token cap merely to protect the runner.

Instead implement external **wall-clock / resource safeguards** separately from the scientific stopping rule.

If a process must be terminated for infrastructure reasons:

```text
termination_reason = infrastructure_timeout
```

Treat it as censored.

Do not recode it as EOS or maximum length.

---

## 32. KV cache

Use KV caching for efficient autoregressive decoding.

The activation hook must still correctly modify the residual state at the intervention layer for the newly generated token.

Test explicitly that:

- cached generation;
- uncached generation

produce equivalent logits at:

\[
\alpha=0.
\]

---

## 33. Intervention pseudocode

```python
for prompt in prompts:
    input_ids = tokenizer(prompt)

    for seed in seeds:
        for alpha in doses:

            reset_rng(seed)
            tokens = input_ids
            generated = []

            while has_context_capacity(tokens):

                outputs = forward_with_intervention(
                    tokens=tokens,
                    layer=FROZEN_LAYER,
                    direction=FROZEN_E_DIRECTION,
                    alpha=alpha,
                    position="final_token",
                    use_cache=True,
                )

                logits = outputs.next_token_logits

                log_eos_probability(...)

                token = sample(logits, frozen_sampling_config)

                if token == eos_token_id:
                    record_event("eos")
                    break

                generated.append(token)
                tokens.append(token)

            else:
                record_event("context_limit")

            save_generation(...)
```

Implementation can differ, but behavior must match this specification.

---

## 34. No EOS masking

Do **not**:

- remove EOS from the vocabulary;
- set EOS probability to zero;
- disable the EOS token;
- use a stopping implementation that ignores EOS.

EOS is the behavioral event being studied.

The goal is to remove external length constraints, **not the model's ability to choose to stop**.

---

## 35. Multiple EOS tokens

If the tokenizer/model defines multiple terminating special tokens, identify them before preregistration.

Record each separately, but primary termination event can be:

\[
EOS=\{\text{all canonical assistant-termination tokens}\}.
\]

Do not accidentally count a formatting token as voluntary model termination unless the model runtime normally treats it as such.

---

## 36. Chat template

Use the model's standard chat template exactly.

Example conceptual structure:

```text
system: standard/default system prompt
user: Write about anything you want.
assistant:
```

Avoid adding a system message instructing the model to be concise or verbose.

If a system prompt is required by the model template, freeze it.

---

## 37. Think/reasoning mode

If the Qwen model has configurable reasoning/thinking behavior, freeze the configuration before the experiment.

Do not mix:

- thinking enabled;
- thinking disabled

within the primary experiment.

If internal reasoning tokens are generated, decide beforehand whether:

\[
T
\]

counts:

1. all autoregressive tokens, or
2. only user-visible answer tokens.

For persistence of the actual model process, **all generated tokens** are the cleaner primary measure.

Visible output length can be secondary.

---

## 38. Main success criteria

Call the OOD generalization result **positive** only if all of the following hold.

### Gate A — Direction

Positive \(E\) intervention decreases EOS hazard:

\[
\beta_\alpha<0
\]

with CI excluding zero.

### Gate B — Dose response

Generation duration changes monotonically with \(\alpha\).

### Gate C — Prompt generalization

Effect is in the predicted direction for the majority of open-ended prompts and no single prompt drives the result.

### Gate D — Random specificity

True \(E\) intervention exceeds matched random subspace interventions.

### Gate E — Nondegeneration

Length effect is not primarily explained by severe repetitive degeneration.

### Gate F — Frozen analysis

No OOD data were used to choose:

- layer;
- direction;
- rank;
- intervention magnitude.

---

## 39. Interpretation matrix

### Outcome A — Strong OOD generalization

\[
+E
\rightarrow
\text{lower EOS hazard}
\rightarrow
\text{longer coherent generation}
\]

with random controls null.

Conclusion:

> **A causal neural state discovered using computational models of structured persistence generalizes to the model's autonomous decision about whether to continue free-form language generation.**

This would be the strongest result of the persistence program.

### Outcome B — Length effect but degeneration

\[
+E
\rightarrow
\text{longer generation}
\]

but mostly through looping/repetition.

Conclusion:

> The subspace regulates termination behavior, but evidence that it supports meaningful persistence is limited.

### Outcome C — EOS probability changes but total length does not

Conclusion:

> The subspace has a local causal effect on termination evidence, but downstream generation dynamics compensate.

Still informative.

### Outcome D — Structured-task only

No reliable free-generation effect.

Conclusion:

> The discovered causal state generalizes across the structured persistence battery but does not appear to govern unconstrained autoregressive persistence.

### Outcome E — random directions perform similarly

Conclusion:

> The free-generation intervention is insufficiently specific to support task-general persistence.

---

## 40. Required figures

### Figure 1 — Generation survival curves

Kaplan-Meier curves by:

\[
\alpha=-2,-1,0,+1,+2.
\]

Y-axis:

\[
P(\text{generation has not terminated by token }t).
\]

### Figure 2 — EOS hazard by dose

Plot estimated:

\[
P(EOS\mid t,\alpha).
\]

### Figure 3 — Length dose response

Median/estimated generation duration versus \(\alpha\).

### Figure 4 — Prompt-wise effects

Each prompt's:

\[
\beta_\alpha.
\]

### Figure 5 — Random-subspace null

True \(E\) effect marked against matched random directions.

### Figure 6 — Quality versus length

Show whether increased generation is associated with degeneration.

### Figure 7 — Continuous versus pulse intervention

Compare full intervention with first-step-only intervention.

---

## 41. Automated report questions

The final report must answer:

1. Was the frozen DAS object hash-verified?
2. Was the \(E\) orientation derived exclusively from pre-OOD data?
3. Was any layer/rank/dose tuned on free-generation results?
4. Were application-level output caps removed?
5. What was the actual model context capacity?
6. How many runs ended via EOS?
7. How many were right-censored at context capacity?
8. Did increasing \(E\) lower EOS hazard?
9. Was the dose response monotonic?
10. How much did median/expected generation duration change?
11. Did the effect generalize across prompt wording?
12. Did it generalize to fixed-topic prompts?
13. Did random subspaces reproduce the effect?
14. Did direct EOS suppression produce qualitatively different output?
15. Did increased persistence produce degeneration?
16. Did a single-pulse intervention produce lasting effects?
17. What is the strongest justified generalization claim?

---

## 42. TDD requirements

Follow:

\[
\boxed{\text{RED}\rightarrow\text{GREEN}\rightarrow\text{REFACTOR}}
\]

throughout implementation.

### Frozen artifact tests

Verify expected hash for:

- DAS rotation;
- subspace;
- layer;
- rank;
- \(E\)-orientation artifact.

### Direction sign test

On original persistence data:

```text
mean(E | high projection)
>
mean(E | low projection)
```

must hold.

### Alpha-zero equivalence

Require logits from:

```text
intervention(alpha=0)
```

to equal baseline logits within floating-point tolerance.

### Context-limit test

Construct a short artificial context limit and verify:

- EOS → event;
- capacity exhaustion → censoring.

### No hidden-cap test

Fail if primary generation configuration contains an arbitrary scientific stop such as:

```text
max_new_tokens = 512
```

unless that equals remaining architecture-supported context.

### EOS test

Verify EOS remains sampleable under all \(E\) doses.

### Cache-equivalence test

At \(\alpha=0\):

\[
logits_{\text{cache}}
\approx
logits_{\text{no-cache}}.
\]

### Hook-position test

Verify intervention modifies only the newly generated final-token activation at the frozen layer.

### Matched-seed test

Confirm all intervention conditions for a prompt share the intended RNG seed.

### Survival test

Synthetic censored dataset should recover known negative hazard coefficient for positive steering.

### Random-control test

Synthetic planted \(E\) direction must exceed random subspace null.

---

## 43. Artifact structure

```text
artifacts/ood_free_generation_v1/

    frozen/
        das_manifest.json
        e_orientation.pt
        hashes.json
        sampling_config.json
        prompt_manifest.json

    generations/
        generation_summary.parquet
        token_events.parquet

    controls/
        random_subspaces.csv
        eos_control.csv
        alpha_zero_checks.csv

    analysis/
        survival_model.json
        dose_response.csv
        prompt_effects.csv
        censoring_summary.csv
        quality_metrics.csv

    pulse/
        generation_summary.parquet
        survival_model.json

    figures/
        figure1_survival.png
        figure2_eos_hazard.png
        figure3_dose_response.png
        figure4_prompt_effects.png
        figure5_random_null.png
        figure6_quality_length.png
        figure7_pulse_vs_continuous.png

    gates.json
    run_metadata.json
    report.md
```

Do not retain full activation banks.

---

## 44. Git/storage requirements

Keep Git artifacts lightweight.

Save:

- frozen subspace;
- small direction tensors;
- generated text;
- token-level scalar statistics;
- analysis summaries.

Do not save:

- full residual streams;
- KV caches;
- per-layer activation banks.

Large temporary generation artifacts should go to scratch storage if needed.

---

## 45. Stop rule

Do not add additional tasks, search new layers, or retrain representations before evaluating this test.

This experiment is specifically valuable because it is a **hard frozen generalization test**.

The result should be accepted whether positive or negative.

---

## 46. Final scientific claims

### If positive

The appropriate claim is:

> **Computational modeling of persistence across structured tasks led to the discovery of a low-dimensional causal state that generalizes to voluntary termination during unrestricted language generation. Increasing this state lowers the model's tendency to terminate and extends generation, while decreasing it produces earlier disengagement.**

Do not claim that the state literally represents any one original cognitive variable.

A stronger conceptual description is:

\[
\boxed{
\text{cognitive variables}
\rightarrow
\text{downstream integrated persistence evidence }E
\rightarrow
\text{general continuation control}
}
\]

### If negative

The appropriate claim is:

> **The causal state discovered through structured persistence tasks generalizes within that experimental family but does not measurably control free-form autoregressive termination.**

Both outcomes sharply constrain the theory.

---

## Implementation note on “unlimited” Qwen output

Do **not** set a conventional `max_new_tokens` cap in the primary custom decoding loop.

Instead, let EOS terminate generation and calculate the only unavoidable ceiling dynamically as:

\[
\boxed{
\text{remaining capacity}
=
\text{model context capacity}
-
\text{prompt tokens}.
}
\]

That removes the experimenter's artificial output-length limit while respecting the model's finite context window.

Runs that consume the entire available context are censored rather than treated as voluntary stops.

This is scientifically cleaner than setting `max_new_tokens=100000`, and it prevents a hidden library default from contaminating the dependent variable.
