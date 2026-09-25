# Proposed computational-discrimination follow-up

Status: design specification, 9 September 2026. **Not run, not preregistered, and no GPU resources provisioned.** This protocol is informed by prior results and must be frozen with executable manifests before any new final evaluation. It is a proposal to test the stronger computational claim, not a prerequisite for reporting the completed narrower case study.

## Question and scope

Does an aligned neural representation implement a particular history computation, or does it mainly provide downstream decision control? A new experiment cannot distinguish E from D while the current code defines them identically. It must compare distinct candidate programs under interventions that make them disagree.

Use Qwen first. Replicate on Llama only if the behavioral and neural tests below provide interpretable evidence. Do not search additional layers or ranks after observing final results.

## Candidate high-level programs

Use one common, explicitly documented decision readout, with training-only task calibration, and compare:

1. Raw outcome integration: a recency-weighted trace over all supplied outcomes.
2. Context-relevant integration: traces for separate contexts and a cue-controlled selection or mixture.
3. Joint action/outcome integration: separate action and outcome traces contributing to the decision.
4. Downstream decision evidence: an output-level reference that predicts replacement of the integrated decision quantity.

All equations, fit parameters, missingness rules, and node definitions must be serialized. If learning rates are fitted, treat that as a new behavioral model rather than silently changing the historical fixed 0.7 trace. Include a flexible behavioral comparator. Behavior-target DAS is a mechanistic comparator, not itself a cognitive theory.

## Manipulations and mapped interventions

Vary history, current context, action repetition, and current incentive independently. Include the following four families with equal semantic counts:

- Hold raw outcomes fixed and change which context is relevant: raw integration and context-sensitive integration should disagree.
- Hold the context-relevant history fixed and change irrelevant outcomes: test invariance to irrelevant history.
- Hold outcomes and current incentives fixed and vary previous actions: separate action repetition from outcome integration.
- Match predicted total decision evidence while compensating history with incentives: distinguish upstream history replacement from downstream output replacement.

The final family needs a proper intermediate-node intervention. For a history trace T and current inputs c, swap T_s into c_b and recompute f(c_b,T_s). The downstream alternative swaps E_s into the base and predicts E_s-E_b, near zero for output-matched pairs. In contrast, f(c_b,T_s)-f(c_b,T_b) can be nonzero even if E_s equals E_b. Both programs must give predictions for the same neural patch and same base/source prompts. State clearly which scalar or vector node is aligned; replacing an entire input history is not automatically equivalent to replacing an intermediate trace.

## Stage A: CPU design and recovery checks

Create new semantic states with a recorded seed and exclude all historical states. Split by connected components of shared base/source states, keeping both mappings and wordings together. Reserve 40 independent semantic groups per family per split: 160 train, 160 selection, 160 final. Each has two action mappings and two wording variants, giving 640 rendered pairs per split. No component may cross a split.

Before model inference, simulate data from each candidate program and perform model/parameter recovery. If the proposed behavioral procedure cannot distinguish the known simulated programs, revise the design and record a new version before collecting data.

Require each hypothesis comparison to have at least 20 final semantic groups with an absolute predicted-effect difference of at least 0.25 logit under the fixed models. This is a design target, not a proven power calculation. Include near-zero and both effect signs. Do not select on observed neural effects. Inspect family-level target energy; report equal-weight family loss so a single high-magnitude family cannot determine the conclusion.

The exact candidate pool, counterfactual outputs, split IDs, model hashes, and recovery-check results must be produced before this stage can be called complete. They are not yet produced by this document.

## Stage B: behavioral measurement and calibration

Collect ordinary, unpatched decisions first. Validate action probability mass and mapping/wording sensitivity on the separate selection set using the established Qwen validity procedure. Fit the candidate programs on training data, retain alternatives that remain statistically unresolved, and freeze them before final behavioral evaluation.

For the final behavioral test, report held-out logit error and paired differences between theories with intervals clustered by independent semantic group. A candidate may proceed as an implementation hypothesis only if its counterfactual predictions improve over zero effect in each diagnostic family and there is evidence that the designed disagreements are behaviorally relevant. If all theories are poorly calibrated, report model misspecification and stop before causal search. Do not redefine the theory using this final set.

Because the operationalization can fail, no neural-run budget should be spent until this behavioral gate is assessed. A revised behavioral program requires fresh final states and a new protocol version.

## Stage C: controlled causal comparison

Fix Qwen layer 28 and rank 2 for the primary comparison. Compare the historical frozen controller, newly fitted alignments for retained cognitive programs, direct behavior-target DAS, ridge decision, output, PCA, and 99 random rank-2 subspaces. Give each newly trained alignment identical training pairs and optimization budgets. This tests a fixed configuration; it does not establish superiority of full discovery searches.

Use the same complete final pairs for every method. Compute basis orthogonality and achieved replacement norms in float32, even if model inference uses reduced precision. Require achieved control norms to be within 5% per pair and report absolute errors near zero norms. If control matching cannot satisfy the frozen numerical criterion, label the comparison invalid; do not filter outcome-dependent subsets. Keep raw results and numerical-failure counts.

Primary endpoint: equal-weight mean squared prediction error across the four families for each candidate abstraction, applied to the same interventions. Report paired loss differences between candidate programs with simultaneous or Holm-adjusted comparisons across the prespecified alternatives. Use 2,000 group-bootstrap samples and preserve all wordings, mappings, and repeated states. Report absolute baseline and intervened-output calibration in addition to effect differences. Report conventional CFR only as a secondary aggregate and per-family metric.

For necessity evidence, compare training-mean replacement against matched random and generic-control ablations on the complete final set. Include current-incentive effects as a specificity diagnostic. Attenuation alone is not selective necessity. Prespecify these analyses before inference; if omitted, do not make a necessity claim.

## Decision rules and resource boundaries

A computational interpretation needs reliable agreement on the distinguishing interventions, appropriate control comparisons, and natural behavioral calibration. Beating random subspaces alone is insufficient. If cognitive and output-level programs remain indistinguishable, report equivalence under the tested intervention family. If the neural method controls natural effects but misses cognitive predictions again, retain the controller-fidelity/calibration distinction as the main result.

Before execution, create an exact forward-pass count from the manifest, benchmark memory and time on training data only, set a hard dollar/runtime cap, and record a stop/download/termination procedure. No spend estimate or capacity reservation is asserted here. Final test data are never used for timing-based method selection. Llama and a new EOS study are outside this first follow-up.
