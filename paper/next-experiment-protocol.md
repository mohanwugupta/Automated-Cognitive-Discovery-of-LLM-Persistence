# Prospective confirmation and response-interface validation

Frozen before new-model inference on this design. This is a follow-up informed by previous outcomes, not an original-study preregistration. Preserve every earlier result and failure.

## Qwen confirmation

- Pin Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; freeze the primary dual-history L28/rank-2 controller. No neural fitting or selection on this test.
- Seed 92001 generates four new backgrounds per task. Two signed history changes use lengths three and five, both response mappings, and two deterministic wording variants (original and rephrased/reordered presentation). Seven tasks, 56 semantic counterfactual pairs, rendered under both mappings (112 pairs per wording), 224 rendered evaluation pairs. Retain all tasks and both signs.
- Fit zero-intercept global and task-specific gains from the **previous** fresh-context natural effects only. Task gains shrink toward the global gain with fixed penalty 10. No test-based choice between calibrators: report original, global, and task gains. These are quantitative recalibrations, not new cognitive architectures. Calibration on all tasks means these revised targets are context-held-out, not task-unseen.
- Primary neural endpoint: normalized squared-error recovery of independently measured natural source-minus-base effects. Primary comparisons pool both wordings, reported separately on familiar and previously held-out tasks. Report per-wording and per-task diagnostics too.
- Require positive recovery with background-cluster bootstrap lower bound above zero and superiority to 99 new layer/rank/norm-matched random subspaces for a confirmatory recovery claim. Apply Holm correction to the two task-group random comparisons. No claim of specificity unless the frozen controller also exceeds the generic controls; bootstrap paired score differences.
- Controls: frozen ridge-persistence and direct-behavior DAS bases from the original training data, train PCA, and the direct output-embedding contrast completed to rank two with frozen train PCA. No control fits on new evaluation data. Match achieved intervention norms as closely as finite precision permits and retain the audit.
- Secondary necessity-related diagnostic: replace the primary subspace with the fixed original-training mean, for every test source/base, compare attenuation of the natural pair effect. This is a perturbation diagnostic, not proof of a uniquely necessary natural computation. Do not pool ablation with source-interchange recovery.
- 2,000 bootstrap resamples of task-stratified background clusters retaining wording/history/mapping dependence. Report all outcomes, exact model/runtime, prompt hashes, calibration coefficients, and raw interventions. New contexts are checked against the prior manifest before inference. No failure-driven retuning in this run.

## Llama response interface

- Pin approved Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Test exactly three Yes/No interfaces: a question, a proposed-decision statement, and an explicit action instruction. Both polarities ask complementary semantic decisions. Retain the original task context verbatim. This avoids arbitrary X/Y labels but is a new measurement interface.
- Calibration: 210 semantic situations (30/task), seed 91001, both polarities per format. Select only among formats passing **every inherited gate on every task**; minimize the worst-task gap, lexical tie-break.
- Independent validation: 490 semantic situations (70/task), seed 91002. Evaluate one selected winner once. No threshold relaxation, re-selection, or dropping failed tasks. If validation fails or no format qualifies, stop replication and report the barrier.
- A successful format permits a separate frozen Llama behavioral-model/discovery protocol; it does not itself establish replication. Never train on the reserved validity set.

## Publication scope

Keep the stronger equal-budget discovery-superiority claim out of the paper: this protocol tests a frozen controller, not the full discovery procedure. Complete an independent code/artifact audit of the original sequential study and prepare an anonymous review package. Author verification, OpenReview identity/profile details, and reciprocal-review eligibility require real author information, not assumptions.

Terminology correction during execution: the Qwen design has 56 semantic pairs per wording, each rendered under two response mappings (112 rendered pairs per wording, 224 total). The original prose called these "112 semantic ... across mappings"; the manifest, executable counts, and selection rules are unchanged. The executed protocol retains the hash of the pre-correction text.
