# Computational-cognitive discovery: revised framing

The paper demonstrates a strategy for using cognitive models to specify quantitative targets for mechanistic discovery. Persistence is the case study, rather than the presumed existence of a single universal persistence direction.

## Main scientific arc

1. Motivate the discovery problem from goal pursuit, human persistence research, and the original Sisyphus dissociation between reward decoding and causal control.
2. Define how behavioral models generate computational counterfactuals, then fit neural transformations to those targets. DAS is the existing search method; the application of cognitive models supplies the scientific constraints.
3. Present the heterogeneous task battery and model comparison, including shared form with task-specific calibration and unresolved behavioral alternatives.
4. Present held-out controller recovery, task transfer, five-seed optimization stability, and the fixed-layer/rank baseline comparison.
5. Interpret integrated persistence evidence E as a candidate, then state the identification and fresh-counterfactual limitations.
6. Present the frozen unrestricted-generation test as a further boundary.

## Claim boundaries imposed by the new experiments

| Proposed framing | Evidence-aligned wording |
|---|---|
| Cognitive models discover natural causal mechanisms | Cognitive models guide discovery of neural interventions that reproduce held-out computational effects; natural implementation and necessity remain unresolved. |
| The state is best understood as E | E is a motivated downstream interpretation, but its abstraction test fails matched-random specificity (p = .667). |
| The mechanism generalizes throughout structured tasks | Strong recovery transfers across selected structured contrast sets; fresh signed-history calibration is weak even within that task family. |
| Cognitive discovery outperforms probe-and-steer | It supplies a quantitative transformation test beyond decoding. Fixed-layer/rank comparisons favor cognitive DAS on original contrasts; equal-budget discovery superiority is untested and fresh contrasts do not preserve that advantage. |
| The OOD null proves it is not a universal persistence variable | The chosen frozen intervention does not establish generalization to EOS stopping; this does not exclude all universal mechanisms. |

## Main text versus supplement

The main text leads with method and positive discovery results. It retains concise accounts of the E identification failure, fresh signed-history calibration failure, and EOS null because those results change the central conclusions.

The supplement contains detailed original probe statistics, sequential-study methods, alternative-theory/recovery diagnostics, the constant-target shuffle audit, unexecuted necessity-gate clarification, full new experiment protocol, and the complete baseline table. The original intro and verified literature are retained as intellectual foundations rather than presented as a research diary.

## Completed replication and current work

The broader Qwen test separates weak frozen cognitive-target calibration from stronger natural-effect recovery, initially measured as a secondary diagnostic. The independent Qwen confirmation prospectively tests this endpoint with limited wording variation, generic controls, and 99 random subspaces. Its full results must be retained without retuning.

Llama response validation is now complete. A new Yes/No interface and an independent behavioral entry test pass the declared criteria. The neural replication achieves cognitive-target recovery of 0.649 within discovery tasks but only 0.011 on neural task holdouts. Generic controls perform better on natural effects. This is partial replication with a transfer boundary, not evidence for a universally shared E state.

No further experiments are being launched. The priority is a coherent manuscript that separates these evidential levels and reports both successful and failed tests. Full-search superiority remains untested and is not claimed.
