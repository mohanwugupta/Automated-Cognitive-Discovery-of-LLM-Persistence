# OOD free-generation persistence report

## Frozen protocol

1. **DAS hash verified:** yes (`ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`).
2. **E orientation derived only from pre-OOD data:** yes, from 2368 structured activation rows.
3. **Layer/rank/dose tuned on free generation:** no.
4. **Application output caps removed:** yes; stopping was EOS, context capacity, or separately labeled infrastructure timeout.
5. **Actual model context capacity:** 262144 tokens.
6. **Retained cache matches fresh recurrent replay:** True.
7. **Native cached versus full-sequence/no-cache equivalence:** True (disposition: `passed`).

## Primary results

8. **EOS terminations:** 1500.
9. **Context-censored generations:** 0.
10. **Did increasing E lower EOS hazard?** False (Cox beta=-0.0137808, 95% CI [-0.0500685, 0.0225069]).
11. **Monotonic dose response:** False (Spearman alpha–RMST=0.100).
12. **Generation-duration change:** endpoint restricted-mean change=8.860 decision steps.
13. **Prompt wording generalization:** True; 4/6 open-ended prompts had negative hazard coefficients.
14. **Fixed-topic generalization:** True (beta=-0.0299403).
15. **Random-subspace specificity:** False (p=0.257426; n=100).
16. **Direct EOS benchmark:** 90 direct-EOS benchmark generations were analyzed; at the largest positive dose their median length was 368.5 tokens and severe-degeneration fraction was 0.000, versus 312.0 and 0.000 for frozen E.
17. **Degeneration:** True; maximum severe-degeneration fraction=0.006.
18. **Single-pulse persistence:** beta=-0.0303035; continuous steering remains primary.

## Conclusion

19. **Strongest justified claim:** The frozen structured-task state did not satisfy all preregistered OOD generalization gates.

All outcomes were analyzed under the frozen analysis plan, including negative outcomes. A native Qwen3.5 cache/no-cache discrepancy remains a failed strict protocol gate even when fresh recurrent cache replay passes; it is not reclassified as equivalence. No OOD result was used to redefine the subspace, direction, layer, rank, dose, prompt set, decoding policy, or analysis.
