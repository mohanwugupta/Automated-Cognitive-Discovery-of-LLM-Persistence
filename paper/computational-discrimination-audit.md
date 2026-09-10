# What the existing contrasts can distinguish

9 September 2026. This is a descriptive post hoc audit of saved data, not a new model run or a replacement for the original analysis plan.

Run `python scripts/audit_computational_discrimination.py` from the repository root. Outputs and SHA-256 source hashes are in `generated/computational_discrimination/`.

## Findings

- The design has 1,400 semantic contrasts / 2,800 rendered pairs. Its final split has 280 contrasts / 560 pairs. All declared final pairs are present for the primary controller. The other 80% belong to train/validation partitions, not missing final evaluations.
- Contrast partitions share base/source states: 260 of the 450 final-test states also appear in the design training partition; 166 appear in validation. The primary basis was frozen before this abstraction study, so this is not evidence of training leakage into that basis. It does mean that these partitions cannot be described as state-disjoint, and shared states matter for uncertainty.
- The design's maximum candidate-variable delta correlation of 0.849 does not describe correlations between predicted decision effects on evaluated pairs. Raw-history O and history-contribution H output predictions correlate at **0.975** there. Of 152 rendered pairs where both exceed 0.1 logit in magnitude, none predict opposite signs. This limits separation of those two accounts.
- Context-relevant O* versus H does disagree in sign on 64 of 150 active rendered pairs. The existing battery contains some substantive computational disagreements; it is not uniformly uninformative.
- H and E predictions are equal to absolute tolerance 1e-10 on **54.6%** of evaluated rendered pairs. D is defined as E, so no experiment using those identical definitions can distinguish E from D.
- **92.5%** of the aggregate E score's squared-target denominator comes from the same-history/different-evidence family. That family achieves E recovery 0.795. Aggregate E recovery 0.707 therefore does not imply uniformly accurate reproduction of all dissociations.
- The same-total-evidence family has near-zero E targets but intervention-to-target RMSE **0.718 logits**. Its normalized recovery is unstable because its target energy is small; absolute error is the interpretable diagnostic.
- The original random controls evaluate five semantic pairs per family, selected by manifest order. The candidate is compared on those same subsets. This is not a full-pair random comparison; the original random-control gate still fails (p=0.667).

The 0.1-logit activity threshold above is descriptive and was chosen for this audit. No p-values are generated from these post hoc subsets. Response mappings are rendered repeats, not independent semantic observations.

## Consequences

The independent Qwen confirmation remains positive for natural-effect recovery. These findings concern the separate E abstraction interpretation. Its current evidence does not identify an upstream integration algorithm, and repeating its pooled score is unlikely to resolve that issue.

The next design needs disagreement in **predicted outputs under mapped interventions**, rather than only decorrelated scalar features. It should report equal-weight family errors and near-zero-target absolute errors, use state-disjoint groups, and compare every control on the same complete final set. See `computational-discrimination-protocol.md`.
