# Causal Abstraction Level of Persistence Computation

**Result:** No candidate abstraction uniquely characterizes the frozen persistence controller.

## Automated questions

1. The dataset decorrelated O/O*/H/E sufficiently: **yes** (maximum |r|=0.849).
2. Same-O/different-O* frozen-DAS distance: **0.621**.
3. Different-O/same-O* frozen-DAS distance: **0.000**.
4. Different histories with equal H converge: **yes** (distance=0.574).
5. Different causal routes with equal E converge: **yes** (CE=0.515).
6. Best natural-state geometry model: **E** (held-out R²=0.700).
7. Best frozen-DAS intervention abstraction: **E** (CFR_G=0.707, r=0.882).
8. It beats shuffled targets: **yes** (paired lower CI=0.656).
9. It beats persistence/output controls: **yes** (minimum lower CI=0.362).
10. Systematic abstraction-through-depth pattern: **yes** (L16=E, L20=E, L24=E, L28=E, L30=E).
11. The winning abstraction generalizes across tasks: **yes**.
12. Neural coordinates are shared across tasks: **supported**; no target-task refitting occurred.
13. Best-supported computational decomposition: **no_candidate_abstraction_passed**.
14. Behavioral-theory refinement: **No candidate abstraction uniquely characterizes the frozen persistence controller.**
15. Justified mechanistic claim: **Level-4 causal controller only; abstraction identity remains unresolved.**

## Guardrails

- Behavioral O/O*/H/E targets and the prediction matrix were frozen before neural evaluation.
- The primary layer-28/rank-2 DAS rotation was hash-verified and never retrained.
- Every abstraction scored the same neural interventions on identical untouched test rows.
- Depth analysis used only preregistered layers 16/20/24/28/30.
- Integrated H/E DAS training remains optional and was not used to choose the primary result.
- No full activation bank was retained.
- Circuit localization remains blocked unless the complete causal-abstraction gate passes.
