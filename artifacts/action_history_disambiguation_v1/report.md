# Action-History Direction Disambiguation

**Outcome:** `H4_mixed_representation`

**Justified claim:** The selected late direction mixes action-history information with downstream decision geometry; only the surviving controlled fraction is supported.

## Primary gates

- Gate 1: **PASS**
- Gate 2: **PASS**
- Gate 3: **FAIL**
- Gate 4: **FAIL**

## Required questions

1. L8 action-history validation R² is 0.667.
2. L30 action-history validation R² is 0.389.
3. Current-persistence overlap is quantified in direction_similarity.csv; L30 raw-versus-persistence cosine is 0.064.
4. Train-only residual action history remains above its random control: Gate 1 is True.
5. Decision-matched history separation passes Gate 2: True.
6. The L30 action-history/persistence-probe cosine is 0.064.
7. The L30 action-history/direct-gradient cosine is 0.015.
8. L30 raw monotonicity |ρ|=0.217; random-direction 95th percentile |ρ|=0.868.
9. Raw L8 steering has an all-task matched-norm slope of 0.001.
10. Residualized L8 causal specificity is included in Gate 3, which is False.
11. Raw L30 clean-vector reproduction cosine is 1.000000; its steering curve is reported in Figure 4.
12. L30 output-subspace-orthogonal steering exceeds the matched random null: False.
13. L30 statistically residualized steering exceeds the matched random null: False.
14. The best task-slope/frozen-beta correlation is 0.008; Gate 4 is False.
15. The preregistered evidence rules select H4_mixed_representation.
16. The selected late direction mixes action-history information with downstream decision geometry; only the surviving controlled fraction is supported.

## Artifact and interpretation guardrails

- The exact original L30 vector and frozen behavioral coefficients were not refit.
- All residualization and direction fitting used training rows only.
- Primary causal comparisons use equal residual-stream displacement norms.
- The random null contains 100 directions at each primary layer and all seven doses.
- No full activation matrices were written.
