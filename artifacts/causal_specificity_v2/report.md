# DAS Causal Specificity with Stable CFR

**Highest justified evidence level:** 4

**Conclusion:** DAS identified an effective persistence controller but not a specific implementation of the proposed cognitive variable.

## Automated questions

1. Corrected global CFR reproduces Level 4: **yes**.
2. Best untouched-test candidate: `das_001_outcome_history_dual_history_L28_r2` (layer 28, rank 2, CFR_G=0.968).
3. Best candidate beats 500 matched random subspaces: **yes** (p=0.0020; null 95th=0.077).
4. It beats shuffled sources with paired-bootstrap support: **yes**.
5. It beats shuffled targets with paired-bootstrap support: **no**.
6. It outperforms direct persistence controls: **yes** (maximum persistence-control CFR_G=0.485).
7. A raw-history subspace specifically recovers do(O): **no**.
8. A contextual-history subspace specifically recovers do(O*): **no**.
9. Functional interchangeability: **DAS_controller_without_variable_identity**; geometry is reported only as secondary evidence.
10. Context-conflict interventions favor: **raw history or unresolved**.
11. Held-out-task matched recovery: mean CFR_G=0.808; no target-task refitting occurred.
12. Target-selective necessity: **no** (mean target NS=nan).
13. Highest evidence level: **4**.
14. Justified claim: **DAS identified an effective persistence controller but not a specific implementation of the proposed cognitive variable.**
15. Unsupported claims: circuit implementation and any variable identity whose candidate did not pass all Level-5A criteria.

## Guardrails

- The behavioral models, DAS layers/ranks/rotations, pair definitions, and splits were frozen before this reanalysis.
- The primary analysis performed no DAS retraining.
- Global CFR is primary; thresholded per-example CFR is descriptive only.
- Bootstrap sampling uses semantic source/base pairs and keeps duplicated response mappings together.
- Every specificity control uses the candidate's identical untouched-test rows.
- Necessity is reported as Level 5B separately from causal specificity.
- Circuit analysis remains blocked unless Level 5A passes.
- No full activation bank was retained.
