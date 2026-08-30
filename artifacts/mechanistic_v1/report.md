# Mechanistic Implementation of Context-Sensitive Outcome-History Integration

Full prompt activations were streamed and discarded. Only directions, scalar projections, aggregate metrics, and intervention results are retained.

## Registered questions

1. **Raw outcome history represented?** Best held-out R²: 0.248 at layer 4.
2. **Context-relevant history represented?** Best held-out R²: 0.094 at layer 8.
3. **Emergence across depth.** Raw peak: 4; contextual peak: 8.
4. **Context beyond recency.** Maximum matched partial ΔR²: 0.092.
5. **Cross-task transfer.** Mean strict-LOTO R²: -0.014; no target-task normalization or refitting was used.
6. **Specificity.** Maximum measured shared variance with registered controls: 0.891.
7. **Representation gate.** Passed; selected target/layers: [{'target': 'action_history', 'layer': 30}].
8. **Calibrated projection movement.** Requested-versus-realized computational correlation: 1.000.
9. **Steering direction.** Quantitative causal correlation: 0.538.
10. **Task-specific prediction.** Frozen behavioral coefficients—not steering outcomes—generated every predicted cell; maximum predicted/observed correlation: 0.538.
11. **Dose monotonicity.** Mean task-wise Spearman rho: -0.964.
12. **Projection mediation.** Detected.
13. **Contextual versus raw patching.** Mean contextual-minus-raw mediated effect on context-conflict examples: not estimable.
14. **Best-supported mechanistic outcome.** Representation without demonstrated causal role.
15. **Claim boundary.** Causal implementation is claimed only when representation, calibrated bidirectional steering, specificity, and projection-patching gates all pass. Decoding alone is not treated as mechanism.

## Gate summary

- Representation: pass
- Quantitative steering: fail/not run
- Projection patching: pass
- Random/control specificity: pass

## Reproducibility

See `run_metadata.json` for the model revision, behavioral handoff hash, condition hash, activation position, layer convention, split, and seed.
