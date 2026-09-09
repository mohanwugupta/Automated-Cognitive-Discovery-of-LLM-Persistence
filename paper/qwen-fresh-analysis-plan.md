# Fresh-context analysis plan and scope

Primary endpoint: global counterfactual recovery against each controller's frozen cognitive model, separately for familiar-task and held-out-task contexts. Report all three controllers, all 20 random subspaces per controller, task-level results, calibration slopes/intercepts, and matched-norm discrepancies. No controller, layer, rank, cognitive coefficient, or dose is selected using these data.

Uncertainty: 1,000 bootstrap resamples of background contexts within task, preserving all four history patterns and both response mappings inside each cluster. There are only four backgrounds per task. Monte Carlo random-comparison values use (1 + null scores at least as large as the frozen score)/21; these are coarse descriptive comparisons, not corrected confirmatory tests across controllers and splits.

Secondary diagnostic added during the intervention run: score natural source conditions and reuse the cached natural base scores, with the same pinned checkpoint/runtime. Compare the frozen cognitive targets with these actual source-minus-base effects. After observing cognitive-target miscalibration, also specify intervention recovery against natural effects for EVERY method, not only a favorable controller. This is a diagnostic endpoint, not a replacement for the original cognitive-target endpoint. Report both even if they disagree.

The natural-effect diagnostic was added after the experiment began; it is not an original preregistered analysis. It does not demonstrate cognitive-variable identity or necessity. No new GPU controller fitting or test-based tuning is performed.
