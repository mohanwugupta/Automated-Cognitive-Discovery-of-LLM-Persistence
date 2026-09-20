# Task-transfer report

Status: **development/incomplete**  
Endpoint: `cognitive_counterfactual_recovery`  
Metric: `global_cfr_v1`  
Completed work units: 1/21.

Behavioral theory status: `unresolved`.  
Frozen survivor set: `['dual_history', 'latent_context', 'outcome_history']`.  
All transfer cells are indexed by survivor theory; transfer results do not retroactively select a behavioral winner.

Off-diagonal cells from a source that failed within-task Gate D are recorded as `unavailable`, not zero. Transfer-predictor inference is a separate Gate-E step and must not be fitted until this matrix and its predictor definitions are frozen.
