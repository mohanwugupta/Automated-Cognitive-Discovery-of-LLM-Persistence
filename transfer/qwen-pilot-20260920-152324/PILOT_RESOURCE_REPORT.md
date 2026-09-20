# Task-transfer pilot resource report

Pilot source controller: `dual_history__single__bandit`  
Measured pilot GPU-hours: 0.0325  
Measured output storage: 0.0001 GB  
Training scale (grid × epochs): 24.00×  
Evaluation scale (random controls): 50.00×  
Projected evaluation GPU-hours per target: 0.1349  
Projected Qwen GPU-hours: 26.6933  
Projected Gemma GPU-hours: 80.0800  
Projected Llama GPU-hours: 53.3867  
Cross-model projection basis: `parameter_count_ratio_planning_only`  
Projected full GPU-hours: 26.6933  
Projected full storage: 0.0839 GB

Leakage audit passed: `True`  
All seven targets evaluated: `True`  
Both mappings emitted: `True`  
Scoped artifacts/no overwrite: `True`  

This smoke pilot is not scientific evidence; its source-gate override exists only to measure all-target integration cost.

The full matrix remains blocked until both projected budgets are explicitly approved.
