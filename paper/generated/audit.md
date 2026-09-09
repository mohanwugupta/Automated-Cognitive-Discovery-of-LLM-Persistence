# Artifact audit

All values are re-extracted from the committed artifacts; no new GPU results.

## Controller results

```csv
candidate_index,theory,layer,rank,train_rows,validation_rows,test_estimate,test_ci_lower,test_ci_upper,holdout_estimate,holdout_ci_lower,holdout_ci_upper
0,latent_context,30,8,18,18,0.9301290076517662,0.9128865929224852,0.9529528189710011,0.6725030824784228,0.4399445453234962,0.8411768709167714
1,dual_history,28,2,24,22,0.9682648384627148,0.9425206582399418,0.9862676319448076,0.8728057199204062,0.7720522187696506,0.9478847223464114
2,outcome_history,30,2,24,22,0.9667386365148124,0.9343287376839996,0.9871730424503974,0.8783020358507125,0.7634133837036995,0.9528131448011676
```

## Shuffled-target integrity

- `artifacts/causal_specificity_v2/shards/candidate_000/shuffled_target.parquet`: 7/18 targets changed; maximum change 0.0507.
- `artifacts/causal_specificity_v2/shards/candidate_001/shuffled_target.parquet`: 0/24 targets changed; maximum change 6.66e-16.
- `artifacts/causal_specificity_v2/shards/candidate_002/shuffled_target.parquet`: 0/24 targets changed; maximum change 1.11e-15.

An ID derangement alone does not guarantee a useful null: target values must change.
An uninformative shuffle cannot establish or refute variable specificity.

Level-5B necessity jobs submitted in this stage: 0. Do not describe missing values as a measured null.

The prior report's 0.808 task-holdout mean averages the three retained controllers. The dual-history controller itself scores 0.873.
The 0.862 behavioral validation R-squared is pooled; task-macro R-squared is 0.763.
All three retained candidates must be reported; the highest test score is a descriptive maximum, not a fresh model-selection rule.
