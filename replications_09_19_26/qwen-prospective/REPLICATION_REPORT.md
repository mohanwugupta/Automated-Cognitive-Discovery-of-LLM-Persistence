# Replication report

## Component summary

Measurement interface:                  PASS
Behavioral history-sensitive structure: PASS
Behavioral theory uniquely resolved:    NO / UNRESOLVED
Low-dimensional causal controller:      PASS
Held-out example generalization:        FAIL
Held-out task-family generalization:    FAIL
Specificity vs random controls:         INCOMPLETE CONTROLS
Abstraction identity:                   NOT_RUN
Far-OOD continuation generalization:    NOT RUN

## 1. Model identity

- Model: `Qwen/Qwen3.5-4B`
- Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Adapter: `qwen`

## 2. Measurement-interface result

Selected `a_b`; approved tasks: 7.

## 3. Behavioral model comparison

Behavioral replication status: `pass`.  
Behavioral best model: `dual_history`.  
Frozen behavioral survivor set: `['dual_history', 'latent_context', 'outcome_history']`.

## 4. Frozen behavioral theory status

Status: `unresolved`. The behavior-only survivor set is frozen before neural fitting; neural CFR cannot change its membership.

| Theory | Behavioral test R² / MSE | Layer/rank | Familiar CFR [95% CI] | Whole-task CFR [95% CI] | Mapping robust | Random comparison | Specificity |
|---|---|---|---|---|---|---|---|
| dual_history | 0.763864 / 0.410229 | L16/rank8 | 0.46796 [0.22166, 0.677815] | -0.138205 [-0.368258, 0.0619182] | fail | fail | fail |
| latent_context | 0.754357 / 0.426746 | L16/rank2 | 0.776436 [0.635916, 0.90933] | 0.221409 [0.10048, 0.329492] | fail | pass | fail |
| outcome_history | 0.754197 / 0.427023 | L16/rank2 | 0.246517 [-0.273507, 0.421915] | -0.301973 [-0.493197, -0.119511] | fail | fail | fail |

### Per-task whole-task-holdout recovery

| Theory | Held-out task | CFR | Correlation | Slope |
|---|---|---:|---:|---:|
| dual_history | effort | -0.1586573955 | -0.041546988 | -0.0521682238 |
| dual_history | information_sampling | -0.0039505478 | -0.3795752297 | -0.5089292045 |
| dual_history | waiting | -0.3371148568 | -0.0070519707 | -0.0144533815 |
| latent_context | effort | 0.1989586019 | 0.347638614 | 0.1785177195 |
| latent_context | information_sampling | -1.1203030724 | 0.0092709824 | 0.0203790553 |
| latent_context | waiting | 0.3400046 | 0.422166544 | 0.3926236188 |
| outcome_history | effort | -0.6014282875 | -0.2681801286 | -0.3306671383 |
| outcome_history | information_sampling | -0.123262342 | -0.2958875107 | -0.4279444351 |
| outcome_history | waiting | -0.2617838679 | -0.3430368668 | -0.3697308943 |

## 5. Mechanistic search grid

Selected controller(s): `dual_history=L16/rank8, latent_context=L16/rank2, outcome_history=L16/rank2`. The fresh-run grid is defined by relative depth, not a fixed Qwen layer.

## 6. Selected controller

Endpoint: `cognitive_counterfactual_recovery`; metric: `global_cfr_v1`.

## 7. Held-out within-task cognitive recovery

Global CFR: {'dual_history': 0.4679603348, 'latent_context': 0.7764361148, 'outcome_history': 0.2465173265}; interval: {'dual_history': [0.2216603837, 0.6778153252], 'latent_context': [0.6359162369, 0.909329971], 'outcome_history': [-0.2735069762, 0.4219145904]}.

## 8. Held-out task-family cognitive recovery

Global CFR: {'dual_history': -0.1382053298, 'latent_context': 0.221408667, 'outcome_history': -0.301973442}; interval: {'dual_history': [-0.3682575828, 0.0619181629], 'latent_context': [0.100480336, 0.3294924854], 'outcome_history': [-0.4931967977, -0.1195111951]}.

## 9. Specificity controls

Matched random-subspace comparison: INCOMPLETE CONTROLS.

| Theory | Split | Mapping CFR / gap | Random mean / max / p | Named controls | Shuffle controls |
|---|---|---|---|---|---|
| dual_history | neural_task_holdout | {"continue_x": -0.16314669835088447, "continue_y": -0.1132639612130717} / 0.0498827371 | -0.064247742 / 0.1241488226 / 0.8243512974 | {"output_readout_direction": -0.07275358027276524, "pca_variance_subspace": -0.14922900641846004, "predictive_ridge_direction": -0.0414518333145959} | source=uninformative_control; target=uninformative_control |
| dual_history | neural_test | {"continue_x": 0.14131323999407952, "continue_y": 0.79460742951576} / 0.6532941895 | -0.1434765696 / 0.1556234597 / 0.001996008 | {"output_readout_direction": -0.2786014260505829, "pca_variance_subspace": -0.08426986668584058, "predictive_ridge_direction": -0.1794754382518784} | source=uninformative_control; target=uninformative_control |
| latent_context | neural_task_holdout | {"continue_x": 0.3738180797392854, "continue_y": 0.06899925423822673} / 0.3048188255 | -0.0539162196 / 0.1053235569 / 0.001996008 | {"output_readout_direction": -0.10943907390460028, "pca_variance_subspace": 0.014887211559417302, "predictive_ridge_direction": -0.041394458414646174, "shuffled_source_base": 0.12604121222136466} | source=informative_control; target=informative_control |
| latent_context | neural_test | {"continue_x": 0.8212872447447206, "continue_y": 0.731584984916841} / 0.0897022598 | 0.0128076815 / 0.208865174 / 0.001996008 | {"output_readout_direction": -0.027785591028886447, "pca_variance_subspace": -0.0796314606377484, "predictive_ridge_direction": 0.12109407442872833, "shuffled_source_base": 0.714697543041799} | source=informative_control; target=informative_control |
| outcome_history | neural_task_holdout | {"continue_x": -0.45536573959617477, "continue_y": -0.14858114435438963} / 0.3067845952 | -0.0516860237 / 0.1542032387 / 1.0 | {"output_readout_direction": -0.013744831519137968, "pca_variance_subspace": -0.05533677134622095, "predictive_ridge_direction": 0.024913934975999807} | source=uninformative_control; target=uninformative_control |
| outcome_history | neural_test | {"continue_x": -0.14246774344693525, "continue_y": 0.635502396446577} / 0.7779701399 | -0.1519011093 / 0.1440628175 / 0.001996008 | {"output_readout_direction": -0.08622434384879973, "pca_variance_subspace": -0.1528679497990315, "predictive_ridge_direction": -0.16116451289124356} | source=uninformative_control; target=uninformative_control |

Detailed random-null and intervention rows are also recorded in `replication_results.json` and `specificity/`.

## 10. Optional abstraction result

`not_run`

## 11. Optional OOD boundary result

`not_run`

## 12. Descriptive historical comparison

Historical values are context only and were not used for tuning or selection.

| Quantity | Historical Qwen | Prospective Qwen |
|---|---|---|
| Behavioral theory status | unresolved | unresolved |
| Surviving theories | ['dual_history', 'latent_context', 'outcome_history'] | ['dual_history', 'latent_context', 'outcome_history'] |
| Selected layer/rank | L28/rank2 primary | dual_history=L16/rank8, latent_context=L16/rank2, outcome_history=L16/rank2 |
| Familiar-task CFR | 0.968265 | {'dual_history': 0.4679603348, 'latent_context': 0.7764361148, 'outcome_history': 0.2465173265} |
| Whole-task CFR | 0.872806 | {'dual_history': -0.1382053298, 'latent_context': 0.221408667, 'outcome_history': -0.301973442} |
| Specificity | partial; recovery and random controls pass, variable identity does not | INCOMPLETE CONTROLS |

## 13. Critical interpretation

### dual_history: Outcome B — prospective Qwen loses whole-task transfer

This suggests the older task-general Qwen result depended on the historical pipeline/design rather than reflecting a robust universal controller.

### latent_context: Outcome A — prospective Qwen retains whole-task transfer

This supports model-dependent differences: prospective Qwen transfers under the modern harness where Gemma/Llama did not.

### outcome_history: Outcome B — prospective Qwen loses whole-task transfer

This suggests the older task-general Qwen result depended on the historical pipeline/design rather than reflecting a robust universal controller.

Do not repair or tune this outcome to agree with historical Qwen.

## 14. Limitations

Component outcomes remain separate. A failure or non-run optional boundary test does not retroactively change the structured-task result.

## 15. Exact provenance hashes

```json
{
  "adapter": {
    "name": "qwen",
    "version": "qwen-hf-residual-v2"
  },
  "analysis_id": "qwen-prospective",
  "baseline_preflight_sha256": "1a87b614295d146fccbaa973352f81808cd942f9e4d2e5fce6b5529fe527630c",
  "behavioral_condition_manifest_hash": "74243cdcbe4166907ffeb8b03bf55b68882c191a1d1b82fdbf6235cb030f9dc5",
  "behavioral_design_hash": "e8209ec6749724ec79808c518cb6167c8734c5d75ef340246381524149144361",
  "behavioral_split_hash": "d9670b1df541e58267487eb15ebda60176600463d7996afdc7ac86dc1379bbed",
  "behavioral_survivor_manifest_sha256": "9a969a94cc1a83da4452a39aec6293d36196b411522b5dc3913dd6fcc0dcc24a",
  "behavioral_survivor_rule_sha256": "ea9c201c5570d70e2f423f578e6c2c7ddf4d925271627d953f8ae00a39eafcc8",
  "behavioral_survivor_set": [
    "dual_history",
    "latent_context",
    "outcome_history"
  ],
  "behavioral_survivor_set_frozen_before_neural": true,
  "behavioral_survivor_set_sha256": "d50375916ff18570386a11232d268c81e589f71fe5d31d98571b322547180562",
  "behavioral_theory_status": "unresolved",
  "config_hash": "2bd48661fc0b88aab0cc1d0d39db09eedc3774bd5ace7f954db987de98be0cd3",
  "counterfactual_pair_manifest_hash": "23ae80776eda6aacef105254babe873383156d3ab837e5184f6ecf1cb9a2308b",
  "counterfactual_prediction_hash": "ac9f4a62143290947ebe63ad4aa5a8c5366c221e2178f6449ab4d867753f42d6",
  "endpoint_id": "cognitive_counterfactual_recovery",
  "environment": {
    "lockfile": "uv.lock",
    "lockfile_sha256": "1da819846ebda7993e472dc2b7c6b44a09a76249dc2026ecc7d77ac1d4903661"
  },
  "excluded_endpoints": [
    "natural_effect_recovery",
    "fresh_context_extension"
  ],
  "frozen_model_hashes": {
    "dual_history/model.pkl": "7fb36b06c48ec9bf48b0bf8557fe32ff976a5bde3179c975401f8203da7fb3f0",
    "dual_history/model_spec.json": "c1c9ebd3a7f197a9e9b73e96dc6a6a5310f9e385917e69b3be491bae7acb3f9f",
    "dual_history/parameters.csv": "836977ceb68a3cd2df9fd2633af678626078aef2ecf278e39cb5d9fc8a4c0dd4",
    "dual_history/training_condition_hashes.json": "5342937a562d515ac09372c4871a5753203506ef341cba9c79b4d051ed2f7fc8",
    "latent_context/model.pkl": "15792364dac9335f924e91c5dce7b0dcbd243a23c578ca4bba53a7d48c6d5803",
    "latent_context/model_spec.json": "05b3cc3fb5ff2b9445ebccdb6c4329bdededa1b25b1ff0369abb8cd3d7e4f7ef",
    "latent_context/parameters.csv": "fa99d54b882a69f8bb252ee41c881368290b25633520ecf9b2ade614da39c057",
    "latent_context/training_condition_hashes.json": "5342937a562d515ac09372c4871a5753203506ef341cba9c79b4d051ed2f7fc8",
    "outcome_history/model.pkl": "c9fb0970bb5d813e73594acddb2e10ff5b69449f60a43399109663e885bb6adc",
    "outcome_history/model_spec.json": "955d41d07519ad1d1bb22d8b9056d3e1c417f091540d72cf0dfc3b3c83a5de10",
    "outcome_history/parameters.csv": "0beeadfb3a3409333a73753368ee01baf023d3e119ae3aa427663ea1226ad8c8",
    "outcome_history/training_condition_hashes.json": "5342937a562d515ac09372c4871a5753203506ef341cba9c79b4d051ed2f7fc8"
  },
  "gates_mutable_after_baseline": false,
  "git_commit": "b5d76994dc13753420995da79693a006466351cd",
  "git_dirty": false,
  "historical_reuse": {
    "reuse_historical_behavioral_models": false,
    "reuse_historical_counterfactual_pairs": false,
    "reuse_historical_das_bases": false,
    "reuse_historical_layer_rank": false,
    "reuse_historical_neural_splits": false,
    "reuse_historical_theory_winner": false
  },
  "layer_rank_grid": {
    "ranks": [
      2,
      8
    ],
    "relative_depths": [
      0.25,
      0.5,
      0.75,
      0.9
    ]
  },
  "metric_id": "global_cfr_v1",
  "model": {
    "id": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
  },
  "neural_split_hash": "918440f243bbf4c9f3d2103ff01eb87a832f0728b0e88746c3641d8f31e7fe50",
  "output_path": "/scratch/gpfs/JORDANAT/mg9965/replications/qwen-prospective",
  "protocol_file_sha256": "a49af9d6bc634b67c107bfd6de86265c78ec21e41029845d8abd46d893d684ad",
  "resolved_layers": [
    8,
    16,
    24,
    28
  ],
  "run_kind": "pipeline_self_replication",
  "run_spec_sha256": "2768e5f0f2810f7b74a75b662dae624aff28be6e30a209279e51ad7ea22f2310",
  "schema_version": "replication-provenance-v1",
  "seeds": {
    "behavior_design": 12001,
    "behavior_fit": 12003,
    "behavior_split": 12002,
    "bootstrap": 15001,
    "counterfactual_design": 13001,
    "interface_calibration": 11001,
    "interface_validation": 11002,
    "mechanism_search": 14001,
    "neural_split": 13002,
    "random_controls": 14002
  },
  "selected_controller_artifact_hashes": {
    "dual_history": "06e7d69f48da254a52db57b4f03b72ebc2d6e9ddd7b12eea28db8907ba93d7a0",
    "latent_context": "6b194779581f6714b3ceb151bc1f04c05843947d663e1ae81fe9591c5a34fa6c",
    "outcome_history": "100f286b8b60389c1adc1d317db0a917d73e6cf192a37bc482d564d12aca8b3c"
  },
  "selected_controller_hash": "a0cafdac860f2007162b6dc97e51c00128c37cee70ecf4e1dcc34323ffcc3403",
  "tokenizer": {
    "id": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
  }
}
```
