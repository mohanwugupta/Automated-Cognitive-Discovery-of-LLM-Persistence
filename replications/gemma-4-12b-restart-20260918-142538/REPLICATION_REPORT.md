# Replication report

## Component summary

Measurement interface:                  PASS
Behavioral history-sensitive structure: PASS
Behavioral theory uniquely resolved:    YES
Low-dimensional causal controller:      PASS
Held-out example generalization:        PASS
Held-out task-family generalization:    FAIL
Specificity vs random controls:         FAIL
Abstraction identity:                   NOT_RUN
Far-OOD continuation generalization:    NOT RUN

## 1. Model identity

- Model: `/scratch/gpfs/JORDANAT/mg9965/models/google--gemma-4-12b-it`
- Revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- Adapter: `gemma`

## 2. Measurement-interface result

Selected `a_b`; approved tasks: 7.

## 3. Behavioral model comparison

Behavioral replication status: `pass`.  
Frozen history-sensitive model set: `['latent_context']`.

## 4. Frozen behavioral theory status

Status: `resolved`. Selection is frozen before neural fitting.

## 5. Mechanistic search grid

Selected controller(s): `latent_context=L24/rank2`. The fresh-run grid is defined by relative depth, not a fixed Qwen layer.

## 6. Selected controller

Endpoint: `cognitive_counterfactual_recovery`; metric: `global_cfr_v1`.

## 7. Held-out within-task cognitive recovery

Global CFR: {'latent_context': 0.3900876596}; interval: {'latent_context': [0.3013576501, 0.5891853343]}.

## 8. Held-out task-family cognitive recovery

Global CFR: {'latent_context': -0.0149813478}; interval: {'latent_context': [-0.0741259858, 0.0431467513]}.

## 9. Specificity controls

Matched random-subspace comparison: FAIL.

## 10. Optional abstraction result

`not_run`

## 11. Optional OOD boundary result

`not_run`

## 12. Limitations

Component outcomes remain separate. A failure or non-run optional boundary test does not retroactively change the structured-task result.

## 13. Exact provenance hashes

```json
{
  "adapter": {
    "name": "gemma",
    "version": "gemma-hf-residual-v2"
  },
  "behavioral_condition_manifest_hash": "74243cdcbe4166907ffeb8b03bf55b68882c191a1d1b82fdbf6235cb030f9dc5",
  "behavioral_design_hash": "e8209ec6749724ec79808c518cb6167c8734c5d75ef340246381524149144361",
  "behavioral_split_hash": "d9670b1df541e58267487eb15ebda60176600463d7996afdc7ac86dc1379bbed",
  "config_hash": "043bb5688f8e7b792cee243d32150611a48f341b999a0aac08b5fa96e84dc13f",
  "counterfactual_pair_manifest_hash": "54a9f2581c9318a3474a66f0ab88e8ef900b25e356b8bff873807ebede53b0ba",
  "counterfactual_prediction_hash": "4b4a86296d01bd8e200e305823cca8fb3580f07ae74803ee473b22feed508d90",
  "endpoint_id": "cognitive_counterfactual_recovery",
  "environment": {
    "lockfile": "uv.lock",
    "lockfile_sha256": "1da819846ebda7993e472dc2b7c6b44a09a76249dc2026ecc7d77ac1d4903661"
  },
  "frozen_model_hashes": {
    "latent_context/model.pkl": "b9d4e464240a671725719714e0c5139a9aa0ffa0432387c9bf041d0f75999eb1",
    "latent_context/model_spec.json": "05b3cc3fb5ff2b9445ebccdb6c4329bdededa1b25b1ff0369abb8cd3d7e4f7ef",
    "latent_context/parameters.csv": "3eec303a7f92df9ea83ff9bf30f5d96c8d97aabd4f4c651e4bdeaec3a4340686",
    "latent_context/training_condition_hashes.json": "5342937a562d515ac09372c4871a5753203506ef341cba9c79b4d051ed2f7fc8"
  },
  "git_commit": "00d427489b1ad8e81c9c60e2488d0689aa74f917",
  "git_dirty": true,
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
    "id": "/scratch/gpfs/JORDANAT/mg9965/models/google--gemma-4-12b-it",
    "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
  },
  "neural_split_hash": "cd873c45de60748562354390c0df62c3a2cb06f0d5aadde7ca11d31184c07134",
  "output_path": "/scratch/gpfs/JORDANAT/mg9965/replications/gemma-4-12b-restart-20260918-142538",
  "resolved_layers": [
    12,
    24,
    36,
    43
  ],
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
    "latent_context": "5bb4fb6172495f507a320736e538b3c1aa0b821ed6c83e69cf108d0cba3e49fd"
  },
  "selected_controller_hash": "e0efd37feec6306e4f64f2afae7fc7bbbc93931145b3fbf853212b2048518b2d",
  "tokenizer": {
    "id": "/scratch/gpfs/JORDANAT/mg9965/models/google--gemma-4-12b-it",
    "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
  }
}
```
