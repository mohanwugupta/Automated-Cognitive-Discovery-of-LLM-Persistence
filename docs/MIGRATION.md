# Selective migration from `digital-minds-hackathon`

The old repository remains provenance and was not copied wholesale. This clean
implementation ports validated ideas at explicit boundaries:

| New component | Provenance in old repository | What was retained |
| --- | --- | --- |
| `participants/qwen.py` | `models/hooked_qwen.py` | Qwen 3.5 loader selection, exact chat-continuation token checks, two-token renormalization, raw action mass |
| `design/counterbalance.py` | `cross_task/common.py` | Reversible semantic X/Y mappings and pair identity |
| seven renderers | `bandit`, `cross_task`, `experiments/persistence_battery` | Natural semantics for reward pursuit, patch leaving, repeated solvability, information sampling, waiting, progressive effort, and debugging |
| collection | `experiments/persistence_battery/collection.py` | Behavior-only logits, semantic action replay, paired latent seeds, no hidden states |
| data validation | `analysis/comparative_persistence` | Pair/episode-safe splits, explicit missingness, future-leakage rejection, task-macro evaluation |
| cognitive/flexible models | `computational_modeling` and `analysis/comparative_persistence` | Modular model registry, shared/task-specific/hierarchical fits, MLP/GRU validation, LOTO and recovery tests |
| SLURM flow | `run_qwen35_bandit.sh` and submit helpers | Offline caches, Qwen cluster path, array collection, dependency-gated pilot → discovery → analysis |

Activation banks, probes, L21/L22 searches, steering, patching, and historical
artifacts are intentionally absent. The new Qwen participant raises if hidden
states appear in a behavior-only call.

