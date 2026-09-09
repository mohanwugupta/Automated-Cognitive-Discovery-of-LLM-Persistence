# Independent Qwen confirmation: complete

The frozen Qwen L28/rank-2 controller was evaluated on 224 rendered pairs spanning 56 semantic history contrasts per wording, two wording variants, both response mappings, and seven tasks. New semantic states exclude overlap with the previous fresh-context study. The 23,296 intervention evaluations include the frozen basis, four comparison methods, and 99 random bases. No model, basis, layer, rank, or intervention was selected on these data.

Natural-effect recovery was the declared primary endpoint. The confirmation passes positive-recovery and random-superiority criteria in both task groups, but **does not pass superiority over every comparator**: direct behavior-target DAS performs slightly better in aggregate.

| Reference endpoint | Familiar recovery [95% interval] | Neural task-holdout recovery [95% interval] |
|---|---:|---:|
| Natural history effects | 0.779 [0.736, 0.837] | 0.816 [0.790, 0.835] |
| Frozen cognitive targets | 0.251 [0.101, 0.386] | -0.428 [-0.885, -0.059] |
| Previously fitted global-gain targets | 0.488 [0.365, 0.597] | 0.025 [-0.285, 0.291] |
| Previously fitted task-gain targets | 0.522 [0.401, 0.635] | -0.045 [-0.382, 0.241] |

## Controls and interpretation

Both natural endpoints exceed all 99 random bases: plus-one p=.01 each, Holm p=.02 across the two task groups. The frozen controller also exceeds ridge persistence, PCA, and output controls. Direct behavior-target DAS reaches 0.802/0.830; selected-minus-direct differences are -0.023 [-0.036, -0.002] on familiar tasks and -0.015 [-0.040, 0.018] on holdouts. This supports useful natural causal control, not unique superiority of cognitive guidance.

The two wording variants give frozen natural recovery 0.834/0.720 on familiar tasks and 0.858/0.794 on holdouts. They preserve factual statements, reorder context paragraphs, and rephrase the action instruction. This limited variation is not a test of broad linguistic generality.

Prediction gains were fitted only on the previous fresh-context dataset. They improve some cognitive-target scores but do not restore reliable held-out target recovery. Calibration includes previous data from all task families: these calibrated targets are tested on new contexts, not wholly unseen behavioral tasks.

## Precision and ablation diagnostics

Norm matching is approximate. Maximum method-level median relative discrepancies are 1.09% on familiar tasks and 1.75% on holdouts, with individual maxima of 21.0%. Full discrepancies are exported.
A post hoc common-pair 5% matching sensitivity retains 103/128 rows for mech_pair_test: frozen natural recovery 0.811, highest random 0.087, plus-one p=0.01.
A post hoc common-pair 5% matching sensitivity retains 59/96 rows for mech_task_holdout: frozen natural recovery 0.828, highest random 0.055, plus-one p=0.01.
This norm-based subset does not replace the prespecified full-data result.

Prespecified training-mean replacement attenuates the projection of natural history effects by 0.699/0.759. It uses only the original training mean, but lacks matched ablation controls and does not establish selective necessity.

## Reproducibility and resources

The run completed in 3,800 seconds after model loading, with 9.22 GB peak allocated VRAM on an L40S. Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Torch `2.8.0.dev20250319+cu128`; Transformers `5.16.1`.

Full archive SHA-256: `22d0cf41610a687c79482f3207323fa27f1db1b324efefabe64e58821fdc6aba`. The downloaded archive was hash-verified and the completion count checked before the final pod was deleted. No experiment pods or network volumes remain.

The archive retains executed scripts, protocol, logs, raw results, bases, calibration inputs, original training reference data, and frozen cognitive packages. A wording-only protocol correction clarified semantic versus rendered pair counts after launch; it did not change the experiment. Archived hashes identify the executed version. CPU replay from compact exports reproduces the reported analyses.
