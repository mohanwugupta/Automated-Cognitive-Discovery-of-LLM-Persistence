# Llama causal replication: completed, partial transfer

After independent response-validity and behavioral entry checks passed, the Llama-specific dual-history model was frozen. A prespecified search over layers 8, 16, 24, 28 and ranks 2, 8 selected **layer 24, rank 2** using 64 selection pairs, after fitting on 64 separate pairs. Neither the final pair-test nor the neural task holdouts entered that selection. Behavioral fitting included all seven tasks; task holdout refers to neural fitting only.

The completed final test contains **11,536 intervention evaluations**: 112 pairs evaluated with the selected basis, three generic controls, and 99 random bases. An additional 512 selection evaluations compare all eight search candidates; three-epoch training forwards are separate from these evaluation counts.

| Endpoint | Familiar task test: recovery [95% interval] | Neural task holdout: recovery [95% interval] |
|---|---:|---:|
| Frozen cognitive targets | 0.649 [0.511, 0.773] | 0.011 [-0.550, 0.506] |
| Natural history-change effects | 0.345 [0.224, 0.434] | 0.319 [0.000, 0.505] |

All four comparisons exceed the 99 sampled random bases: plus-one p=.01, Holm p=.02 within each endpoint, or .04 across all four comparisons. These comparisons use approximate finite-precision norm matching. Exceeding random controls does not imply accurate target recovery: the held-out cognitive score remains near zero with broad uncertainty, and the natural held-out interval touches zero.

On familiar tasks, the selected controller recovers its cognitive targets better than the ridge-persistence, direct output, and PCA controls. **Those generic controls recover natural effects better than the selected controller.** The selected-minus-control natural recovery differences are -0.222, -0.245, and -0.223; all familiar-task bootstrap intervals are negative. The corresponding held-out differences are -0.106, -0.113, and -0.156; the PCA interval is negative, while the other two include zero. Therefore this run does not establish cognitive-specific superiority for natural behavior.

Norm matching is approximate in bfloat16. The largest method-level median relative discrepancy is 2.05% on familiar tasks and 5.11% on holdouts. Maximum individual discrepancies are 37.2% and 221.5%, respectively; the latter concerns small reference norms. The full discrepancies are retained and limit interpretation of the matched-null comparison. A post hoc common-pair sensitivity check retains only pairs for which every control is within 5% of the achieved reference norm. It retains 30/64 familiar and 7/48 held-out pairs; the selected controller still exceeds all 99 random controls on both endpoints. This heavily reduced subset does not replace the full evaluation or resolve transfer uncertainty. No failure-driven retuning or additional search was performed.

This is a partial replication: cognitive-guided search recovers held-out cognitive effects within the discovery task family, but strong neural task transfer does not reproduce. The observed natural effects are not specific to this controller relative to generic alternatives. These results strengthen the distinction between matching a cognitive target and identifying a natural computation.

## Reproducibility

Model revision `0e9e39f249a16976918f6564b8830bc894c89659`; Torch `2.8.0.dev20250319+cu128`; Transformers `5.16.1`; L40S; neural run 615 seconds after model loading, peak allocated VRAM 17.17 GB. The full archive SHA-256 is `c8c4ec0ef8a17235186605747a2f0f2e1f7619731977760e0cfd1904784e4cd2`. Results were downloaded and verified before deleting the pod.

The condition generator also writes generic per-phase Qwen design metadata/predictions. Those are unused by the Llama objective. The top-level Llama protocol, frozen Llama model hash, and top-level pair predictions specify the actual experiment. The executed script explicitly recomputes every target using the frozen Llama model.
