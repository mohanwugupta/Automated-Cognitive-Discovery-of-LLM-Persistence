# Llama access and behavioral feasibility results

Access is enabled and real Llama experiments have completed. Full mechanistic replication has **not** been run: neither tested response format passed the inherited behavioral acceptance gates across all tasks.

## Completed runs

- Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Original pilot: 490 semantic situations (70 per task), each with both X/Y mappings, for 980 forward evaluations. Seed 85001.
- Post-pilot correction: one prospectively chosen A/B format compared with X/Y on the same 490 fresh situations. Seed 86001; 1,960 evaluations. These situations have zero semantic overlap with the original pilot, even when environment seed is excluded from the comparison.
- Total: 2,940 behavioral evaluations. Prescribed history endpoints only, not a full replication of the expanded-history discovery dataset. No Llama cognitive model or DAS subspace was fitted.
- Both protocols were saved before their model evaluations. The A/B correction was conceived after the original failure; it is not an original preregistered condition. Both A/B and X/Y outcomes are retained, with no further format search or threshold relaxation.

## Results

Every evaluation had an action label as its top token. Every task passed the probability-range and logit-variation gates. The failure was counterbalancing: the mean absolute difference in semantic continuation probability when labels were swapped exceeded 0.25.

Values below are percentage-point gaps. Lower is better; the inherited acceptance threshold is at most 25 points.

| Task | Original X/Y | Fresh X/Y | Fresh A/B |
|---|---:|---:|---:|
| bandit | 37.9 | 37.2 | 21.4 |
| debugging | 52.2 | 55.2 | 41.6 |
| effort | 51.1 | 52.2 | 56.3 |
| foraging | 43.5 | 45.9 | 33.3 |
| information sampling | 49.7 | 49.2 | 49.5 |
| solvability | 29.1 | 29.0 | 49.1 |
| waiting | 32.8 | 33.5 | 37.8 |

X/Y passed the counterbalance gate on zero of seven tasks in both sets. A/B passed on bandit only. Changing labels is therefore not sufficient to establish a usable seven-task Llama battery.

This is a measurement-validity result. It does not show that Llama lacks a persistence mechanism or that cognitive-model-guided discovery fails. Label swapping changes both answer labels and their presentation within the existing renderers; the current experiments do not isolate token identity from answer-option order or other prompt effects.

## Technical validation and limits

The saved credential passed a HEAD request to an actual gated weight shard before download. The real 8B model loaded on an L40S. At layer 16, identity patching had exactly zero measured logit error and the alignment gradient was finite and nonzero (norm 0.354). The randomly initialized rank-2 edit had zero measured forward logit effect at bfloat16 precision; this check alone does not establish a useful causal intervention. The tiny random-Llama integration test separately passes nontrivial interchange and gradient checks.

The container ran Torch `2.8.0.dev20250319+cu128` and Transformers `5.16.1`; the image tag must not be mistaken for the observed Torch build. Peak allocated VRAM during the original pilot was 16.63 GB. Protocol files retain the executed script hashes. All 2,940 compact exported observations independently reproduce the task-level gate statistics.

## Resources and next experiment

Pod `6z9vhkulwj53jc` was deleted after a SHA-256-verified result download; an empty pod listing confirmed cleanup. No network volume was created. Estimated compute/storage cost is below $0.20, based on the roughly eight-minute allocation at $1.09/hour plus prorated temporary storage; this is not a posted billing amount.

Next, freeze a prompt-design study that independently varies label identity and answer-option order. Develop changes only on calibration situations, then test the chosen format on untouched semantic situations with the same acceptance thresholds. Only after that gate passes should Llama-specific behavioral models be fitted and frozen, followed by layer/rank selection and held-out causal tests. Averaging mappings or dropping failed tasks would change the measurement or scope and must be declared as a separate protocol, not silently substituted for this failed pilot.

Compact results and metadata: `paper/generated/llama_pilots/`. Full manifests, prompts, observations, logs, and executed scripts are in the accompanying reproducibility archive. The existing manuscript's Qwen causal findings remain a single-model mechanistic result.

## Parallel follow-up: label pair and option order

A third protocol, specified after the first two failures, crossed X/Y versus A/B with continue-first versus disengage-first option presentation. It used 140 new semantic situations (20 per task), both mappings, and four formats: 1,120 additional evaluations. Only a format passing every inherited task gate could be selected; the selection rule chose none. The separately generated 490-situation validation set was not evaluated. Total completed Llama behavioral evaluations are now **4,060**.

| Calibration format | Tasks passing all gates | Mean continuation probability |
|---|---:|---:|
| A/B, continue first | 1/7 | 0.337 |
| A/B, disengage first | 3/7 | 0.812 |
| X/Y, continue first | 0/7 | 0.412 |
| X/Y, disengage first | 5/7 | 0.852 |

Option order therefore materially affects behavior in this calibration set. The best passing-task count still failed debugging and information sampling; it was not promoted to validation or mechanistic fitting. These small calibration samples do not establish a stable corrected measurement. A revised response interface and independent validation remain necessary; no threshold was relaxed and no failed task was dropped.

All executed prompt hashes were checked against saved messages and observations. Compact replayable exports are in `paper/generated/llama_order/`; full prompts, manifests, and logs are in `llama-order-reproducibility.tar.gz`. Pod `o3mh87kcmzrv48` was deleted after a checksum-verified download. Its allocation cost is estimated below $0.45; this is not a posted invoice.
