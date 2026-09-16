# Claim-to-evidence map

This file maps manuscript-level claims to frozen evidence on the approved `origin/main` base (`3a5c8749d881632beccd2f12cc1016992ccc358f`). A claim is only as strong as the status and limitation recorded here. Full artifact hashes and stage relationships are in `canonical_manifest.yaml`.

## Endpoint firewall

The repository contains two distinct recovery endpoints:

| Endpoint | Target effect | Canonical name | May be substituted for the other? |
|---|---|---|---|
| Cognitive-model target | Frozen behavioral model's predicted source-minus-base effect | `cognitive_counterfactual_recovery` | No |
| Natural model effect | Unmodified language model's empirical source-minus-base effect | `natural_effect_recovery` | No |

Both use `global_cfr_v1` when expressed as recovery: `1 - sum((observed-target)^2) / sum((0-target)^2)`. Every claim below names its endpoint. Mean per-example CFR is superseded and is not a canonical claim metric.

## C01 — Structured persistence is behaviorally predictable, but the core Qwen theory is unresolved

- **Claim:** Structured persistence is captured by history-sensitive behavioral models in the core Qwen study, while the discriminating comparison does not uniquely resolve dual history, latent context, and outcome history.
- **Analysis ID:** `discovery_v1` → `discovery_v2` → `theory_resolution_v1`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`; the historical core revision is not pinned (`null`, `historical_unknown`).
- **Behavioral object:** Persistence logit and the three frozen theory models. File hashes: dual history `656639e358adf65bf2079d924f9e6dcadd6f2549508957743effd3aa6dd39ca3`; latent context `d92be7907416acb2f74ad66b64a74e8342d87d4ec3062b36f42b90a8fdffed6f`; outcome history `fe945cff238ba65ea327f4f4d1d16625017a4f06d02f5781232b6f41313e1667`.
- **Neural object:** Not applicable.
- **Data/pair/split identity:** `discovery_v1/behavior/standardized.parquet` hash `36f0102d32563fcf10602707cbe2febd31a94caa88567fe828591f548d5584a2`; `discovery_v2/final_validation/condition_manifest.parquet` hash `7418d8683b1c70be517b71faef0fa9de508197a2a8c9597161029da35f6d1de6`; theory training-condition identity is recorded by the frozen-model metadata.
- **Metric/version:** Held-out behavioral prediction metrics plus the paired uncertainty/equivalence rule in `theory_decision.json`; not a neural CFR claim.
- **Supporting artifact:** `artifacts/theory_resolution_v1/discrimination/theory_decision.json` (`98aec2161a23c9bf2ac2c6cbcb6dd5491cf2d2ffa2c53a3bf5f42ca0f5ea3b28`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** The winner is `null`. Downstream analyses must not rewrite the behavioral handoff as a unique dual-history, latent-context, or outcome-history victory.

## C02 — Low-dimensional Qwen controllers recover cognitive-model counterfactuals

- **Claim:** Low-dimensional DAS controllers recover structured cognitive-model counterfactual effects on held-out pairs and, for selected controllers, held-out tasks.
- **Analysis ID:** `causal_mech_v1`, interpreted through `causal_specificity_v2`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`; exact historical revision unknown.
- **Behavioral object:** Frozen dual-history, latent-context, and outcome-history targets; hash registry in `artifacts/causal_specificity_v2/frozen_models/behavioral_hash.json` (`75e677528483c23852594e6c320e664d2f4a5e7293ed81c34d9e29e66ca61bab`).
- **Neural object/hash/layer/rank:** Primary manuscript controller is dual-history outcome-history at layer 28, rank 2, basis SHA-256 `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`. The saved latent-context layer-30 rank-8 and outcome-history layer-30 rank-2 bases are separately identified in their files and the paper inventory.
- **Data/pair/split identity:** Pair manifest SHA-256 `6591f8430e52232932e56f79a768409363ddece9c9da0bcd1b6bd089b4e266e4`; splits are train, validation, test, and task holdout. Neural fitting tasks are bandit, debugging, foraging, and solvability; held-out tasks are effort, information sampling, and waiting.
- **Metric/version:** `cognitive_counterfactual_recovery` using `global_cfr_v1`, with paired/bootstrap uncertainty and a companion correlation gate.
- **Supporting artifact:** `artifacts/causal_specificity_v2/corrected_metrics/bootstrap_intervals.csv` (`e506a63831faa8aa9a377d267ba3c12d24329170f0a907289e98632e594cfd9d`) and `paper/generated/controllers.csv` (`89de8c8429a0552b943f495ea11fb3b3f99b4c0033fe4eeb49eb87ac3abbb1f2`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** This is Level-4 causal representation evidence, not necessity, unique theory identity, or natural-effect recovery. Old per-example CFR interpretations are superseded.

## C03 — Corrected Qwen specificity is partial; Level-5B is absent

- **Claim:** The corrected stable-metric reanalysis identifies Level-4 candidates that pass its approved recovery criteria and some informative controls; it does not provide Level-5B necessity evidence.
- **Analysis ID:** `causal_specificity_v2`, `level5b_necessity_v2`, `shuffled_target_dual_history_v2`, and `shuffled_target_outcome_history_v2`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`; exact historical revision unknown.
- **Behavioral object:** The same frozen three-model registry as C02.
- **Neural object/hash/layer/rank:** Selected saved DAS candidates; primary layer-28 rank-2 basis hash `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`.
- **Data/pair/split identity:** Frozen `causal_mech_v1` pair manifest `6591f8430e52232932e56f79a768409363ddece9c9da0bcd1b6bd089b4e266e4`; corrected bootstrap and candidate-gate tables are exact downstream views of those pairs.
- **Metric/version:** `global_cfr_v1`, bootstrap lower-bound and correlation criteria, plus individually named specificity controls.
- **Supporting artifact:** `artifacts/causal_specificity_v2/candidate_gates.csv` (`28bdf81ab7b53768bd8a0f4badc05786bea007b0f344d5d6ceba90b9a159dd62`) and empty `frozen_models/necessity_jobs.json` (`37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** Level-5B is `not_run`, not failed. Target shuffles found by the later audit to preserve dual-history or outcome-history target values are `uninformative_control`, not passes or failures. No unmerged branch conclusion beyond these owner decisions is implied.

## C04 — E is descriptive, not an identified abstraction

- **Claim:** Downstream evidence E is the strongest descriptive abstraction candidate, but no candidate uniquely passes the full identity gate.
- **Analysis ID:** `abstraction_discovery_v1`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`; historical revision unknown.
- **Behavioral object:** Candidate downstream abstractions of the unresolved frozen computational models.
- **Neural object/hash/layer/rank:** The three selected DAS controllers from `causal_mech_v1`; primary layer-28 rank-2 basis hash `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`.
- **Data/pair/split identity:** Contrast manifest SHA-256 `abc4b863d19aa62417264e1e8ffd34636808a6b01e6be4b90f724337aec06497`; 2,800 rendered rows covering train, validation, and test; interchange results hash `8b1be70a4d8a52751987a8b18b278b018665f4484b18eef8a87315b0e98288cc`.
- **Metric/version:** Full abstraction identity gate combining held-out geometry, `global_cfr_v1`, correlation, random-control, and uniqueness checks.
- **Supporting artifact:** `artifacts/abstraction_discovery_v1/gates.json` (`1b7be5e32c5b2f17e62be0e25ad9540cfc0e70601cae97d079cf6e793e5da02c`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** E's descriptive geometry/recovery does not license an identity claim; the random/uniqueness portion of the full gate is not passed.

## C05 — Free-generation OOD is a boundary result

- **Claim:** The frozen controller's free-generation experiment defines a negative/limited boundary result rather than successful evidence for abstraction E.
- **Analysis ID:** `ood_free_generation_v1`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`; exact historical revision unknown.
- **Behavioral object:** Free-generation persistence/termination, not the structured two-token cognitive target.
- **Neural object/hash/layer/rank:** Frozen layer-28 rank-2 DAS basis `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`; saved E-orientation object hash `98732c97b59d21bb7fb15e47aa959eea55f4e90130f6864b6e7b7b3a5049e5f6`.
- **Data/pair/split identity:** 1,500 primary generations from six prompts, 50 seeds, and five doses; `generation_summary.parquet` hash `edcfb532e49feb8f0c9482fbefe3e29c225ca39e037ef2841080b63f9bd0157d` and `token_events.parquet` hash `7476e408f4502ad86fb7258529a876857d0f0bf31657bb535488901d8121dd69`.
- **Metric/version:** Saved Cox/survival dose coefficient with direction, dose, random-direction, prompt, and topic gates. The saved dose coefficient is -0.0138 with interval [-0.0501, 0.0225] and random p-value 0.257.
- **Supporting artifact:** `artifacts/ood_free_generation_v1/gates.json` (`fbbeadc1e1f25fe8f001e61a3616b4fb37ef3c59277c5fef382a974273c667c0`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** The primary direction/dose/random gates do not pass. This experiment is not evidence that E identity was established.

## C06 — Fixed-setting Qwen results are seed-stable

- **Claim:** The selected controller's recovery is stable across the saved repeated-seed fixed-setting analysis.
- **Analysis ID:** `qwen_fixed_setting_stability_v1`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Behavioral object:** Frozen cognitive target used in the selected controller analysis.
- **Neural object/hash/layer/rank:** Five seed-specific layer-28 rank-2 DAS refits conditional on the original selection. Their fitted basis bytes are not committed; `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0` identifies the original selected controller that fixed the setting, not the five refitted bases.
- **Data/pair/split identity:** Pair and seed identity specified in `paper/generated/runpod_protocol.json` (`8f2f4414fa495f33b319676bb10a0563020bb797dc0c10891745b2bc529ea6ce`).
- **Metric/version:** Seed-level and pooled `global_cfr_v1` with bootstrap summaries.
- **Supporting artifact:** `paper/generated/runpod_metrics.csv` (`319faaab55387021b8be7b7f0cd5714f182792027eb0ee9c73838ce11cb43a46`).
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** This is supporting stability in the fixed setting, not independent context/model generalization.

## C07 — Independent Qwen confirmation supports natural-effect generalization

- **Claim:** In the independent confirmation, the frozen controller recovers the model's natural source-minus-base effect on familiar and held-out tasks.
- **Analysis ID:** `qwen_confirmation_v2`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Behavioral object:** Primary target is empirical natural effect. The original frozen cognitive target remains a separate secondary endpoint.
- **Neural object/hash/layer/rank:** Frozen Qwen layer-28 rank-2 controller, basis SHA-256 `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`.
- **Data/pair/split identity:** Protocol SHA-256 `8db8a14d9882a504448d6e376a5f0902e99c431b21ba98cf7fba86f1cc4a421e`; two wordings, both mappings, seven tasks, and disjoint semantic pairs. Compact interventions hash `5cccfc1b89ee9b6dd5c143e7429e343ecd838ebc95a3568055917acb96d55140`.
- **Metric/version:** **Natural-effect recovery**, `global_cfr_v1`, semantic-pair bootstrap. Saved frozen-controller CFR: 0.779 familiar [0.736, 0.837] and 0.816 held out [0.790, 0.835]. Direct DAS is a separately named comparison.
- **Supporting artifact:** `paper/generated/qwen_confirmation_v2/metrics.csv` (`6b7063f3b56983760af0021de9e7e55d46fabee65ca42121dbfc25d9475fa8fb`).
- **Replay command:** `python scripts/analyze_qwen_confirmation.py`
- **Limitation:** Compact exports replay the arithmetic, but the raw condition manifest expected by one existing clean-clone test is absent. This claim does not imply strong cognitive-target recovery.

## C08 — The same Qwen confirmation does not recover the cognitive target well

- **Claim:** Cognitive-target recovery in the independent Qwen confirmation is materially different from natural-effect recovery and is weak or negative on the held-out split.
- **Analysis ID:** `qwen_confirmation_v2`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Behavioral object:** Frozen dual-history cognitive target from the original study.
- **Neural object/hash/layer/rank:** Same frozen layer-28 rank-2 basis `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0` as C07.
- **Data/pair/split identity:** Same frozen protocol/intervention identities as C07.
- **Metric/version:** **Cognitive-counterfactual recovery**, `global_cfr_v1`. Saved values: 0.251 familiar [0.101, 0.386] and -0.428 held out [-0.885, -0.059].
- **Supporting artifact:** `paper/generated/qwen_confirmation_v2/metrics.csv` (`6b7063f3b56983760af0021de9e7e55d46fabee65ca42121dbfc25d9475fa8fb`).
- **Replay command:** `python scripts/analyze_qwen_confirmation.py`
- **Limitation:** These values must not be pooled with, renamed as, or used to negate the distinct natural-effect endpoint in C07.

## C09 — Broader fresh-context Qwen evidence is supporting, not the independent confirmation

- **Claim:** A broader fresh-context analysis shows strong saved natural-effect recovery while cognitive-target recovery is weaker.
- **Analysis ID:** `qwen_fresh_contexts_v1`.
- **Model/checkpoint:** `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Behavioral object:** Separate cognitive and empirical natural targets.
- **Neural object/hash/layer/rank:** Frozen layer-28 rank-2 controller `ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`.
- **Data/pair/split identity:** Fresh-context protocol hash `aab5baed00a77d852f43b0551b71da1720ba5bf97143bf35f47563a01e865ab0`; compact metrics hash `2d7bf4e0c512fa51cd25f141d4334ca91ab73631492c5c4f43ca90fbb5dbda2d`.
- **Metric/version:** `global_cfr_v1`, reported separately for cognitive and natural endpoints.
- **Supporting artifact:** `paper/generated/qwen_fresh_contexts/metrics.csv` and `natural_effect_recovery.csv`.
- **Replay command:** `PYTHONPATH=src python scripts/analyze_qwen_fresh_contexts.py`
- **Limitation:** Natural-effect analysis is post-hoc/supporting and is not a substitute for C07's independent confirmation protocol.

## C10 — Llama Yes/No behavior is a conceptual replication

- **Claim:** Under a measurement-valid Yes/No interface, Llama shows structured persistence for which dual history improves held-out prediction over immediate state.
- **Analysis ID:** `llama_interface_v2` and `llama_behavior_v2`.
- **Model/checkpoint:** `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- **Behavioral object:** Yes/No persistence logit; selected dual-history model with fixed decay alpha 1 in this protocol.
- **Neural object:** Not applicable to the behavioral claim.
- **Data/pair/split identity:** 2,800 training, 980 selection, and 980 test observations; behavior protocol hash `f35224b2ae6199f8af047284d438d15c69d59b0494ea811848a4b9e86650836b`; final-test dual-history predictions hash `f27738b156ad98763370208a1a1de9dbf5ca46d19ac4fed30f8d6354a60bf4e7`.
- **Metric/version:** Held-out R-squared on the untouched test split. Saved dual-history R-squared is 0.835 versus 0.821 for immediate state.
- **Supporting artifact:** `paper/generated/llama_behavior_v2/metrics.csv` (`18e62b0aaf5af781eac947e5f39151bca1b367d0c3543c4b3c5f5d47f48f3dc4`).
- **Replay command:** `python scripts/analyze_llama_behavior.py`
- **Limitation:** This is a conceptual replication with a new validated interface, not evidence that the unresolved Qwen model comparison has become resolved.

## C11 — Llama provides qualified conceptual mechanistic replication

- **Claim:** A selected low-rank Llama controller recovers cognitive counterfactuals on familiar tasks and measurable natural effects, but cognitive recovery does not generalize to held-out tasks.
- **Analysis ID:** `llama_mechanistic_v2`.
- **Model/checkpoint:** `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- **Behavioral object:** Frozen Llama dual-history target, behavioral hash `ea8b0cfa0533bfde5b641dad5e77fec63ccd0b24ef6960d3d4a01c0e172cfebe` as recorded in the theory-freeze metadata.
- **Neural object/hash/layer/rank:** Selected layer 24, rank 2 controller. Its content hash is `unknown_external_archive`; the selected basis is not committed. The compact external archive hash is `c8c4ec0ef8a17235186605747a2f0f2e1f7619731977760e0cfd1904784e4cd2` where recorded.
- **Data/pair/split identity:** 64 familiar and 48 held-out semantic pairs; protocol hash `c02af8f14f105e781d3ca80edb79bd48c56f89f2e758a6b23f23f741cb528b88`; compact interventions hash `935969103a2b22fae27fe941a136826141d101d36bf53937ccfa7e7627c6715a`.
- **Metric/version:** Primary **cognitive-counterfactual recovery** and secondary **natural-effect recovery**, each using `global_cfr_v1`. Saved cognitive CFR: 0.649 familiar and 0.011 held out. Saved natural CFR: 0.345 familiar and 0.319 held out.
- **Supporting artifact:** `paper/generated/llama_mechanistic_v2/metrics.csv` (`3673898bda89fb2c52cdc2714c58984bc29d7e1e4bf5693d40dbc85a9d1ef5d2`).
- **Replay command:** `python scripts/analyze_llama_mechanistic.py`
- **Limitation:** Generic controls can outperform the selected controller on natural familiar effects; the held-out cognitive endpoint does not pass. Missing basis identity prevents a full clean-clone neural replay.

## C12 — Failed early Llama labels are measurement-validity history

- **Claim:** Earlier Llama X/Y, A/B, and ordering attempts diagnose invalid response interfaces; they are not failed replications of the scientific hypothesis.
- **Analysis ID:** `llama_label_interface_history`.
- **Model/checkpoint:** Llama replication family; see each historical protocol for the exact recorded run metadata.
- **Behavioral object:** Label-token/interface validity, not persistence theory.
- **Neural object:** Not applicable.
- **Data/pair/split identity:** Historical compact exports under `paper/generated/llama_pilots/` and `paper/generated/llama_order/`.
- **Metric/version:** Calibration direction, order sensitivity, and task-level validity gates.
- **Supporting artifact:** `paper/generated/llama_pilots/llama_label_validation_v1_gates.json` and `paper/generated/llama_order/calibration_gates.json`.
- **Replay command:** `python scripts/paper_results.py --output /tmp/persistence-paper-audit`
- **Limitation:** These runs cannot be counted as negative behavioral or mechanistic replications because the measurement interface itself failed.

## C13 — The original hackathon study is external historical provenance

- **Claim:** The original digital-minds-hackathon work motivated the current study and supplies historical context only.
- **Analysis ID:** `external_original_study`.
- **Model/checkpoint:** As recorded in the external repository; not promoted to the canonical current model registry.
- **Behavioral object:** Historical persistence and steering analyses.
- **Neural object/hash/layer/rank:** External; not a canonical current basis.
- **Data/pair/split identity:** External repository commit `5b968d0bb8de64f67556a7b300200aa488a591fa`; large-file payloads are not part of this repository.
- **Metric/version:** Arithmetic checks captured by the local audit, not a current canonical endpoint.
- **Supporting artifact:** `paper/generated/original_audit/audit.json` (`bf3086a60c1bf15e0bb33f5b3dd0bdf8363ab04c8a3f3853995024dac3b71a52`).
- **Replay command:** `python scripts/audit_original_study.py --root /path/to/digital-minds-hackathon --output /tmp/original-study-audit`
- **Limitation:** Full replay requires the external repository and its LFS artifacts; this evidence must not be presented as a clean-clone product of the current repository.
