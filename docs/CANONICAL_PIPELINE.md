# Canonical pipeline

This document records the owner-approved scientific pipeline at repository commit `3a5c8749d881632beccd2f12cc1016992ccc358f` (`origin/main`). It is a provenance map, not a reinterpretation of the saved results. Paths and hashes are machine-readable in `canonical_manifest.yaml`.

## Reading rules

- `discovery_v1` supplies upstream behavioral data; it is not the final behavioral conclusion.
- The behavioral handoff is unresolved among dual history, latent context, and outcome history. Downstream analyses therefore retain all three where designed to do so.
- Canonical causal recovery uses global counterfactual recovery, `CFR_G = 1 - sum((observed-target)^2) / sum((baseline-target)^2)`, with a zero-effect baseline. Old means of per-example CFR values are not canonical evidence.
- Cognitive-counterfactual recovery and natural-effect recovery are different endpoints. The first targets a frozen cognitive model; the second targets the unmodified language model's observed source-minus-base effect.
- `origin/computational-discrimination-audit` remains unmerged. Only the two conclusions explicitly adopted by the owner are included: target-preserving shuffles are uninformative, and E is descriptive rather than a uniquely identified abstraction.

## Dependency diagram

~~~text
external digital-minds study (historical only)

discovery_v1
  └─> discovery_v2
       └─> theory_resolution_v1 [dual_history vs latent_context vs outcome_history: UNRESOLVED]
            ├─> mechanistic_v1 matched-condition manifest [live dependency]
            │    ├─> mechanistic_v1 probes/steering [supporting/legacy interpretation]
            │    └─> action_history_disambiguation_v1 [supporting diagnostic]
            └─> causal_mech_v1 [DAS bases + Level-4 causal representation]
                 ├─> causal_specificity_v2 [canonical CFR_G and specificity]
                 │    ├─> Level-5B necessity [NOT RUN]
                 │    ├─> target-preserving shuffles [UNINFORMATIVE]
                 │    ├─> abstraction_discovery_v1 [E descriptive; identity unresolved]
                 │    │    └─> ood_free_generation_v1 [boundary test]
                 │    ├─> Qwen fixed-setting stability [supporting]
                 │    └─> Qwen fresh-context extension [supporting]
                 └─> independent Qwen confirmation [claim-bearing natural-effect extension]

Llama failed label interfaces [measurement-validity history]
  └─> Llama Yes/No interface validation
       └─> Llama behavioral replication
            └─> Llama mechanistic replication

All claim-bearing nodes ─> paper evidence replay and figures
~~~

## Behavioral stages

### `discovery_v1` — canonical upstream discovery/data

- **Scientific question:** Which manipulations in the task design space elicit structured persistence behavior in the core Qwen system?
- **Input stage/artifacts:** Rendered task conditions and model responses produced by the discovery workflow; no prior canonical stage.
- **Target/estimand:** Persistence logit, defined as log probability of continuing minus log probability of disengaging after two-token renormalization.
- **Primary metric:** Standardized condition-level persistence responses and exploratory effect estimates across task families.
- **Status:** `canonical`, with role `upstream_behavioral_discovery_and_data`.
- **Exact output artifacts:** `artifacts/discovery_v1/behavior/standardized.parquet` (SHA-256 `36f0102d32563fcf10602707cbe2febd31a94caa88567fe828591f548d5584a2`).
- **Establishes:** The upstream behavioral dataset and design-space evidence used by later model comparison.
- **Does not establish:** The final behavioral theory, a unique computational model, or a neural mechanism.

### `discovery_v2` — canonical model comparison and active discovery

- **Scientific question:** Which behavioral models best predict persistence, and which actively selected conditions discriminate among them?
- **Input stage/artifacts:** `discovery_v1` standardized observations plus the version-2 condition and hierarchy workflow.
- **Target/estimand:** Held-out persistence logit under the candidate hierarchical behavioral models.
- **Primary metric:** Held-out predictive error/model-comparison metrics recorded in `final_validation/metrics.json`; active selection uses predicted discrimination among candidate models.
- **Status:** `canonical`.
- **Exact output artifacts:** `artifacts/discovery_v2/final_validation/condition_manifest.parquet` (`7418d8683b1c70be517b71faef0fa9de508197a2a8c9597161029da35f6d1de6`), `artifacts/discovery_v2/final_validation/metrics.json` (`c92da0fcce9d93734f9de95b4838eae7697c7dd3edb979663260f2d6361aff37`), and `artifacts/discovery_v2/hierarchy/best_hierarchical_model.pkl` (`656639e358adf65bf2079d924f9e6dcadd6f2549508957743effd3aa6dd39ca3`).
- **Establishes:** The canonical behavioral model-comparison and active-discovery record.
- **Does not establish:** A uniquely resolved theory; that decision is made only in `theory_resolution_v1` and remains unresolved.

### `theory_resolution_v1` — canonical behavioral handoff

- **Scientific question:** Do discriminating behavioral conditions uniquely favor dual history, latent context, or outcome history?
- **Input stage/artifacts:** `discovery_v2`, the frozen candidate models, and the theory-resolution discrimination set.
- **Target/estimand:** Generalization error of each frozen theory on discriminating behavioral observations.
- **Primary metric:** Paired model-error contrasts with the preregistered uncertainty/equivalence decision rule recorded in the decision artifact.
- **Status:** `canonical`; decision status `unresolved`.
- **Exact output artifacts:** `artifacts/theory_resolution_v1/discrimination/frozen_model_metrics.csv` (`3746f88ae0c95f64cc36ddf43cf31e0fe47a53192addbfa4bb6f1081e3577726`), `artifacts/theory_resolution_v1/discrimination/theory_decision.json` (`98aec2161a23c9bf2ac2c6cbcb6dd5491cf2d2ffa2c53a3bf5f42ca0f5ea3b28`), and the three frozen model files listed in `canonical_manifest.yaml`.
- **Establishes:** No unique behavioral winner: the comparison among dual history, latent context, and outcome history remains unresolved.
- **Does not establish:** That any one of the three theories is true, or permission to report a downstream controller as uniquely corresponding to a resolved theory.

## Matched-condition dependencies and historical diagnostics

### `mechanistic_manifest_v1` — canonical matched-condition dependency

- **Scientific question:** Which behaviorally matched conditions define the controlled neural comparisons used by mechanistic analyses?
- **Input stage/artifacts:** The theory-resolution handoff and rendered task conditions.
- **Target/estimand:** Condition identity, matching variables, prompt identity, and analysis split; this is a design object rather than a fitted effect.
- **Primary metric:** Manifest integrity and exact condition/prompt matching.
- **Status:** `canonical` as a dependency.
- **Exact output artifacts:** `artifacts/mechanistic_v1/manifests/mechanistic_conditions.jsonl` (SHA-256 `a75c0795779c62aa8ff7f36765f84826021f458585176a4642a46a4132255342`).
- **Establishes:** The matched-condition population that later analyses are meant to use.
- **Does not establish:** Probe validity, steering specificity, DAS recovery, or causal abstraction.

### `mechanistic_probe_steering_v1` — supporting historical analysis

- **Scientific question:** Are cognitive-model variables decodable from activations, and do early one-direction steering/patching interventions move behavior as predicted?
- **Input stage/artifacts:** `mechanistic_manifest_v1`, frozen behavioral coefficients, captured activations, fitted directions, and steering/patching results.
- **Target/estimand:** Decodability and intervention-induced persistence-logit changes for early candidate directions.
- **Primary metric:** Saved representation gates, steering dose response, predicted-versus-observed effects, and patching summaries.
- **Status:** `supporting`; its probe/steering interpretation is `legacy` relative to DAS.
- **Exact output artifacts:** `artifacts/mechanistic_v1/representation/gates.json` (`be7357a0848e2a7ac44726110008feb20e94180763824bfd925998d247f583fd`) and `artifacts/mechanistic_v1/report.md` (`ba406e65643484a717bbc700c10d5e0a5714f79451043691a43be3a8ad2c285a`), with detailed tables under the same artifact directory.
- **Establishes:** Historical supporting evidence and the provenance of the matched-condition/mechanistic workflow.
- **Does not establish:** The core mechanistic claim, a unique computational identity, or Level-4/Level-5 specificity.

### `action_history_disambiguation_v1` — supporting diagnostic

- **Scientific question:** Can the proposed persistence direction be distinguished from action-history and related direction constructions?
- **Input stage/artifacts:** The mechanistic matched manifest, action residualization targets, fitted directions, and diagnostic interventions.
- **Target/estimand:** Residual action-history coding and dissociation from persistence-related effects.
- **Primary metric:** The representation, patching, dose-response, random-control, and behavioral-correspondence gates in the saved diagnostic.
- **Status:** `supporting`.
- **Exact output artifacts:** `artifacts/action_history_disambiguation_v1/gates.json` (`e00167bab30455dd42abfcec8148bdd2e81e7187c8dd2a58a2267c44fd77d725`) and `artifacts/action_history_disambiguation_v1/report.md` (`faf7fd3f9e1bd32a2fb131b2e5f8b5799403acb4caa422e67a613ff67b275514`).
- **Establishes:** A targeted diagnostic about action-history direction confounding.
- **Does not establish:** The canonical DAS result, necessity, or a unique cognitive computation.

## Canonical causal-mechanistic stages

### `causal_mech_v1` — DAS alignments and Level-4 causal representation

- **Scientific question:** Can low-dimensional distributed alignment search recover cognitive-model counterfactual effects on held-out pairs and tasks?
- **Input stage/artifacts:** The unresolved frozen behavioral models, `mechanistic_manifest_v1`, rendered counterfactual pairs, and Qwen activations.
- **Target/estimand:** Cognitive-counterfactual recovery: the intervention effect relative to each frozen behavioral model's predicted source-minus-base effect.
- **Primary metric:** Canonical interpretation is `global_cfr_v1`; the saved pair-level intervention data are the estimand source. Later stable reanalysis is in `causal_specificity_v2`.
- **Status:** `canonical` as the source of DAS bases and Level-4 results.
- **Exact output artifacts:** `artifacts/causal_mech_v1/counterfactuals/pair_manifest.parquet` (`6591f8430e52232932e56f79a768409363ddece9c9da0bcd1b6bd089b4e266e4`), `predicted_effects.parquet` (`dc87d567f8560c079f6671c24d25df81f13f6990b795daf9f7ac4f44ef681351`), and the primary dual-history layer-28 rank-2 basis (`ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`).
- **Establishes:** Low-dimensional DAS controllers can reproduce structured cognitive-model counterfactuals, including held-out-task recovery for selected controllers.
- **Does not establish:** Necessity, unique theory identity, natural-effect recovery, or that old per-example CFR summaries are valid canonical metrics.

### `causal_specificity_v2` — canonical metric and specificity reanalysis

- **Scientific question:** Do the Level-4 DAS conclusions survive a stable aggregate recovery metric and planned specificity controls?
- **Input stage/artifacts:** Frozen `causal_mech_v1` pairs, interventions, alignments, behavioral models, and controls.
- **Target/estimand:** Cognitive-counterfactual recovery only.
- **Primary metric:** `global_cfr_v1` with paired/bootstrap uncertainty; correlation is a companion criterion, not a substitute endpoint.
- **Status:** `canonical`; supersedes old per-example CFR interpretation.
- **Exact output artifacts:** `artifacts/causal_specificity_v2/corrected_metrics/bootstrap_intervals.csv` (`e506a63831faa8aa9a377d267ba3c12d24329170f0a907289e98632e594cfd9d`), `candidate_gates.csv` (`28bdf81ab7b53768bd8a0f4badc05786bea007b0f344d5d6ceba90b9a159dd62`), and `frozen_models/behavioral_hash.json` (`75e677528483c23852594e6c320e664d2f4a5e7293ed81c34d9e29e66ca61bab`).
- **Establishes:** Which saved candidates meet the corrected Level-4 recovery gate and which informative specificity controls they pass.
- **Does not establish:** Level-5B necessity, unique behavioral-theory identity, or evidence from target shuffles that leave targets unchanged.

### Level-5B necessity — explicit `not_run`

- **Scientific question:** Is the discovered neural subspace necessary for the behavior under an approved necessity intervention?
- **Input stage/artifacts:** Would require selected `causal_specificity_v2` candidates and an executed necessity design.
- **Target/estimand:** Loss of task-relevant effect after selective removal or disruption of the candidate subspace.
- **Primary metric:** No metric was evaluated.
- **Status:** `not_run`, not failed.
- **Exact output artifacts:** `artifacts/causal_specificity_v2/frozen_models/necessity_jobs.json` (`37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`), whose job list is empty.
- **Establishes:** Only that no valid Level-5B evaluation is present.
- **Does not establish:** Necessity or failure of necessity.

### Target-preserving shuffled-target controls — explicit `uninformative_control`

- **Scientific question:** Would the candidate cease to recover effects if target values were meaningfully reassigned?
- **Input stage/artifacts:** The dual-history and outcome-history specificity candidates and their shuffled-target control records.
- **Target/estimand:** Difference between recovery of original and genuinely permuted target values.
- **Primary metric:** Target-change audit before any recovery/gate statistic.
- **Status:** `uninformative_control` for the later-audited shuffles that preserve target values.
- **Exact output artifacts:** The corresponding rows in `artifacts/causal_specificity_v2/candidate_gates.csv`, cross-checked by `paper/generated/results_inventory.json`.
- **Establishes:** Nothing about specificity when the values are unchanged; it establishes a control-construction defect.
- **Does not establish:** A passed or failed shuffled-target specificity test.

## Abstraction and boundary generalization

### `abstraction_discovery_v1` — canonical descriptive abstraction analysis

- **Scientific question:** Which downstream abstraction of the behavioral computation best describes interchange-intervention effects?
- **Input stage/artifacts:** Selected DAS controllers from `causal_mech_v1`/`causal_specificity_v2` and a frozen semantic contrast manifest.
- **Target/estimand:** Recovery and geometry for candidate abstractions, including downstream evidence variable E, on held-out semantic contrasts.
- **Primary metric:** Full preregistered identity gate combining geometry, `global_cfr_v1`, correlation, random-control, and uniqueness requirements.
- **Status:** `canonical`; identity status `unresolved`.
- **Exact output artifacts:** `artifacts/abstraction_discovery_v1/design/contrast_manifest.parquet` (`abc4b863d19aa62417264e1e8ffd34636808a6b01e6be4b90f724337aec06497`), `frozen_das/interchange_results.parquet` (`8b1be70a4d8a52751987a8b18b278b018665f4484b18eef8a87315b0e98288cc`), and `gates.json` (`1b7be5e32c5b2f17e62be0e25ad9540cfc0e70601cae97d079cf6e793e5da02c`).
- **Establishes:** E is the strongest descriptive downstream candidate.
- **Does not establish:** That E, or any candidate abstraction, uniquely passes the full identity gate.

### `ood_free_generation_v1` — canonical boundary/generalization experiment

- **Scientific question:** Does the frozen controller produce dose-dependent persistence in free generation outside the structured task interface?
- **Input stage/artifacts:** The frozen Qwen controller, fixed E orientation used for dosing, prompts, seeds, and generated token-event histories.
- **Target/estimand:** Change in generation persistence/termination hazard as a function of intervention dose.
- **Primary metric:** Dose coefficient and interval from the saved survival analysis, with direction, dose-response, prompt/topic, and random-control gates.
- **Status:** `boundary`.
- **Exact output artifacts:** `artifacts/ood_free_generation_v1/generations/generation_summary.parquet` (`edcfb532e49feb8f0c9482fbefe3e29c225ca39e037ef2841080b63f9bd0157d`), `generations/token_events.parquet` (`7476e408f4502ad86fb7258529a876857d0f0bf31657bb535488901d8121dd69`), `gates.json` (`fbbeadc1e1f25fe8f001e61a3616b4fb37ef3c59277c5fef382a974273c667c0`), and `analysis/survival_model.json` (`cfb92d1cbae89611209676e1fc1dcf0a90bab9bf41d21d3d94345c73c1b24095`).
- **Establishes:** The measured boundary behavior of the frozen intervention in free generation; the canonical dose effect did not pass the primary direction/dose/random gates.
- **Does not establish:** E identity, a unique abstraction, or a successful OOD generalization claim.

## Qwen extensions and confirmation

### `qwen_fixed_setting_stability_v1` — supporting stability analysis

- **Scientific question:** Are selected-controller results stable across repeated seeds in the fixed original setting?
- **Input stage/artifacts:** The selected Qwen controller and fixed-setting pair/intervention exports.
- **Target/estimand:** Seed-to-seed stability of aggregate recovery.
- **Primary metric:** Across-seed `global_cfr_v1` summaries and bootstrap intervals.
- **Status:** `supporting`.
- **Exact output artifacts:** `paper/generated/runpod_protocol.json` (`8f2f4414fa495f33b319676bb10a0563020bb797dc0c10891745b2bc529ea6ce`) and `paper/generated/runpod_metrics.csv` (`319faaab55387021b8be7b7f0cd5714f182792027eb0ee9c73838ce11cb43a46`).
- **Establishes:** Stability of the saved result in the specified fixed setting.
- **Does not establish:** Independent context generalization, model-version portability, or a new neural identity.

### `qwen_fresh_contexts_v1` — supporting broader-context extension

- **Scientific question:** How does the frozen Qwen controller behave on freshly rendered broader contexts?
- **Input stage/artifacts:** The frozen layer-28 rank-2 controller, new context pairs, and the pinned later Qwen checkpoint.
- **Target/estimand:** Separately named cognitive-counterfactual recovery and natural-effect recovery.
- **Primary metric:** `global_cfr_v1` for each endpoint, reported separately.
- **Status:** `supporting`.
- **Exact output artifacts:** `paper/generated/qwen_fresh_contexts/protocol.json` (`aab5baed00a77d852f43b0551b71da1720ba5bf97143bf35f47563a01e865ab0`) and `paper/generated/qwen_fresh_contexts/metrics.csv` (`2d7bf4e0c512fa51cd25f141d4334ca91ab73631492c5c4f43ca90fbb5dbda2d`).
- **Establishes:** Supporting evidence about broader-context behavior, including strong natural-effect recovery in the saved post-hoc analysis.
- **Does not establish:** The preregistered independent confirmation or equivalence between natural and cognitive endpoints.

### `qwen_confirmation_v2` — claim-bearing independent extension

- **Scientific question:** Does the frozen Qwen controller generalize to independently rendered pairs when the primary target is the model's natural source-minus-base effect?
- **Input stage/artifacts:** The original layer-28 rank-2 basis (`ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0`), two prompt wordings, both mappings, seven tasks, and Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Target/estimand:** **Primary:** natural-effect recovery. **Secondary and separately reported:** cognitive-counterfactual recovery.
- **Primary metric:** `global_cfr_v1` with semantic-pair bootstrap intervals; direct-DAS and random controls are separate methods.
- **Status:** `canonical`, category `claim_bearing_extension`.
- **Exact output artifacts:** `paper/generated/qwen_confirmation_v2/protocol.json` (`8db8a14d9882a504448d6e376a5f0902e99c431b21ba98cf7fba86f1cc4a421e`), `interventions.csv.gz` (`5cccfc1b89ee9b6dd5c143e7429e343ecd838ebc95a3568055917acb96d55140`), and `metrics.csv` (`6b7063f3b56983760af0021de9e7e55d46fabee65ca42121dbfc25d9475fa8fb`).
- **Establishes:** Independent natural-effect generalization for the frozen controller under the specified protocol; familiar-task CFR is 0.779 and held-out-task CFR is 0.816 in the saved summary.
- **Does not establish:** Strong cognitive-target recovery on the same extension; those saved values are 0.251 and -0.428 and must not be relabeled as natural-effect results.

## Llama conceptual replication

### `llama_interface_v2` — supporting Yes/No measurement validation

- **Scientific question:** Does a semantically explicit Yes/No response interface validly measure continuation versus disengagement in Llama?
- **Input stage/artifacts:** Balanced semantic conditions, both response polarities, and the pinned Llama checkpoint.
- **Target/estimand:** Correct semantic direction and stability of the Yes/No persistence-logit measurement.
- **Primary metric:** Calibration and task-level interface gates.
- **Status:** `supporting`.
- **Exact output artifacts:** `paper/generated/llama_interface_v2/protocol.json` (`71175c0e4cc703ff04c4b7f931a7471034fe1fefa618b4363be89dab5c042a7f`) plus `calibration_gates.json` and `validation_gates.json` in the same directory.
- **Establishes:** The interface used by the canonical Llama replication passed its measurement-validity gates.
- **Does not establish:** Behavioral model fit or neural recovery.

Earlier X/Y, A/B, and order experiments remain under `paper/generated/llama_pilots` and `paper/generated/llama_order`. They are `legacy` measurement-validity history, not failed conceptual replications.

### `llama_behavior_v2` — canonical conceptual behavioral replication

- **Scientific question:** Does structured persistence and history-sensitive model fit replicate in Llama under a valid Yes/No interface?
- **Input stage/artifacts:** `llama_interface_v2`, 2,800 training observations, 980 selection observations, and 980 final-test observations from the pinned Llama revision.
- **Target/estimand:** Held-out Yes/No persistence logit.
- **Primary metric:** Final-test predictive R-squared after model selection; dual history is selected in the saved protocol.
- **Status:** `replication`.
- **Exact output artifacts:** `paper/generated/llama_behavior_v2/protocol.json` (`f35224b2ae6199f8af047284d438d15c69d59b0494ea811848a4b9e86650836b`), `metrics.csv` (`18e62b0aaf5af781eac947e5f39151bca1b367d0c3543c4b3c5f5d47f48f3dc4`), and `test_dual_history_predictions.csv` (`f27738b156ad98763370208a1a1de9dbf5ca46d19ac4fed30f8d6354a60bf4e7`).
- **Establishes:** A conceptual behavioral replication using the valid interface; saved final-test dual-history R-squared is 0.835 versus 0.821 for immediate state.
- **Does not establish:** That the Qwen behavioral comparison was resolved, or that the Llama neural mechanism is identical to Qwen's.

### `llama_mechanistic_v2` — canonical conceptual mechanistic replication

- **Scientific question:** Does a low-rank Llama controller recover counterfactual effects on familiar and held-out tasks?
- **Input stage/artifacts:** The frozen Llama behavioral model, selected layer-24 rank-2 intervention, familiar/held-out pairs, generic and random controls.
- **Target/estimand:** **Primary:** cognitive-counterfactual recovery. **Secondary:** natural-effect recovery. They remain separate.
- **Primary metric:** `global_cfr_v1` with semantic-pair bootstrap intervals and control comparisons.
- **Status:** `replication`.
- **Exact output artifacts:** `paper/generated/llama_mechanistic_v2/protocol.json` (`c02af8f14f105e781d3ca80edb79bd48c56f89f2e758a6b23f23f741cb528b88`), `theory_freeze.json` (`59cf6bc5641012053eea3197a75c460a493da0360830f6a70721b037fd09cf72`), `interventions.csv.gz` (`935969103a2b22fae27fe941a136826141d101d36bf53937ccfa7e7627c6715a`), and `metrics.csv` (`3673898bda89fb2c52cdc2714c58984bc29d7e1e4bf5693d40dbc85a9d1ef5d2`).
- **Establishes:** Familiar-task cognitive recovery (0.649) and measured natural-effect recovery (0.345 familiar, 0.319 held-out) under the Llama protocol.
- **Does not establish:** Held-out cognitive recovery (saved value 0.011), superiority to generic natural-effect controls, or content identity with the Qwen basis. The selected Llama basis hash is not present in the committed compact export.

## Manuscript replay layer

### `paper_evidence_replay` — supporting arithmetic and figure inventory

- **Scientific question:** Do manuscript values and figures agree with the committed compact artifacts?
- **Input stage/artifacts:** All claim-bearing stages above.
- **Target/estimand:** Exact arithmetic and provenance consistency, not a new scientific endpoint.
- **Primary metric:** Deterministic regeneration and equality of reported values/gates to the frozen artifact inventory.
- **Status:** `supporting`.
- **Exact output artifacts:** `paper/generated/results_inventory.json` (`dc12093befa8dbfd864c9be26811a2e14805fb89d5798d9ac5d58fd61301c302`) and `paper/generated/controllers.csv` (`89de8c8429a0552b943f495ea11fb3b3f99b4c0033fe4eeb49eb87ac3abbb1f2`).
- **Establishes:** An auditable bridge from compact artifacts to reported manuscript numbers.
- **Does not establish:** Correctness of upstream model execution when raw activations, condition manifests, or external archives are unavailable.

## Replay entry points available before refactoring

These commands replay committed summaries; they are not yet the promised one-command final pipeline:

~~~bash
python scripts/paper_results.py --output /tmp/persistence-paper-audit
python scripts/analyze_qwen_confirmation.py
python scripts/analyze_qwen_fresh_contexts.py
python scripts/analyze_llama_behavior.py
python scripts/analyze_llama_mechanistic.py
python scripts/audit_original_study.py --root /path/to/digital-minds-hackathon --output /tmp/original-study-audit
~~~

GPU generation and intervention commands remain stage-specific and should not be rerun merely to verify manuscript arithmetic. Stage 2B will put those commands behind explicit configurations only after regression tests freeze the current results.
