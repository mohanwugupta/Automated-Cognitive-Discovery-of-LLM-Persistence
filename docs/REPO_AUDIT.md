# Repository audit: scientific and computational provenance

**Stage:** 1 — audit only  
**Audit date:** 2026-09-16  
**Repository:** Automated-Cognitive-Discovery-of-LLM-Persistence  
**Checked-out commit:** af1583038b719129b0a306ece4c95cb8fcedbfed  
**Scope rule:** no scientific code, configuration, artifact, or analysis result was changed. This document is the only Stage-1 deliverable.

## 1. Scope and important repository-state warning

This repository has three scientifically relevant visible refs across the checked-out core and later follow-up lineage:

| Ref | Commit | Contents | Audit treatment |
| --- | --- | --- | --- |
| Checked-out local main | af15830 | The staged Qwen pipeline through frozen OOD free generation | Audited as the executable core currently present in the worktree |
| origin/main | 3a5c874 | The same core plus a manuscript, original-study audit, later Qwen confirmations, and Llama replication work | Audited from the local remote-tracking ref as a separate follow-up/replication lane |
| origin/computational-discrimination-audit | f28ec42 | A post hoc audit of whether the abstraction contrasts discriminate computations, plus an unexecuted follow-up protocol | Audited as an unmerged supporting branch, not treated as a completed canonical analysis |

The checked-out main is two commits behind origin/main. I did not pull, merge, switch branches, or choose which ref should become the canonical base. That decision is required before refactoring.

The status labels below are deliberately provisional:

- **Canonical candidate:** lies on the apparent final Qwen dependency chain, but still needs owner approval.
- **Supporting:** validates, diagnoses, or supplies an input to a likely canonical analysis without itself being the main claim.
- **Replication/extension:** asks a new generalization, confirmation, or model-transfer question.
- **Legacy candidate:** historically useful but apparently superseded for inference.
- **Ambiguous:** code or artifacts are still consumed downstream, while their own scientific interpretation may have been superseded.

No ambiguous analysis is designated canonical here.

## 2. High-level reconstruction

The checked-out Qwen code and artifacts implement this progression:

1. Define a seven-task behavioral ontology and collect counterbalanced continue/disengage logits.
2. Compare cognitive feature models and flexible predictors under fixed behavioral splits.
3. Add hierarchical task structure, active sampling, and an untouched validation collection.
4. Freeze the surviving behavioral theories and collect a targeted contextual-history discrimination round.
5. Build a matched mechanistic condition set and run an early representation/probe/steering analysis.
6. Reuse that condition manifest for counterfactual-first causal localization and Distributed Alignment Search (DAS).
7. Re-evaluate the frozen DAS candidates with stable global counterfactual recovery and stronger controls.
8. Ask which abstraction, O, O-star, H, or E, best describes the frozen controller.
9. Freeze the selected layer/rank/basis/orientation and test transfer to unrestricted EOS stopping.

The visible later manuscript branch adds three distinct lanes that should not be collapsed into the core chain:

- an arithmetic replay of the earlier digital-minds-hackathon sequential study;
- Qwen stability, fresh-context, wording, natural-effect, and independent-confirmation studies;
- Llama response-interface validation, behavioral replication, and causal replication.

The later evidence changes the interpretation of the core results. In particular, it distinguishes recovery of a frozen cognitive target from recovery of the model's natural source-minus-base effect, and it documents that the cognitive-guided controller is useful without uniquely identifying the cognitive computation.

## 3. Primary estimands and definitions

### 3.1 Behavioral response

The primary structured-task outcome is the semantic persistence logit:

~~~text
persistence_logit = log p(continue) - log p(disengage)
~~~

For Qwen, both semantic labels must be clean one-token chat continuations. Probabilities are renormalized over the two labels, while raw probability mass assigned to either action and whether the top vocabulary token is an action are retained as validity diagnostics. The model is called with thinking disabled where supported and with use_cache false for behavior-only collection.

Response mappings are paired so that semantic continuation is represented by both label assignments. A sampled semantic action is shared across the two mappings for a pair. This sampling is secondary; the fitted behavioral target is the continuous logit.

### 3.2 Behavioral predictors

The cognitive registry contains explicit feature models, not complete canonical cognitive algorithms. It includes immediate state, dynamic re-evaluation, choice perseveration, outcome history, dual history, latent motivation, latent context, option termination, task-set reinstatement, meta-control, and generic sequential choice.

Every cognitive fit also includes response mapping as a nuisance variable. RidgeCV is used under fully shared, task-specific, or hierarchical parameterizations. The principal behavioral comparison ranks models by task-macro R-squared and pooled R-squared on fixed held-out splits. Sampled-action log loss and Brier score are secondary.

History kernels use a fixed decay of 0.7. This value appears in both behavioral feature generation and mechanistic target code. It is stipulated in source code, not fitted or declared in the experiment YAML files.

### 3.3 Behavioral splits

The initial split constructor unions paired-condition and episode identities, then assigns connected groups within each task. Roughly 20 percent per task are structural test, 20 percent of the remaining groups are interpolation test, and the remainder are discovery. Both response mappings and complete episodes remain together.

Later stages introduce different split namespaces:

- Round 2 active collection and final untouched validation;
- Round 3 round3_train and model_discrimination;
- mechanistic mech_pair_train, mech_pair_validation, mech_pair_test, and mech_task_holdout;
- separate train, selection, and test designs in the later Llama replication.

These names are not interchangeable. In the later Qwen and Llama work, neural task holdout means exclusion from neural fitting; the behavioral model may still have been calibrated using all seven task families.

### 3.4 Mechanistic intervention and metrics

The central DAS intervention replaces the base state's coordinates inside a learned low-dimensional basis with source coordinates at one residual-stream layer, then runs the rest of the model and measures the semantic persistence-logit change.

Two target endpoints coexist:

- **Cognitive target:** the counterfactual logit change predicted by a frozen behavioral theory.
- **Natural target:** the unmodified LLM's source-minus-base logit change for the same pair.

They answer different questions and are not substitutes.

Two counterfactual-recovery implementations also coexist:

- The original causal_mech_v1 selection and gating use the mean of per-example normalized recovery values. This is unstable when an individual predicted effect is near zero.
- causal_specificity_v2 and later analyses use global CFR: one minus aggregate squared error divided by aggregate squared target magnitude. A no-op scores zero and exact aggregate recovery scores one. This is not accuracy, percent mediation, or interchange-intervention accuracy.

The repository still emits both metric families, so any claim must name the exact one.

### 3.5 Abstraction variables

The abstraction code operationalizes:

- O: raw recency-weighted outcome history;
- O-star: context-relevant outcome history;
- H: the reference dual-history prediction minus a version with action/outcome regressors neutralized;
- E: the full reference dual-history model prediction;
- D: numerically defined as E in the implementation.

The later computational-discrimination audit finds that H and E are identical on 54.6 percent of evaluated rendered pairs and that O and H predicted effects correlate at 0.975. Therefore a feature-level decorrelation result does not imply that the evaluated neural interventions cleanly discriminate these computations.

## 4. Repository inventory

| Location | Role | Main inputs | Main outputs | Provisional status |
| --- | --- | --- | --- | --- |
| PRD*.md | Successive scientific specifications for behavioral discovery, hierarchy, theory resolution, early mechanisms, action-history disambiguation, causal DAS, stable CFR, abstraction, and OOD generation | Prior results and scientific hypotheses | Requirements, gates, figures, and artifact plans | Supporting provenance; not executable and not proof that every specified analysis ran |
| README.md | Combined installation, local, cluster, and stage-specific operating notes | Current CLI and Slurm layout | Human instructions | Supporting; long and partly chronological rather than a canonical pipeline guide |
| docs/MIGRATION.md | Maps selected code concepts from digital-minds-hackathon into this repository | Older repository modules | Component-level migration table | Supporting provenance; migration source commit is not recorded here |
| configs/*.yaml | Parameters for each main Qwen stage | Model ID, seeds, thresholds, paths, cluster settings | Runtime configuration | Canonical candidates, but model revision is null in every core config |
| configs/models and configs/sampling | Standalone cognitive and coverage declarations | Model names/features and sampling weights | Declarative YAML | Ambiguous: no executable references were found from the current CLI/source |
| src/cognitive_discovery | Reusable implementation | YAML, manifests, model weights, saved upstream artifacts | Designs, data, fits, interventions, metrics, reports | Mixed; mapped by package below |
| scripts/*.py | Thin CLI wrappers for the core plus standalone later paper experiments | CLI arguments and relative artifact roots | Stage outputs | Core wrappers are supporting orchestration; later standalone scripts are replication/extension candidates |
| scripts/submit_*.sh | Submit dependency-gated Della workflows | Repository root, env vars, Slurm | Job chains | Supporting operations |
| scripts/dispatch_*.sh | Read prepared job manifests/gates and submit exact arrays | Prepared JSON job lists | GPU arrays plus CPU aggregation jobs | Supporting operations |
| run_*.slurm | CPU/GPU phase dispatch with resource requests and CUDA preflight | PHASE and cluster env vars | Logs and stage artifacts | Supporting operations; Princeton-specific defaults |
| artifacts | Tracked lightweight scientific products | All stage outputs | Tables, manifests, compact weights, reports, figures | Mixed scientific record; detailed below |
| config/artifact_allowlist.txt | Exceptions to the 10 MB tracked-file limit | Repository-relative patterns | Allowlist | Supporting safeguard |
| scripts/check_artifact_sizes.py | Rejects large tracked files and activation-bank formats | Git index and allowlist | Pass/fail | Supporting safeguard |
| tests | Unit, synthetic, integration, resource, and protocol checks | Source and some committed artifacts | pytest pass/fail | Supporting; not yet a complete saved-result regression suite |
| .github/workflows/artifact-size.yml | CI workflow | Git checkout | Artifact-size check only | Supporting; CI does not run pytest |
| logs | Forty-three tracked historical Slurm logs | Past cluster jobs | stdout/stderr evidence | Legacy/supporting operational history; not part of scientific inputs |
| paper on origin/main | Manuscript, reasoning maps, protocols, replayable compact exports, and later follow-up scripts | Core artifacts plus external raw archives | Paper tables, figures, results, replication summaries | Mixed claim layer and replication/extension; absent from checked-out tree |

## 5. Source-package map

| Package | Scientific/computational question | Inputs | Outputs | Provisional status |
| --- | --- | --- | --- | --- |
| ontology | What factors, histories, response mappings, and condition fields define a persistence task? | Declarative factors and condition values | ConditionSpec, HistorySpec, ResponseMapping | Canonical foundation candidate |
| design | How are balanced, constrained, counterbalanced conditions sampled and hashed? | Ontology and design YAML | JSONL/Parquet manifests, semantic hashes, SweetPea spec | Canonical foundation candidate |
| experiments | How are the seven task families rendered without changing semantic factors? | ConditionSpec | Chat messages and prompt hashes | Canonical foundation candidate |
| participants | How are model logits converted into semantic continue/disengage responses? | Model/tokenizer and rendered messages | Two-token logits/probabilities and validity diagnostics | Canonical foundation candidate |
| data | How are records validated, split, stored, and minimally provenance-tagged? | Conditions and observations | Standardized Parquet, split IDs, run metadata | Canonical foundation candidate |
| models/cognitive and models/fitting | Which explicit feature theory predicts persistence logits? | Standardized behavioral rows | Ridge fits, predictions, R-squared/MSE/correlation | Canonical behavioral candidate |
| models/flexible | What flexible predictive ceiling is achievable and can it recover synthetic teachers? | Behavioral rows and synthetic teachers | MLP/GRU/static fits and recovery audits | Supporting validity analysis |
| hierarchy | Are computational forms shared with task-specific or ontology-conditioned parameters? | Behavioral rows and task descriptors | M1–M4 fits, variance decomposition, few-shot/LOTO tables | Canonical behavioral candidate |
| sampling | Which new conditions maximize coverage, parameter information, or theory disagreement? | Prior observations, candidate pools, frozen fits | Sampling scores and selected manifests | Canonical acquisition candidate |
| audits and theory_resolution | Are model ceilings/calibration valid, and can targeted contextual histories resolve surviving theories? | Discovery v1/v2 data and fits | Frozen model packages, targeted design, comparison and handoff | Canonical behavioral-resolution candidate |
| mechanistic | Are raw/action/contextual history variables represented and causally steerable using mean directions/probes? | Theory handoff and a newly matched condition set | Direction tensors, scalar projections, probe/steering/patching tables | Ambiguous: scientific analysis looks superseded, but its condition manifest is a live downstream dependency |
| action_history | Is the late action-history direction upstream history, downstream decision state, or mixed? | mechanistic_v1 directions, manifest, coefficients | Alternative directions, matched tests, steering, report | Replication/extension or supporting diagnostic; not on the final core chain |
| causal_mechanistic | Can frozen behavioral counterfactuals guide causal localization and DAS? | theory_resolution_v1 plus mechanistic_v1 condition manifest | Pair manifests, localization, DAS bases, validation and gates | Canonical mechanistic candidate |
| causal_specificity | Does stable global CFR preserve Level 4 and pass Level-5 specificity/necessity controls? | Frozen causal_mech_v1 candidates and results | Recomputed CFR, controls, bootstrap intervals, gates | Canonical corrective-analysis candidate |
| causal_abstraction | Does the controller correspond to O, O-star, H, or E? | theory models and frozen specificity candidates | Contrast design, geometry/interchange metrics, random/shuffle controls | Canonical abstraction candidate, with a negative identity gate |
| ood_generation | Does the frozen structured-task E-oriented controller affect EOS stopping in free generation? | Frozen abstraction/DAS artifacts and prompts | Token events, generation summaries, survival/dose/control results | Canonical frozen-generalization candidate, with a negative overall result |
| reporting | How are stage-specific figures and reports rendered? | Tables under each artifact root | report.md and stage-local Figure 1–7 files | Supporting; no global manuscript figure mapping in checked-out tree |

## 6. Executable core stages and artifact record

### 6.1 discovery_v1

**Question.** Which behavioral variables and cognitive models predict structured persistence across seven task families?

**Entry points.** cognitive-discovery-design, cognitive-discovery-run, cognitive-discovery-fit, cognitive-discovery-residuals, cognitive-discovery-validate; scripts/submit_discovery.sh orchestrates pilot, collection, fitting, and validation.

**Inputs.** configs/discovery_v1.yaml, task ontology/renderers, Qwen/Qwen3.5-4B, locally supplied model folder, and seeds 270126–270128.

**Outputs.** artifacts/discovery_v1 contains condition manifests, fixed split IDs, 18,240 standardized observations from 2,800 semantic conditions, response-surface/model/residual analyses, a frozen cognitive model, independent validation, seven figures, report, and run metadata. Raw collection shards are not retained in the tracked artifact root.

**Saved result.** dual_history is the best held-out cognitive theory; independent-validation task-macro R-squared is 0.772.

**Provenance.** Git commit, dependency versions, config hash, design hash, seeds, model identifier, and renderer version are recorded. The actual Qwen revision is null; the observed model is only an absolute cluster path.

**Status.** Canonical foundation candidate. Its headline result is superseded/narrowed by later hierarchical and theory-resolution stages, but its data are direct upstream inputs.

### 6.2 pilot_v1 and smoke_v1

**Question.** Does the design/collection pipeline pass validity gates, and can the pipeline run without a real model?

**Inputs/outputs.** pilot_v1 contains a 5.7 MB Qwen gate collection; smoke_v1 contains a 1.5 MB deterministic model-free run and synthetic report.

**Status.** Supporting only. They must never be cited as scientific evidence about Qwen persistence.

### 6.3 discovery_v2

**Question.** Does a shared architecture with task-structured parameters outperform invariant coefficients, and does active acquisition improve an untouched validation result?

**Entry point.** cognitive-discovery-active and scripts/submit_active_discovery.sh.

**Inputs.** discovery_v1 is passed as a required CLI argument rather than encoded as a path in discovery_v2.yaml. Active and final collections have separate roots under artifacts/discovery_v2.

**Outputs.** hierarchical M1–M4 comparisons, task parameters, variance decomposition, LOTO/few-shot results, candidate and selected active conditions, residual and validity audits, an independent final manifest/collection, final metrics, report, and five figures.

**Saved result.** dual_history/M4 has interpolation task-macro R-squared 0.738; dual_history and latent_context remain observationally equivalent; 30/36 flexible teacher checks pass; frozen final pooled R-squared is 0.862, while the later paper audit identifies task-macro R-squared as 0.763.

**Status.** Canonical behavioral candidate. Pooled and macro metrics must not be interchanged.

### 6.4 theory_resolution_v1

**Question.** Can contextual-history and targeted disagreement trials resolve dual_history versus latent_context before neural analysis?

**Entry point.** cognitive-discovery-theory and scripts/submit_theory_resolution.sh.

**Inputs.** discovery_v1, discovery_v2 active collection, and discovery_v2 final collection are supplied through CLI arguments. The YAML alone is insufficient to reconstruct these paths.

**Outputs.** frozen dual_history, latent_context, and outcome_history packages; a 20,000-condition candidate pool; 1,200 selected conditions; contextual-history manifests; Round-3 collection; frozen discrimination; post-update fits; theory decision; final parameters; and mechanistic handoff targets.

**Saved result.** the behavioral theory remains unresolved because the paired interval crosses zero outside the equivalence margin. The frozen mechanistic variables are history integration and contextual-history relevance.

**Status.** Canonical final behavioral-stage candidate. It explicitly retains multiple theories; downstream use of one theory must not be described as behavioral resolution.

### 6.5 mechanistic_v1

**Question.** Are outcome/action/context-sensitive history variables decodable across depth and causally manipulable with calibrated directions?

**Entry point.** cognitive-discovery-mechanistic and scripts/submit_mechanistic.sh.

**Inputs.** theory_resolution_v1 handoff, Qwen weights, and a newly generated matched mechanistic condition set.

**Outputs.** a condition manifest and split, direction tensors, layerwise projections/probes/LOTO/specificity, steering and projection-patching results, report, figures, and metadata. Full activations are streamed and discarded.

**Saved result.** only action_history at layer 30 passes the representation selection gate. The report classifies the outcome as representation without demonstrated causal role.

**Internal discrepancy.** The numbered questions report a quantitative steering correlation of 0.538, while the gate summary says quantitative steering failed/not run. Both appear in the same report and require interpretation before reuse.

**Status.** Ambiguous. The early probe/steering inference appears superseded by counterfactual DAS, but causal_mech_v1 consumes this stage's condition manifest. Moving the whole stage to legacy would break current provenance.

### 6.6 action_history_disambiguation_v1

**Question.** Is the selected late action-history direction upstream history, downstream persistence readout, or a mixture?

**Inputs.** The exact mechanistic_v1 layer-30 action vector, mechanistic condition manifest, and frozen behavioral coefficients.

**Outputs.** reproduced/raw/residualized/orthogonal/gradient directions at layers 8 and 30, matched representation tests, random-null steering, optional patching, and report.

**Saved result.** Gates 1–2 pass and Gates 3–4 fail; the registered outcome is H4_mixed_representation.

**Status.** Supporting diagnostic or extension. It is not consumed by the later core chain. Owner approval is needed before calling it legacy.

### 6.7 causal_mech_v1

**Question.** Can cognitive counterfactuals guide causal localization and a low-dimensional DAS alignment that reproduces held-out effects?

**Entry point.** cognitive-discovery-causal-mech and scripts/submit_causal_mechanistic.sh. CPU preparation/diagnostics precede a 32-layer localization array; an exact-size DAS array is dispatched only after Level 3 passes.

**Inputs.** Frozen theories from theory_resolution_v1 and the mechanistic_v1 condition manifest. The latter is a design bridge, even if the earlier mechanistic analysis is not retained as claim-bearing.

**Outputs.** frozen behavioral handoff, mechanistic counterfactual pairs and predictions, diagnostic decoding, whole-state patching, DAS rank-2/rank-8 alignments, validation/control tables, gates, five figures, and report.

**Saved result.** Three candidates pass the original Level-4 gate: latent_context layer 30 rank 8, dual_history/outcome target layer 28 rank 2, and outcome_history layer 30 rank 2. Specificity and necessity fail; circuits are not eligible.

**Status.** Canonical mechanistic-search candidate. Its original specificity conclusions use unstable per-example mean CFR and should be distinguished from the v2 reanalysis.

### 6.8 causal_specificity_v2

**Question.** Does the Level-4 result survive stable global CFR, and does any frozen candidate pass variable-specific controls and necessity?

**Entry point.** cognitive-discovery-specificity and scripts/submit_causal_specificity.sh.

**Inputs.** causal_mech_v1 source artifacts and all three frozen DAS candidates. The primary reanalysis performs no DAS retraining.

**Outputs.** source hashes, corrected global CFR, calibration and bootstrap intervals, random/shuffled/named controls, cross-variable tests, candidate gates, necessity tables, seven figures, and report.

**Saved result.** Corrected Level 4 passes for two candidates. The best untouched-test score is dual_history/outcome layer 28 rank 2 with global CFR 0.968; it beats 500 matched random subspaces but not the shuffled-target criterion. No candidate passes Level 5A. The stated outcome is an effective DAS controller without variable identity.

**Later audit correction.** paper/generated/results_inventory.json on origin/main shows that shuffled-target values did not materially change for candidates 1 and 2 because the target was essentially constant within task. Those nulls are uninformative, not evidence for or against specificity. The same audit finds zero submitted Level-5B necessity jobs. This conflicts with wording in the v2 report that answers target-selective necessity as “no” with a NaN mean; the accurate status for that stage is “not run.”

**Status.** Canonical corrective-analysis candidate, subject to the later integrity qualification.

### 6.9 abstraction_discovery_v1

**Question.** Does the frozen controller implement raw outcome history O, context-transformed history O-star, integrated history contribution H, or total evidence E?

**Entry point.** cognitive-discovery-abstraction and scripts/submit_causal_abstraction.sh.

**Inputs.** theory_resolution_v1, causal_specificity_v2, and all three frozen DAS objects. The primary object is the dual_history/outcome layer-28 rank-2 basis with SHA-256 ef8776641159fe67d02fdd1eae16d342305c4118ccb640114e9eb62cf4fedab0.

**Outputs.** a 50,000-candidate pool, 2,400 selected semantic conditions, six contrast families, O/O-star/H/E variables, natural geometry, interchange results, cross-abstraction matrices, controls, depth results, seven figures, gates, and report.

**Saved result.** E is the best natural-geometry and intervention-scoring label, but it fails the matched-random-subspace gate with p=0.667. The formal outcome is no_candidate_abstraction_passed and the justified claim remains Level 4 only.

**Later branch audit.** The unmerged computational-discrimination audit finds state overlap across abstraction partitions, strong predicted-effect correlations for some competing variables, concentration of 92.5 percent of E target energy in one contrast family, and unstable normalized recovery for the near-zero same-total-evidence family.

**Status.** Canonical abstraction candidate with a negative identity result. Calling E a “winner” without also stating that the overall abstraction gate failed would be misleading.

### 6.10 ood_free_generation_v1

**Question.** Does a frozen E-oriented structured-task subspace alter voluntary EOS stopping in free generation?

**Entry point.** cognitive-discovery-ood and scripts/submit_ood_generation.sh.

**Inputs.** abstraction_discovery_v1, frozen layer-28 rank-2 DAS, frozen E orientation, prompt manifest, doses, seeds, and random subspaces. The entry plan requires E to be the within-analysis winner but explicitly does not require the abstraction gate to have passed.

**Outputs.** frozen plan/hashes, 1,500 primary generation summaries and token events, pulse and fixed-topic analyses, EOS/random controls, survival and dose-response models, seven figures, gates, and report.

**Saved result.** all 1,500 primary runs terminate by EOS; increasing E does not significantly lower EOS hazard; dose monotonicity and random specificity fail; prompt and fixed-topic directions pass; the overall OOD generalization gate fails.

**Numerical validation.** Final gates mark alpha-zero, retained-cache replay, and native cache/no-cache equivalence as passed. The report's final generic sentence nevertheless says a native cache/no-cache discrepancy remains a failed strict gate. That sentence contradicts gates.json and the earlier numbered result.

**Provenance issue.** run_metadata.json contains git commit f0bef5d, while the final data/report were committed at af15830 after cache fixes at a297588. The metadata file itself was rewritten in af15830 without updating its embedded preparation commit. The recorded commit therefore does not uniquely identify the code that generated the final artifacts.

**Status.** Canonical frozen-generalization candidate, negative result. Whether it is a core claim or an extension is an owner decision.

## 7. Later manuscript, replication, and extension lane on origin/main

These files are visible on origin/main but absent from the checked-out worktree. They are scientifically important and must be resolved before reorganizing the repository.

### 7.1 Original sequential-study audit

**Scripts/artifacts.** scripts/audit_original_study.py; paper/original-study-audit.md; paper/generated/original_audit.

**Question.** Can claims from the earlier digital-minds-hackathon sequential bandit be independently recomputed from stored predictions and steering rows?

**Input.** External repository commit 5b968d0bb8de64f67556a7b300200aa488a591fa and eight Git-LFS steering shards.

**Output/result.** Arithmetic replay of future-return R-squared 0.239862, within-state relative-incentive R-squared 0.783644, persistence steering 1.993510, and near-zero return/advantage steering.

**Status.** Historical supporting evidence. It is a different task and estimand and must not be pooled with the new binary structured battery. It is also not independently reproducible from this repository alone because the full external input repository/LFS objects are required.

### 7.2 Qwen fixed-controller stability and narrow fresh histories

**Scripts/artifacts.** scripts/qwen_runpod_experiment.py, scripts/analyze_qwen_runpod.py, paper/runpod-analysis-plan.md, paper/runpod-results.md, and paper/generated/runpod_*.

**Question.** Is the fixed layer-28/rank-2 controller stable across optimization seeds and competitive with conventional/direct baselines on original and narrow new signed-history contrasts?

**Result.** Five seeds retain high original test/task-holdout global CFR but weak fresh-context calibration; target derangement correspondence is strong, while 20-random-subspace specificity is not. This is fixed-setting stability, not a repeat of the full discovery search.

**Status.** Supporting extension. The runner uses hard-coded relative upstream paths and requires the raw core artifacts; compact CPU replay outputs are committed.

### 7.3 Broader Qwen fresh contexts

**Scripts/artifacts.** scripts/qwen_fresh_contexts.py, scripts/qwen_fresh_natural_effects.py, scripts/analyze_qwen_fresh_contexts.py, paper/qwen-fresh-analysis-plan.md, paper/qwen-fresh-results.md, and paper/generated/qwen_fresh_contexts.

**Question.** Do all three frozen controllers generalize to new semantic contexts and signed histories?

**Result.** Cognitive-target recovery is weak, especially on task holdouts, while a natural-effect endpoint added during the run is strong for the primary controller. Because the natural endpoint was post hoc, it motivates but does not itself confirm that claim.

**Status.** Extension/exploratory diagnostic.

### 7.4 Independent Qwen confirmation

**Scripts/artifacts.** scripts/qwen_confirmation.py, scripts/analyze_qwen_confirmation.py, paper/next-experiment-protocol.md, paper/qwen-confirmation-results.md, and paper/generated/qwen_confirmation_v2.

**Question.** Does the frozen controller recover natural history effects on new contexts and two wording variants, beat 99 matched random bases, and outperform generic controls?

**Inputs.** Qwen revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, frozen core controller, and target-gain calibration learned only from the preceding fresh-context run.

**Result.** Natural-effect recovery is 0.779 on familiar neural tasks and 0.816 on neural task holdouts; both beat random controls. Cognitive-target recovery remains 0.251 and -0.428. Direct behavior-target DAS is slightly better in aggregate, so the run supports causal control but not unique cognitive-guidance superiority.

**Status.** Strong replication/extension candidate and likely claim-bearing, but not part of the checked-out core chain. Owner approval is required.

### 7.5 Llama feasibility, interface validation, behavior, and mechanism

**Scripts/artifacts.** scripts/llama_pilot.py, llama_label_validation.py, llama_order_diagnostic.py, llama_interface_validation.py, llama_behavior_replication.py, llama_mechanistic_replication.py and matching analysis scripts; paper/generated/llama_*; paper/llama-*.md.

**Questions.** First, can continuation be measured without label/order artifacts? Second, does a separately fitted Llama history model pass behavioral entry criteria? Third, does a Llama-specific causal alignment recover cognitive and natural effects?

**Sequence.** The original X/Y, A/B, and order-crossed pilots fail counterbalance validity and should remain as failed measurement history. A later complementary Yes/No interface passes calibration and untouched validation. A separate behavioral run selects dual_history with final R-squared 0.835 versus 0.821 for immediate state. A Llama-specific layer/rank search selects layer 24/rank 2.

**Mechanistic result.** Cognitive recovery is 0.649 on familiar tasks and 0.011 on neural task holdouts; natural-effect recovery is 0.345 and 0.319. Random comparisons pass, but generic controls recover natural effects better and norm matching has important reduced-precision discrepancies. This is a partial replication with a transfer boundary.

**Provenance.** Llama revision 0e9e39f249a16976918f6564b8830bc894c89659 is pinned in later scripts/results. Compact row-level exports and archive hashes are committed, but the full raw archives and model packages are external.

**Status.** Replication. The failed pilots are legacy/supporting validity records; the Yes/No behavioral and mechanistic runs are completed replication candidates.

### 7.6 Manuscript and paper-derived figures

**Files.** paper/main.tex, paper/experiment-reasoning-map.md, paper/framing-and-claims.md, scripts/paper_results.py, scripts/paper_figures.py, and paper/generated/results_inventory.json.

**Question.** Which stored results support the manuscript's bounded claims?

**Behavior.** paper_results.py hash-checks selected core files, rebuilds controller tables, audits shuffled-target integrity, and exposes missing necessity. paper_figures.py renders publication figures from the rebuilt table. Other analysis scripts rebuild later replication summaries from compact exports.

**Status.** Candidate claim layer. It is not presently in the checked-out tree. Some status prose is stale: paper/README.md still says the PR is unmerged although origin/main contains the merge; submission-readiness says confirmation is ongoing despite a completed confirmation report; llama-results.md says mechanistic replication has not run, while later documents correctly report it as complete. Historical failure documents should be clearly labeled rather than edited to erase chronology.

## 8. Proposed dependency graph

This is a provenance graph, not a final canonicality decision.

~~~mermaid
flowchart TD
    OLD[External digital-minds-hackathon\ncommit 5b968d0] --> OA[Independent original-study arithmetic audit]
    OA --> PAPER[Manuscript tables and claims]

    MODEL0[Qwen3.5-4B local checkpoint\nrevision unrecorded] --> D1[discovery_v1\nbehavioral design and collection]
    SPEC[Ontology + renderers + config] --> D1
    D1 --> D2[discovery_v2\nhierarchy + active collection + final validation]
    D1 --> TR[theory_resolution_v1]
    D2 --> TR
    TR --> MV1[mechanistic_v1\nmatched condition manifest + early probe/steer analysis]
    MV1 --> AH[action_history_disambiguation_v1\nextension]
    TR --> CM[causal_mech_v1\ncounterfactual pairs + localization + DAS]
    MV1 -->|condition manifest| CM
    CM --> CS[causal_specificity_v2\nstable global CFR + controls]
    TR --> CA[abstraction_discovery_v1]
    CS --> CA
    CA --> OOD[ood_free_generation_v1\nfrozen EOS test]

    D2 --> PAPER
    CM --> PAPER
    CS --> PAPER
    CA --> PAPER
    OOD --> PAPER

    CM --> QR[Qwen fixed-controller stability]
    CM --> QF[Qwen fresh contexts]
    QF --> QC[Independent Qwen confirmation]
    CM --> QC
    QR --> PAPER
    QF --> PAPER
    QC --> PAPER

    SPEC --> LP[Llama failed label/order pilots]
    LP --> LI[Llama Yes/No interface validation]
    LI --> LB[Llama behavioral replication]
    LB --> LM[Llama causal replication]
    LM --> PAPER

    CA --> CDA[Unmerged computational-discrimination audit]
    CDA --> PROP[Unexecuted discriminating follow-up protocol]
~~~

### 8.1 Core artifact edges

| Downstream stage | Required upstream data/artifacts | Where dependency is declared |
| --- | --- | --- |
| discovery_v2 | discovery_v1 standardized behavior and splits | Required round1-output CLI argument and Slurm env vars, not discovery_v2.yaml |
| theory_resolution_v1 | discovery_v1, discovery_v2 active collection, discovery_v2 final collection | CLI arguments and Slurm variables, not fully captured in YAML |
| mechanistic_v1 | theory-resolution handoff/frozen coefficients | CLI default theory-output and config/runtime metadata |
| action_history_disambiguation_v1 | mechanistic_v1 direction, conditions, coefficients | source_mechanistic_root in YAML and source hashes in run metadata |
| causal_mech_v1 | theory-resolution frozen models plus mechanistic_v1 conditions | YAML, CLI defaults, and frozen hashes/source manifest hash |
| causal_specificity_v2 | causal_mech_v1 alignments, pairs, localization, recovery | source_root and frozen_models/source_artifact_hashes.json |
| abstraction_discovery_v1 | theory_resolution_v1 plus causal_specificity_v2 | YAML theory_output/specificity_output and frozen DAS manifest |
| ood_free_generation_v1 | abstraction result, primary DAS, E orientation, prompts | YAML abstraction_output and frozen hash manifest |
| paper_results.py | selected DAS, specificity intervals/gates/shuffle shards, discovery_v2 metrics, abstraction/OOD gates | Hard-coded relative paths plus SHA-256 inventory |

## 9. Duplicated or competing implementations

| Area | Implementations | Scientific risk |
| --- | --- | --- |
| Mechanistic evidence | mechanistic_v1 probe/direction/steering workflow versus causal_mech_v1 counterfactual localization/DAS | Both are called “mechanistic,” but they answer different questions. The earlier condition manifest is still required downstream even if its inference is superseded. |
| Action history | mechanistic_v1 action-history direction, action_history_disambiguation alternatives, and causal_mech outcome/context targets | Similar layer/direction language can conceal distinct target definitions and causal tests. |
| Counterfactual recovery | per-example mean/median CFR in causal_mech_v1 versus global CFR in causal_specificity_v2 and later work | Near-zero targets can make old mean CFR unstable; reports and gates must state the version. |
| Mechanistic endpoint | Frozen cognitive counterfactual versus natural LLM source-minus-base effect | Later Qwen results are strong for natural effects and weak for cognitive targets; treating them as one endpoint reverses the interpretation. |
| Abstraction identity | O, O-star, H, E, with D explicitly equal to E | Some candidates are identical or highly correlated on many evaluated pairs. A winning pooled score does not establish a unique computation. |
| Target shuffling | Original within-task ID shuffles versus later explicit target-value-change audit/derangements | Two core shuffled-target controls change no target values, so the original control label overstates informativeness. |
| Behavioral hierarchy | Initial shared/task-specific/hierarchical fits, Round-2 M1–M4 hierarchy, and later Llama fixed M3 ridge | Similar architecture names do not imply identical fitting or validation regimes. |
| Measurement interface | Qwen X/Y semantic counterbalancing; failed Llama X/Y/A/B/order pilots; successful Llama complementary Yes/No polarity | Cross-model values are not directly comparable unless the interface change is explicit. |
| Later analyses | Reusable core metric/bootstrap functions versus standalone recovery/bootstrap code in paper scripts | Definitions currently agree in important cases but are duplicated, increasing drift risk. Do not consolidate until regression tests freeze saved outputs. |
| Figures | Every stage has its own Figure 1–7 naming; manuscript has separate figures | “Figure X” is ambiguous without an artifact root or manuscript label. |
| Original versus new study | Sequential three-option bandit persistence logit logsumexp(A,B)-C versus binary semantic continue-disengage logit | These are separate estimands and datasets and must never be pooled silently. |

## 10. Hidden defaults, hard-coded paths, and execution assumptions

### 10.1 Model and software defaults

- Every core YAML names Qwen/Qwen3.5-4B but sets model_revision to null.
- Core saved artifacts likewise record no immutable Qwen revision; an absolute local model directory is not a checkpoint identity.
- Later Qwen and Llama scripts improve this by hard-coding exact revisions, but those revisions are not propagated back into core metadata.
- The loader defaults to local_files_only, device_map auto, bfloat16, and trust_remote_code true.
- pyproject.toml pins scikit-learn 1.9.0 because pickles were serialized with it, but leaves NumPy, pandas, SciPy, Torch, Transformers, Accelerate, and PyArrow as lower-bounded ranges. uv.lock provides a resolution, yet documented pip editable installs need not reproduce it.
- Pickled behavioral models are therefore sensitive to Python and package versions. A local audit of origin/main emitted an InconsistentVersionWarning when scikit-learn 1.7.1 read a 1.9.0 pickle.

### 10.2 Scientific defaults in source

- History decay 0.7 is hard-coded in multiple behavioral and mechanistic modules.
- Many thresholds, layer lists, ranks, bootstrap counts, sample sizes, and fallback output roots are read with config.get defaults inside pipeline code. Omitting a YAML field does not necessarily fail closed.
- The behavioral target builder can fall back from a primary handoff to comparator frozen models when required coefficients are missing. The fallback order affects target provenance and is not summarized in reports.
- Response mapping is always included as a nuisance feature in cognitive fits even when absent from a model's named scientific features.
- The initial split algorithm falls back to arbitrary shuffled groups if the configured structural-holdout rules do not fill the desired quota.
- Round-2 and Round-3 upstream roots are required CLI values, not complete config fields.
- The OOD plan is the compute_efficient_initial profile, not necessarily every sample-size recommendation in the PRD.
- OOD entry requires the within-analysis E winner but sets require_abstraction_gate false. That scientific eligibility choice is explicit in the frozen plan but easy to miss in the report narrative.

### 10.3 Path and cluster defaults

- Della scripts assume /scratch/gpfs/JORDANAT/USER/... model and scratch roots.
- They assume module anaconda3/2025.6 and a Conda environment named llm-cognitive-discovery.
- Most GPU jobs request one nomig gpu40 GPU, four CPUs, and 64 GB; discovery_v1 requests gpu80. Wall limits range from six to 24 hours.
- Relative output roots and many paper-script inputs assume execution from the repository root.
- CLI defaults, YAML paths, Slurm environment defaults, and standalone paper-script paths duplicate the same dependency information.
- The later RunPod/Qwen/Llama runners write to untracked artifacts/qwen_* and artifacts/llama_* directories, while only compact paper/generated exports are committed.

### 10.4 Artifact retention defaults

- artifacts/.gitignore ignores everything except itself. Existing tracked artifacts remain, but new scientific outputs are silently ignored unless explicitly force-added.
- The size checker rejects tracked files above 10 MB and files that look like activation banks. This correctly protects the lightweight-artifact design.
- The checked-out commit's allowlist omits the 21 MB OOD token_events.parquet, so its artifact-size check fails. origin/main adds that reviewed exception.
- Full activations are not committed. Compact directions, low-rank bases, scalar projections, pair-level outcomes, and aggregate tables are retained.
- Later result documents reference full external reproducibility archives by SHA-256, but those archives do not have a repository-resolvable location.

## 11. Provenance coverage by stage

| Stage | Git commit | Model revision | Seeds | Dataset/pair hash | Behavioral-model hash | Neural artifact hash/layer/rank | Target and metric definition | Output root |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| discovery_v1 | Present | Missing/null | Present | Design hash present; standardized behavior file hash absent | Best-model pickle not content-hashed in run metadata | N/A | persistence_logit and model metrics in code | Present by config |
| discovery_v2 | Present | Missing/null | Present | Design/previous semantic hashes partial; collection file hashes absent | Pickle exists; no universal content hash | N/A | pooled and task-macro metrics coexist | Present by config |
| theory_resolution_v1 | Present | Missing/null | Present | Design hash present; selected/frozen files not all enumerated by hash in run metadata | Frozen packages/specs exist; hashes are consumed downstream | N/A | frozen comparison versus post-update is explicit | Present by config |
| mechanistic_v1 | Present | Missing/null | Present | Mechanistic condition hash present | behavioral_model_hash present | Layer count/position and direction files present; not every file hash centralized | target definitions and gates partial | Present by config |
| action-history v1 | Present | Missing/null | Present | Source manifest/run hashes present | Frozen coefficients declared | Original vector SHA and layers present | Gate definitions in config/code | Present by config |
| causal_mech_v1 | Present | Missing/null | Present | Source manifest and frozen-theory hashes present; pair hash not in run metadata | Frozen theory hashes present | Paths/layers/ranks present; SHA values centralized only downstream | Original mean CFR and split roles present in code | Present by config |
| causal_specificity_v2 | Present | Inherited but not repeated immutably | Present | Six source artifact hashes present | Behavioral source hash file exists | Frozen DAS manifest with hashes exists | global CFR primary; per-example descriptive | Present by config |
| abstraction v1 | Present | Missing/inherited | Present | Design counts recorded; not all input hashes centralized | Theory root recorded | Primary SHA, layer 28, rank 2 present | O/O-star/H/E and global CFR in code | Present by config |
| OOD v1 | Embedded commit is stale | Missing/null | Present | Frozen plan, prompt, job, sampling, random-basis hashes present | E orientation source identified | DAS/orientation hashes and layer/rank present | EOS hazard, survival, dose gates frozen | Present by config |
| Later Qwen confirmation | Script/protocol hashes in raw protocol; branch commit available | Exact Qwen SHA present | Present | Compact exports plus external archive SHA; raw conditions absent from repo | Frozen inputs referenced | Frozen controller plus control definitions | cognitive and natural global recovery explicit | Hard-coded artifacts path and paper export path |
| Llama replication | Script/protocol hashes in raw protocol; branch commit available | Exact Llama SHA present | Present | Compact exports plus external archive SHA | Llama model pickle hash recorded in raw run, not committed package | Selected layer 24/rank 2 and bases in external raw archive | cognitive/natural recovery explicit | Hard-coded artifacts path and paper export path |

The requested final provenance schema is therefore only partially present. No current record uniformly includes git commit, immutable model revision, seeds, dataset/pair hashes, behavioral model hash, neural artifact hash, layer/rank, target, metric, and output directory in one object.

## 12. Specific unclear or conflicting items

### 12.1 Model versions

1. Which exact Qwen revision generated discovery_v1 through OOD is unknown. All core model_revision fields are null.
2. The local folder may have changed between runs; no weight-directory hash or Hugging Face commit is stored.
3. Final OOD artifacts postdate the commit embedded in their metadata.
4. Later Qwen confirmation uses a pinned revision, but reproduction drift relative to the unpinned core is observed and explicitly reported.

### 12.2 Splits and units of independence

1. Behavioral pair/episode grouping is explicit, but each later stage introduces new split names and sampling units.
2. Neural task holdout does not mean behavioral zero-shot transfer in the later Qwen/Llama runs.
3. Response mappings and wording variants are repeated measurements, not independent semantic examples.
4. The unmerged abstraction audit reports that train/validation/test contrast partitions reuse many base/source states. The frozen basis predates that design, so this is not necessarily basis-training leakage, but uncertainty and “untouched state” language require correction.
5. Fresh-history RunPod diagnostics reuse one template per task/mapping and differ across mappings for several tasks; they do not establish response-label invariance.

### 12.3 Target definitions

1. Behavioral theory resolution retains dual_history, latent_context, and outcome_history; no unique behavioral winner is frozen.
2. mechanistic_v1's action-history representation is not the same target as the later outcome/context DAS candidates.
3. The primary layer-28/rank-2 basis was trained against a dual_history model's outcome-history counterfactual, not directly against total E or natural behavior.
4. E is the full fitted reference prediction; D equals E in code. Neither names a uniquely isolated upstream integration node.
5. Later confirmation primarily tests natural effects, whereas the core DAS search was trained against cognitive effects.
6. Fixed 0.7 traces and cue-reliability mixing are operationalizations, not learned latent-state algorithms.

### 12.4 Metrics and gates

1. discovery_v2's 0.862 is pooled R-squared; task-macro R-squared is 0.763.
2. causal_mech_v1 mean CFR and causal_specificity_v2 global CFR are competing metric generations.
3. Global CFR is sensitive to target-energy concentration; abstraction E is dominated by one contrast family in the later audit.
4. Near-zero targets require absolute error, not normalized recovery alone.
5. The core specificity report treats missing necessity as a failed result; later audit shows no necessity jobs were submitted.
6. Two shuffled-target controls are numerically unchanged and cannot support specificity conclusions.
7. Abstraction reports E as the best scoring candidate while the formal causal-abstraction gate fails.
8. OOD allows execution after a failed abstraction gate and then reports a negative OOD result. This is a frozen extension test, not validation of E identity.
9. Later Qwen and Llama random comparisons use different null counts and correction scopes; p-values are not directly comparable.

### 12.5 Figures and claims

1. Each artifact stage has local figure numbering. There is no checked-out global map from manuscript claim to exact figure and artifact.
2. origin/main's paper layer partially supplies that map but includes stale status documents and is absent from the worktree.
3. No CLAIMS.md exists.
4. The manuscript-facing “canonical Qwen result” could refer to original cognitive CFR, stable-CFR Level 4, independent natural-effect confirmation, or the negative abstraction/OOD boundary. This must be resolved explicitly.

## 13. Reproducibility and integrity checks performed

- The checked-out worktree was clean before this audit.
- The checked-out test suite passes: 134 tests passed in 39.10 seconds with pytest cache and bytecode writing disabled.
- The checked-out artifact-size check fails because artifacts/ood_free_generation_v1/generations/token_events.parquet is 21.16 MB and is not allowlisted at af15830. origin/main adds the exception.
- A clean archive of origin/main ran 144 tests in the local environment: 142 passed and 2 failed.
  - tests/test_confirmation_protocol.py requires artifacts/qwen_confirmation_v2/conditions.jsonl, which is not committed on origin/main. This is a clean-clone reproducibility failure, not merely a local package issue.
  - tests/test_llama_runner_compatibility.py failed while importing LlamaForCausalLM because the local Torch/Torchvision/Transformers combination lacks the expected torchvision operator. This demonstrates an environment-pinning gap; it may pass in the recorded GPU environment.
- origin/main's documented paper test command includes both failing tests above.
- The checked-out branch has only artifact-size CI; pytest is not run by GitHub Actions.
- Core artifact reports and metadata were cross-read against their gates, hashes, source paths, and Git history.
- No model inference or scientific recomputation was performed in Stage 1.

## 14. Missing inputs for independent reproduction

1. Exact model revision or weight hash for the core Qwen pipeline.
2. A fully pinned, documented environment matching each saved run; the core spans different dependency states even though artifacts record versions.
3. Raw collection shards for several behavioral stages. Standardized rows and design manifests are retained, which may be sufficient for analysis replay but not byte-for-byte collection reconstruction.
4. Full raw RunPod/Qwen/Llama follow-up archives. Reports contain archive hashes, but no durable retrieval location is declared in the repository.
5. The Qwen confirmation condition manifest expected by a committed test.
6. The older digital-minds-hackathon Git-LFS objects required by the independent original-study audit.
7. A single machine-readable manifest linking every final figure to exact source files, code commit, model revision, and metric definition.
8. A definitive statement of which branch is the canonical repository base.

## 15. Questions requiring owner approval before Stage 2

These are decisions, not recommendations silently applied by this audit.

1. **Repository base:** Should Stage 2 start from origin/main at 3a5c874, the checked-out af15830 tree, or the unmerged computational-discrimination branch?
2. **Behavioral chain:** Are discovery_v1, discovery_v2, and theory_resolution_v1 all canonical stages, or should discovery_v1 be treated only as an upstream dataset after v2 exists?
3. **Behavioral conclusion:** Should the canonical handoff retain all three frozen theories, or should one architecture be privileged for manuscript claims despite the unresolved frozen comparison?
4. **Early mechanism:** Is mechanistic_v1 claim-bearing, or should only its matched condition manifest remain in the canonical dependency chain while its probe/steering analysis becomes legacy?
5. **Action-history study:** Is action_history_disambiguation_v1 a core result, a supporting diagnostic, or an extension?
6. **Metric succession:** Does causal_specificity_v2 formally supersede all causal_mech_v1 specificity/necessity conclusions that used mean CFR, while retaining causal_mech_v1 as the source of trained alignments and Level-4 data?
7. **Missing necessity:** Should Level-5B be recorded as not run, and should any tables/figures implying an empirical necessity failure be relabeled?
8. **Uninformative shuffles:** Should the later paper audit be adopted as the canonical interpretation of candidates 1–2 shuffled-target controls?
9. **Abstraction result:** Is the canonical conclusion “E is the best descriptive candidate but no abstraction passes,” or should E be excluded from downstream naming because the random gate failed?
10. **OOD eligibility:** Is it scientifically intended that the OOD test proceed with require_abstraction_gate false? Is its negative result part of the core pipeline or an extension/boundary test?
11. **Later Qwen evidence:** Are RunPod stability, fresh contexts, and independent Qwen confirmation part of the final canonical Qwen claim set, or separate follow-ups?
12. **Primary mechanistic endpoint:** For the final claims, is the principal endpoint frozen cognitive-target recovery, natural-effect recovery, or both at explicitly separate evidential levels?
13. **Llama work:** Should the completed Yes/No behavior/mechanism work be maintained as a formal replication package, with earlier failed X/Y/A/B pilots under a measurement-validity history?
14. **Original study:** Should the digital-minds-hackathon sequential study remain external historical provenance, or be vendored/referenced as a required reproduction dependency?
15. **Figures:** Which stage-local figures correspond to the final manuscript figures and claims?
16. **Raw archives:** Where should external Qwen/Llama/raw-LFS archives be durably stored and how should access be documented?

## 16. Recommended CLAIMS.md shape after approval

Creating CLAIMS.md is strongly recommended, but doing so now would require choosing among the ambiguities above. After approval, each row should include at least:

| Field | Required content |
| --- | --- |
| Claim ID and exact wording | A bounded statement that matches the passed gates |
| Status | Supported, null, boundary, replication, extension, or unresolved |
| Analysis ID | Exact stage/protocol version |
| Model | Model ID plus immutable revision/weight hash |
| Behavioral object | Frozen model file/hash and target definition |
| Neural object | Basis hash, layer, rank, activation position, and intervention rule |
| Data | Manifest/pair/split hashes and independent sampling unit |
| Metric | Exact formula/version, aggregation, uncertainty, and gate |
| Artifact | Source table/JSON/Parquet and row/filter identity |
| Figure | Manuscript figure/panel, not only stage-local numbering |
| Command | One deterministic replay or rerun command |
| Limit | What the analysis does not establish |

An example should not yet be filled with a definitive canonical label. The current evidence would require, at minimum, separate entries for behavioral prediction, Level-4 cognitive counterfactual recovery, natural-effect confirmation, failed abstraction identity, partial Llama replication, and failed EOS generalization.

## 17. Stage-1 stopping point

The repository has enough retained lightweight evidence to reconstruct most numerical claims without committing activation banks. The main obstacles are not a lack of code; they are unresolved canonical scope, an unpinned core Qwen checkpoint, metric/target succession, missing external raw archives, a divergent branch state, and several report/gate contradictions.

No RED/GREEN/REFACTOR work should begin until the questions in Section 15 are answered. In particular, files should not be moved under legacy while live downstream dependencies still reference them.
