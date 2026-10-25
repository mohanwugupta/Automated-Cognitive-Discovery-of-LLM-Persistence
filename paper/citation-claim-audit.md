# Claim and citation audit

9 September 2026. This audit checks whether the manuscript's specific statements are supported by the source passages examined. It does not certify that every page or appendix of every citation has been read. Detailed coverage and source links are in `literature-review-revised.md`; local downloaded-source hashes are in the acquisition manifest.

## Manuscript claims

| Location | Supported claim | Required boundary / change |
| --- | --- | --- |
| Abstract and introduction | Cognitive targets guide a search yielding useful interventions | No claim that cognitive guidance uniquely identifies a natural program or outperforms direct DAS. |
| Behavioral methods | Feature models test history-sensitive accounts | Fixed 0.7 decay; prescribed histories; no inference of autonomous learning dynamics. |
| Counterfactual equation | Replace history inputs and recompute a frozen predictor | Corrected wording: not an independently identified intermediate-state intervention. |
| Main Qwen result | Selected contrasts have high cognitive recovery | Small sets, constant within-task rank-2 targets, fixed-setting seed replication remain explicit. |
| Independent confirmation | Natural-effect recovery 0.779/0.816 exceeds random controls | Cognitive recovery 0.251/-0.428; direct behavior-target DAS remains competitive. |
| Llama | Within-task cognitive recovery 0.649 | Task-holdout 0.011; generic controls better on natural effects; partial replication. |
| E interpretation | E is a candidate downstream quantity | Random p=0.667; D=E by definition. Added pooled-score weighting and predicted-effect discrimination audit. |
| EOS | Frozen intervention does not establish stopping control | Not evidence against every possible universal persistence computation. |
| Discussion | Cognitive models are testable guides | Removed wording implying uniqueness is a prerequisite for causal abstraction. |
| Reproducibility | Saved analyses and new experiments have artifacts | CPU replay is distinct from original inference/training reruns. |
| Submission | Qwen/Llama results complete | Removed stale “ongoing confirmation” entry; author approval remains pending. |

## Citation-specific checks

The claim column is the use in our manuscript, not a comprehensive summary of each paper.

| BibTeX key | Source claim used | Passages checked and interpretation |
| --- | --- | --- |
| mcguire2015persistence | Context-sensitive revaluation during waiting | Publisher abstract/figures and indexed author-manuscript text support this narrow claim. Direct PMC access still presents a CAPTCHA; full methods/supplement reading is not certified. |
| constantino2015foraging | Opportunity-cost learning in patch leaving | Task, MVT/TD comparison and learning methods. Our fixed feature models are not equivalent implementations of those algorithms. |
| sukhov2025quitting | Optimal quitting under uncertainty | Task, optimal-policy overview, human results and discussion. Finite horizon and irreversible abandonment distinguish it from ordinary bandit exploration. |
| klein2016comparison | Distinguishing reward/effort constituents from comparison | Independent variation of both options, conjunction and subjective-value results, discussion. This is human fMRI evidence, not a causal neural intervention or homologous LLM circuit. |
| pisauro2017evidence | Behavioral accumulation model informs neural dynamics | EEG/fMRI results and discussion of endogenous variability. Temporal neural evidence differs from our final-prompt logit measurement. |
| hagendorff2024machine | Controlled experiments and computational analysis of LMs | Framing/methodology passages; human constructs require adaptation. |
| binz2023psychology | Experimental design diagnoses strategy beyond performance | Vignettes, bandit/two-step and causal-task passages in the preprint. |
| aher2023simulate | Simulated human studies contain systematic distortions | Wisdom-of-crowds hyper-accuracy and risk/conclusion sections; population simulation is not validation of shared cognition. |
| shanahan2023roleplay | Role-play cautions against anthropomorphic interpretation | Introduction and framing; not a claim that computational modeling is impossible. |
| hayes2024biased | Relative reward encoding and computational learning models | Cognitive-model, hidden-state and conclusion sections; do not present as hidden-state intervention evidence. |
| schmied2025greedy | Greediness, frequency bias, effects of RL fine-tuning | Failure-mode analysis and overview; distinguishes training changes from inference-time interventions. |
| chen2025greedy | Good regret can coexist with poor exploration | Section 5.2 suffix failures and conclusion. Avoid equating low regret with comprehensive exploration. |
| zhang2026exploration | Exploration, perseveration and deliberation | Sections 4.1–4.3 model equations/estimation/recovery and 5.4–6 overview. The Kalman learner is more explicit than our fixed history trace. |
| everitt2025goals | Capability-relative pursuit of a prompted goal | Framework and discussion assumptions/limitations. Their score is not intrinsic motivation or task success alone. |
| arghal2026goals | Behavioral and representational goal-directedness tests | Introduction and conclusion. The paper also reports that simple activation patching does not reliably change behavior; causal linkage remains open. |
| geiger2025foundation | Corresponding interventions test causal abstraction | Approximate/interchange definitions and sections 3.6.3–3.7. Steering is representable in the framework; control alone does not establish internal reasoning. |
| geiger2024alignments | DAS learns distributed alignments to high-level variables | Main methods and equality/NLI analysis, including alternative decomposition. Counterfactual alignment is prior art. |
| wu2023scale | Boundless DAS scales search and compares algorithms | Main methods, pricing alternatives, transfer and controls. Our novelty cannot be “first theory-guided counterfactual search.” |
| makelov2024illusion | Subspace intervention success can mislead | Earlier preprint plus published ICLR passages checked separately. Published four-author version is cited; residual-stream DAS is not dismissed wholesale. |
| arditi2024refusal | Direction-based refusal work provides causal evidence | Extraction, addition, ablation, evaluation and weight editing. Expanded the bibliography's abbreviated author list from the PDF. |
| chen2025persona | Persona vectors support monitoring and causal steering | Sections 3, 8–9 and extraction overview. Supervised trait choice and mechanistic limitations matter. |
| xu2026reward | Sparse reward-related components have intervention evidence | Value-neuron ablation and transfer passages. Do not describe as probing-only. |
| gupta2026sisyphus | Original return-decoding/control dissociation | Original draft plus stored numeric audit. No original inference rerun is claimed. |

## Remaining reading and author verification

Every cited work has a documented, source-supported use above. This is not the same as comprehensive reading: unexamined methods, derivations and appendices remain in several sources, especially those explicitly marked initial/focused in the coverage log. A full accessible McGuire/Kable copy remains unresolved. No stronger claim should rely on those unread portions.

The authors still need to verify that these are the claims they intend, review the new audit and its post hoc status, confirm bibliographic versions, and approve the disclosure. The manuscript and this audit must not be described as author-approved until that review occurs.
