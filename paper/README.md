# Cognitive-guided discovery of LLM persistence

This is a scientific working draft for author review. The contribution is a
pipeline from behavioral models to computational counterfactuals and held-out
neural interventions. The results support transferable control within structured
stay/disengage tasks, with substantial limits on computational identity and
unrestricted-generation transfer. The PR remains draft and unmerged.

## Start here

1. [Manuscript](main.tex): scientific argument, results, and limitations.
2. [Experiment reasoning map](experiment-reasoning-map.md): diagrams, hypotheses,
   actual manipulations, evidence, and remaining discriminating tests.
3. [Revised literature review](literature-review-revised.md): cognitive foundations,
   Geiger causal abstraction and DAS, relevant controls, and explicit reading coverage.
4. [Source lineage](source-lineage.md): relationship to the original Sisyphus draft.

The literature review distinguishes full/main-text reading from partial reading;
it is not a claim that every cited paper has been read in full.

## Completed evidence and its limits

| Evidence | Report | Interpretation |
| --- | --- | --- |
| Original sequential-bandit artifacts | [Audit](original-study-audit.md) | Stored-data replay, not a new inference run. |
| Fixed Qwen controller and new histories | [Initial follow-up](runpod-results.md) | Optimization stability; weak fresh cognitive calibration. |
| Broader Qwen contexts | [Context follow-up](qwen-fresh-results.md) | Natural-effect endpoint was added during this run; independent confirmation follows. |
| Independent Qwen confirmation | [Confirmation](qwen-confirmation-results.md) | Natural-effect recovery transfers and beats random controls; direct DAS remains competitive. |
| Llama measurement validation | [Failed pilots](llama-results.md), [valid Yes/No behavior](llama-behavior-results.md) | Failed label interfaces are retained; the redesigned interface passes. |
| Llama causal replication | [Mechanistic results](llama-mechanistic-results.md) | Partial replication; cognitive task-holdout recovery is weak and generic controls limit specificity. |
| Candidate E and frozen EOS test | [Manuscript](main.tex), [reasoning map](experiment-reasoning-map.md) | E fails its random control; the frozen EOS intervention fails the generalization test. |

The distinction between matching a cognitive counterfactual and reproducing a
natural neural effect is essential. Neither a successful intervention nor a
negative EOS test alone establishes the strongest computational interpretation.
No new GPU experiment is needed to inspect or replay the committed analyses.

## Reproduce the paper and final analyses

From the repository root, with Python 3.10+ and Tectonic installed:

```bash
python -m pip install -e '.[parquet,dev]'
python scripts/paper_results.py
python scripts/paper_figures.py
python scripts/analyze_qwen_confirmation.py
python scripts/analyze_llama_mechanistic.py
tectonic paper/main.tex
```

The final two analyses fall back to committed row-level CSV evidence when raw
Parquet output is absent. They regenerate summary tables on CPU. For an isolated
replay, copy the repository first; analysis commands intentionally rewrite their
summary outputs. Full model inference requires the separately declared protocol,
model access, pinned checkpoint, and GPU resources.

```bash
python -m pytest -q tests/test_paper_results.py tests/test_confirmation_protocol.py tests/test_qwen_runpod_protocol.py tests/test_llama_runner_compatibility.py
python scripts/check_artifact_sizes.py
```

The official ICLR style files are vendored. PDFs and TeX build intermediates are
untracked build outputs. GitHub collapses generated evidence through
`.gitattributes`; every evidence file remains available to expand or download.
Summary tables remain expanded for review.

## Protocols, historical notes, and remaining work

[Qwen confirmation protocol](next-experiment-protocol.md),
[broader Qwen plan](qwen-fresh-analysis-plan.md), and
[Llama replication protocol](llama-replication-protocol.md) preserve the decisions
made before their respective runs. [Initial RunPod plan](runpod-analysis-plan.md)
is historical context, not an instruction to launch more compute.

Remaining scientific work is to discriminate competing computational abstractions,
resolve the specificity and calibration limits, and complete author review of
claims and citations. Further experiments should follow that question, rather
than repeat a successful steering demonstration. See the reasoning map for the
proposed tests; these are not completed results.

[Submission readiness](submission-readiness.md) records administrative work for
the authors. It is separate from experimental completion and is not a completed
attestation or submission.
