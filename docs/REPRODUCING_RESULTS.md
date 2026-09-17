# Reproducing results

This guide distinguishes CPU replay of committed evidence from regeneration of model outputs. CPU replay is the clean-clone reproducibility product. GPU regeneration is narrower because some historical identities or raw inputs were never recorded or committed.

## Installation

Python 3.10 or newer is required. The repository contains `uv.lock`; use it rather than resolving a fresh environment:

```bash
git clone https://github.com/mohanwugupta/Automated-Cognitive-Discovery-of-LLM-Persistence.git
cd Automated-Cognitive-Discovery-of-LLM-Persistence
uv sync --frozen --extra dev --extra parquet
source .venv/bin/activate
```

If `uv` is unavailable, `python -m venv .venv && pip install -e '.[dev,parquet]'` is supported for development but is not the locked installation. GPU reruns additionally need `uv sync --frozen --extra dev --extra parquet --extra qwen`, a compatible CUDA/PyTorch stack, and authorized model access.

## CPU-only replay

Replay all manuscript claims:

```bash
python -m cognitive_discovery.reproduce claims
```

Replay one claim and display its navigation:

```bash
python -m cognitive_discovery.reproduce claim C07
```

Verify one stage's hashed outputs and show its dependency closure:

```bash
python -m cognitive_discovery.reproduce stage causal_specificity_v2
```

Machine-readable output is available with `--json`. The complete CPU contract is:

```bash
python scripts/validate_canonical.py
pytest -q tests/provenance tests/regression
python -m cognitive_discovery.reproduce claims
pytest -q -m 'not gpu and not model_download'
```

Expected local runtime is seconds for claim replay, under a minute for the frozen provenance/regression suite, and roughly one minute for the ordinary CPU suite on a modern laptop. Exact timing depends on Parquet and scientific-library versions.

## GPU reruns

First run a dry validation:

```bash
python -m cognitive_discovery.run \
  --stage qwen_fixed_setting_stability_v1 \
  --config configs/canonical/qwen_fixed_setting_stability_v1.yaml
```

The command validates the stage against `canonical_manifest.yaml`, requires a pinned revision, verifies all committed input hashes, checks that the output cannot overwrite frozen evidence, and prints the delegated command and complete provenance record. To execute:

```bash
python -m cognitive_discovery.run \
  --stage qwen_fixed_setting_stability_v1 \
  --config configs/canonical/qwen_fixed_setting_stability_v1.yaml \
  --output artifacts/reruns/qwen_fixed_setting_stability_v1-NEW \
  --execute
```

Use a new output directory for every run. The runner writes `run_provenance.json` before model execution. Cluster scheduling remains the responsibility of the preserved `scripts/submit_*.sh` and `dispatch_*.sh` wrappers; do not request GPUs for CPU aggregation jobs.

The repository does not promise a single GPU runtime for every stage. Runtime depends on GPU type, scheduler load, pair count, model cache, and generation censoring. The fixed-setting job is a multi-seed neural refit and should be budgeted as an expensive accelerator experiment, not a laptop command. Historical Della resource requests are visible in the SLURM scripts, but they are historical settings rather than guaranteed wall-clock estimates.

## External dependencies

- Qwen checkpoint: `Qwen/Qwen3.5-4B`. Later confirmation uses revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Llama checkpoint: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`; access may require Hugging Face authorization.
- The original study requires external repository commit `5b968d0bb8de64f67556a7b300200aa488a591fa` and its LFS payloads.
- Full GPU work requires a compatible CUDA driver, PyTorch, Transformers, and adequate cluster storage for temporary activations. Large activation banks must remain outside Git.

## What cannot be fully regenerated from a clean clone

- Core Qwen stages C01–C05 have the checkpoint name but not the resolved historical revision. Their committed values are replayable; exact inference regeneration is not.
- C06's five refitted basis files are not committed, although its pinned protocol and compact metrics are.
- C07–C08 lack the raw Qwen confirmation condition manifest; committed intervention rows fully replay both endpoint metrics.
- C09 lacks raw inference and activation banks.
- C10 replays from committed predictions, not raw Llama inference.
- C11 lacks the exact selected Llama neural basis, so only compact metric replay is available.
- C12 is compact measurement-validity history.
- C13 requires the external historical repository and LFS data.

These are explicit `replay_status` or identity fields in `canonical_manifest.yaml`; no missing revision, basis, or raw input is inferred. See [CLAIMS.md](CLAIMS.md) for claim-specific limitations.

## Prospective model-agnostic replication

The historical replay commands above do not initialize a new model. For a prospective structured-task replication, first run the frozen Llama Phase-A regression and then use `python -m cognitive_discovery.replicate_model` with a new checkpoint's exact 40-hex model and tokenizer revision. The command is dry-run by default and writes only under a new `artifacts/replications/<run-id>` directory when `--execute` is supplied.

The complete protocol, stage-resume commands, and Della CPU/GPU submission chain are documented in [the replication guide](stages/replication/README.md). Phase B remains owner-gated until a genuinely new model and immutable revision are selected.
