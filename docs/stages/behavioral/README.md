# Behavioral and computational-theory stages

`discovery_v1` is the canonical upstream behavioral discovery/data stage. `discovery_v2` performs canonical model comparison and active discovery. `theory_resolution_v1` is the behavioral handoff: dual history, latent context, and outcome history remain unresolved.

The target is the signed semantic persistence logit. This stage contains no neural endpoint. Configs are in `configs/canonical/`; frozen outputs remain at their original `artifacts/` paths. Claims C01 and the behavioral part of C10 are the main entry points.

Replay C01 with `python -m cognitive_discovery.reproduce claim C01`. Exact core-Qwen inference regeneration is blocked because the historical resolved model revision is unknown.
