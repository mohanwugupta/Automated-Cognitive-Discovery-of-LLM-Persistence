# Legacy and supporting analysis inventory

Stage 3 does not relocate scientific scripts merely for visual tidiness. Existing imports, artifact reconstruction paths, public commands, and cluster wrappers still refer to several historical files, so moving them would create compatibility risk without improving scientific provenance.

The canonical interpretation is:

- `mechanistic_probe_steering_v1`: supporting/legacy interpretation; its sibling matched-condition manifest remains canonical and live.
- `action_history_disambiguation_v1`: supporting diagnostic.
- `qwen_fixed_setting_stability_v1` and `qwen_fresh_contexts_v1`: supporting extensions.
- `llama_label_interface_history`: measurement-validity history, not a failed replication.
- `external_original_study`: external historical provenance.
- Superseded per-example CFR interpretations: legacy; use `global_cfr_v1`.

All corresponding scripts remain at their original paths as compatibility wrappers. Use `canonical_manifest.yaml` or `python -m cognitive_discovery.reproduce ...` to determine whether an analysis is canonical, supporting, boundary, replication, legacy, unresolved, not run, or an uninformative control. Frozen artifacts are not moved into this directory.
