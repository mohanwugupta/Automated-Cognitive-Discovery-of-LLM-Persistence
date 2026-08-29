# Task renderers

Task-specific prose and semantic action names live in
`src/cognitive_discovery/experiments/`. Scientific factor levels and availability
remain centralized in `ontology/factors.py`; task configuration must not redefine
the experimental ontology or encode unavailable factors as zero.

