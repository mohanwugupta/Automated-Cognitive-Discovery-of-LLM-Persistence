# External benchmark gate

`BENCHMARK_SPEC.md` is owner-frozen for the Sugawara–Katahira factual web
experiment. Fetch and verify Figshare Version 3 with:

```bash
cognitive-discovery-benchmark-fetch
cognitive-discovery-benchmark-fetch --execute
```

The first command is a dry run. The second writes the three public source files
and an acquisition record beneath `raw/figshare_10042319_v3/`. Raw files are
ignored by Git; `DATASET_MANIFEST.json` commits their immutable IDs, sizes, MD5s,
SHA-256 hashes, roles, and download URLs.

The analysis reads only `factual.csv`, but acquisition retains the authors'
counterfactual file and readme for source completeness.

The first full execution completed all 858 MAP fits. Exact-Hessian validation
then found six nonregular gradual-model fits at the active `tau=1` boundary.
Under the frozen positive-definite/no-jitter rule, Gate B is therefore
`not_evaluable` pending owner review—not `pass`, `partial`, or `fail`.
`REPLICATION_REPORT.md`, `benchmark_execution_status.json`, and
`fit_diagnostics.csv` preserve the run. No `results.csv` exists because no
primary contrast, bootstrap, or ranking was computed.

`NONBLOCKING_BOUNDARY_AMENDMENT.md` records the later owner decision:
benchmark-claim permission remains `stop`, while downstream Qwen/transfer
pipeline permission is `continue` because implementation equivalence passed.
`V2_BOUNDARY_METHOD_REQUIREMENTS.md` freezes admissibility requirements for a
possible v2 without selecting a post-result estimator.
