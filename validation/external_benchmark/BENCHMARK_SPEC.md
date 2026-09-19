# External computational-history benchmark specification

status: **OWNER-FROZEN — 2026-09-18**  
analysis_id: `sugawara_katahira_factual_v1`

These choices were frozen before inspecting any fitted benchmark result. They
must not be changed after seeing the benchmark results. Any future alternative
is a separately versioned analysis, not a repair of this one.

## Paper and source result

- Michiyo Sugawara and Kentaro Katahira (2021), “Dissociation between
  asymmetric value updating and perseverance in human reinforcement learning,”
  *Scientific Reports* 11:3574.
- DOI: `10.1038/s41598-020-80593-7`.
- Reproduced result: the factual-context model comparison for the authors’ web
  experiment, reported in Supplementary Table S2 and summarized in the main
  text. Figure 4 supplies the secondary learning-rate-bias result.

## Dataset identity and acquisition

- Figshare article: “Cognitive bias and perseverance”, article `10042319`,
  Version 3.
- Version DOI: `10.6084/m9.figshare.10042319.v3`.
- Public API: `https://api.figshare.com/v2/articles/10042319`.
- License: CC BY 4.0.
- Acquisition command:

  ```bash
  cognitive-discovery-benchmark-fetch --execute
  ```

- Raw files are downloaded under
  `validation/external_benchmark/raw/figshare_10042319_v3/`, verified, and
  excluded from Git. Immutable expected identities are committed in
  `DATASET_MANIFEST.json`.
- Primary analysis input: `factual.csv`, file ID `18106115`, MD5
  `b75a549db4227edda7bfd6d03e997728`, SHA-256
  `f80fc526606efbb1869af3c021d7caed6894881a8aae03aaff9093d13023b791`.

The counterfactual file and author readme are acquired and verified for source
completeness but are outside this benchmark’s analysis subset.

## Frozen subset and preprocessing

- Authors’ web experiment only.
- Factual-learning context only.
- All 143 retained participants after the published exclusions.
- Both factual sessions: 96 trials per session, 192 trials per participant.
- No additional participant exclusion.
- No additional trial filtering beyond the published dataset/task definition.
- File order is preserved within participant. Session is derived as trials
  1–96 and 97–192 because stimuli and values reset between the two sessions.
- Missed responses and false starts remain present and are handled according to
  the published task/model likelihood; they are not silently dropped.
- Initial action values and choice traces reset at the start of each session.

Expected raw factual data contract: 27,456 rows, 143 participants, 192 rows per
participant, and columns `subjectid,type,reward,choice,false_start,rt,key`.

## Frozen models

Fit these six models independently for every participant:

1. Standard RL
2. Asymmetry
3. Perseverance impulsive
4. Perseverance gradual
5. Hybrid impulsive
6. Hybrid gradual

The benchmark implementation is isolated from the project cognitive-model
registry. Project models must never be modified to improve benchmark agreement.

## Published mathematical and fitting contract

- Factual chosen-option Q-learning and softmax follow published Eqs. 1–2.
- The Asymmetry model uses separate positive/negative chosen-option learning
  rates as in Eq. 3.
- Choice probability and the choice-trace update follow Eqs. 4–5.
- Impulsive perseverance fixes `tau = 1`; gradual perseverance fits
  `0 <= tau <= 1`.
- Q values initialize to zero. Choice traces initialize to zero and reset with
  each session.
- Learning-rate prior: Beta(1.1, 1.1), constrained to `[0, 1]`.
- Inverse-temperature prior: Gamma(shape=1.2, scale=5.0), constrained to
  `beta >= 0`.
- Gradual-decay prior: Beta(1, 1), constrained to `[0, 1]`.
- Perseverance prior: Normal(mean=0, variance=5), constrained to `[-10, 10]`.
- Published optimizer: MAP using `solnp` from R package `Rsolnp`.
- Model comparison: participant-level Laplace-approximated log marginal
  likelihood (LML) under the published MAP specification and priors.

### Frozen independent-implementation details

The paper does not uniquely state every numerical setting. Before any benchmark
fit was inspected, the following implementation details were frozen:

- Eight deterministic `Rsolnp::solnp` starts per participant/model: one fixed
  central start plus seven seeded draws from the published priors, clipped only
  to the optimizer's open numerical interior, with seed `24001`.
- `tol=1e-8`, `delta=1e-8`, 400 outer iterations, and 1000 inner iterations.
- Rows with `choice=0` are retained in the dataset but contribute neither a
  choice likelihood nor a value/trace update. False-start rows with an observed
  choice remain in the likelihood and update sequence.
- Q values and choice traces reset to zero at each 96-trial session boundary.
- The Laplace Hessian is the exact float64 autodiff Hessian of the negative log
  posterior in the published parameterization at the best converged MAP.
- The Hessian must be finite and positive definite. No post-result eigenvalue
  flooring or jitter is allowed; a violation is an explicit fit failure.
- Numerical outputs retain 17 significant digits.

These unpublished details are implementation limitations. They may not be tuned
after viewing Gate-B contrasts.

### Pre-contrast numerical erratum

The initial integration implementation used the default Richardson Hessian from
`numDeriv`. Before any aggregate LML contrast or model ranking was computed, the
full integration pass showed that its default 10% relative perturbation crossed
the published `[0,1]` domain for valid near-boundary learning-rate MAPs, yielding
non-finite Hessian entries. This is a derivative-domain bug, not a scientific
result. The frozen correction uses an exact float64 autodiff Hessian at the
unchanged Rsolnp MAP. The MAP objective, priors, data, models, estimands,
positive-definite requirement, no-jitter rule, bootstrap, and Gate-B thresholds
are unchanged. The failed finite-difference integration attempt is retained in
the report as implementation history.

## Frozen primary estimands

Across participants:

```text
mean[LML(Perseverance-gradual) - LML(Asymmetry)]
mean[LML(Perseverance-gradual) - LML(Standard-RL)]
```

Uncertainty is a paired participant bootstrap with `B=2000` and a 95% percentile
interval. The bootstrap seed is frozen in
`configs/validation/external_benchmark_v1.yaml`.

## Frozen decision rule

Gate B is `pass` iff:

1. both primary mean contrasts are greater than zero;
2. both 95% bootstrap intervals have lower bounds greater than zero;
3. Perseverance-gradual is top-2 by mean LML; and
4. if another model ranks above Perseverance-gradual, it is Hybrid-gradual.

Gate B is `partial` iff both primary contrasts point in the published direction,
but one or more uncertainty/ranking criteria fail.

Gate B is `fail` iff either primary mean contrast is non-positive, or the
empirical ranking is clearly dominated by a non-history model.

## Secondary, nonblocking outputs

- Reduction of `alpha+ - alpha-` bias from Asymmetry to Hybrid-gradual.
- Fitted `tau` and `phi` distributions.
- Agreement with an independent/reference implementation.

## Failure diagnosis order and isolation

Diagnose discrepancies in this fixed order:

```text
preprocessing → equations → optimizer → likelihood → Laplace comparison → reporting
```

Do not modify core project models to improve benchmark performance.

Do not change these thresholds after seeing benchmark results. A faithful
negative or partial result remains a result.

## First frozen execution status

The first full execution completed all 858 participant/model Rsolnp MAP fits
with convergence code zero. The independent float64 implementation reproduced
the R log posterior to a maximum absolute discrepancy of `5.116e-13`, and 852
exact Hessians were positive definite.

Six gradual-model fits had `tau` at its active upper boundary and a
non-positive-definite unconstrained Hessian: Perseverance-gradual for
participants 22, 55, and 71, and Hybrid-gradual for participants 55, 71, and
105. This is a constrained-boundary Laplace problem rather than the earlier
finite-difference domain bug.

Accordingly, the execution status is `owner_review`, Gate B is
`not_evaluable`, and benchmark-claim permission is `stop`. No primary contrast,
bootstrap interval, ranking, or `pass | partial | fail` label was computed. The
run must not be completed post hoc by excluding participants, applying Hessian
jitter/eigenvalue flooring, taking an absolute determinant, or dropping the
boundary parameter.

The owner-frozen `NONBLOCKING_BOUNDARY_AMENDMENT.md` subsequently separated the
unresolved benchmark claim from downstream permission. Because implementation
equivalence passed, Qwen/transfer `pipeline_permission` is `continue`. If a
claim-bearing benchmark result is pursued, the authors' implementation must be
recovered or a separate, universally applied boundary-aware v2 must be
preregistered before aggregate results are examined.

See `benchmark_execution_status.json`, `fit_diagnostics.csv`, and
`REPLICATION_REPORT.md` for the machine-readable status, all fit diagnostics,
and the human-readable execution report.
