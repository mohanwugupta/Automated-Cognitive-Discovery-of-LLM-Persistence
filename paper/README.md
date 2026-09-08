# ICLR 2027 working draft

This manuscript reports the existing Qwen artifacts and a new read-only audit.
It is an early scientific draft, not a completed submission or a claim that
new GPU experiments have run. Do not merge automatically.

## Build

From the repository root, using Python 3.10+ with the project's dependencies,
PyArrow and Matplotlib:

```bash
python scripts/paper_results.py
python scripts/paper_figures.py
cd paper
tectonic main.tex
```

Alternatively run `pdflatex main`, `bibtex main`, and `pdflatex main` twice.
The official 2027 style and bibliography style are included unchanged from
[ICLR's style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip),
retrieved 2026-09-08. Generated PDFs and build intermediates are not source inputs.

## Audit findings that change the previous draft plan

1. **The history-target shuffle is uninformative.** In each rank-2 controller's
   stored test set, 0/24 target values change under the within-task permutation
   at tolerance 1e-10. Maximum change is below 1.2e-15. IDs change; target values
   do not. This test cannot establish or refute within-task specificity.
2. **Stable-CFR necessity was not run.** Its job manifest is empty because the
   preceding specificity gate failed. Blank values must not become measured
   nulls. Earlier exploratory necessity results are a different stage.
3. **Report all three retained controllers.** Test recovery is 0.930, 0.968,
   0.967. Task-holdout recovery is 0.673, 0.873, 0.878. The 0.808 figure is an
   across-controller mean, not the main controller's score.
4. **Behavioral validation metrics differ.** Pooled R-squared is 0.862;
   task-macro R-squared is 0.763. The interpolation macro score is 0.738.
5. **Current OOD cache gates pass.** The current committed JSON reports both
   cache checks passing; generic warning prose in older documentation does
   not override these artifacts. The OOD generalization gate still fails.
6. **The E abstraction still fails matched random controls.** Its p-value is
   0.667. Favorable geometry or descriptive recovery does not establish identity.

`generated/results_inventory.json` lists source hashes and extracted values.
`generated/audit.md` summarizes the row-level audit. Existing scientific
artifact directories are never modified by these scripts.

## Remaining work in priority order

| Priority | Work | Completion evidence |
|---|---|---|
| P0 | Freeze fresh within-task specificity contrasts with varied magnitudes/signs | Untouched manifest, hashes, target-variation and permutation-change audit |
| P0 | Same-data baseline comparison including direct behavior-target DAS, probe, mean difference, PCA, random | Matched rows, layer/rank protocol, paired test metrics, GPU/update/evaluation budgets |
| P0 | Five initialization/data-order DAS seeds | Every seed's test/holdout recovery and principal angles; no test-based seed selection |
| P0 | Complete Methods details, cognitive-model definitions and related-work coverage | Author-reviewed equations, citations, examples, counts and model revisions |
| P1 | Llama pilot then gated replication | Real semantic-token checks, numerical intervention checks, behavioral gate, Llama-specific frozen behavioral models |
| P0 | Final scientific and submission review | Checked claims, anonymous supplement, compiled PDF and complete author information |

The within-task null issue raises the priority of new contrast design. Merely
rerunning the old shuffle or adding random seeds cannot fix absent target
variation. The original and follow-up outcomes must remain distinct.

## Compute access and execution

The RunPod connector was verified by a successful empty Pod listing. The
console showed a $250 balance and an empty Secrets list on 2026-09-08. No Pod
was created during preparation. No rate or GPU-hour promise is inferred from
the earlier chat's stale price estimates.

To enable the Llama check, the account owner must sign into Hugging Face and
verify access to the requested model. Store a read token as a RunPod secret
(suggested name `hf_token`) and inject it as `HF_TOKEN` using
`{{ RUNPOD_SECRET_hf_token }}`. Do not put tokens in Git or chat. A browser login
alone is not sufficient for code running on a Pod.

Before reserving a GPU, execute this small HEAD-based check in an environment
with the secret injected:

```bash
python scripts/check_llama_access.py
```

Success returns `gated_weight_access_verified` and the model commit SHA. It
does not download weights. Use that SHA to pin the subsequent download.
Check current GPU stock and the full compute/storage quote, enforce a bounded
first run, and retain the $25 contingency from the original budget plan.
Do not provision an idle Pod while credentials or the experimental design
are unresolved. A stopped Pod can still bill for storage.

The existing participant uses AutoModelForCausalLM for Llama, and its hook
locator supports `model.layers`. A tiny random-Llama integration test passes
identity patching, nontrivial interchange, and alignment gradients. This is
software verification only: actual gated tokenizer behavior, 8B weights,
CUDA placement, and behavioral adequacy still require a real pilot.

## Submission administration

Sandy Tanwisuth owns OpenReview submission. Author identities must be finalized
by September 18, 2026; order may change until September 25. The abstract and
paper deadlines are 11:59 PM AoE on those dates. Main text is limited to nine
pages. Prepare a genuine abstract, update every author's OpenReview profile,
confirm reciprocal-reviewer eligibility/exemption, complete AI disclosure,
and anonymize paper and supplementary materials. See the
[official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

The abstract in `main.tex` is based only on existing results and the audit;
it does not promise completed Llama replication. The AI-use statement is a
draft for the authors to complete, not an attestation of finished review.
