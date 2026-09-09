# Independent audit of the original sequential study

Audited repository `mohanwugupta/digital-minds-hackathon` at immutable commit `5b968d0bb8de64f67556a7b300200aa488a591fa`. This is independent arithmetic on the stored data, not a rerun of original model inference or probe fitting. The new audit script does not call the original analysis functions.

| Claim | Independently recomputed result | Data checked |
|---|---:|---|
| Ridge future-return test R² ≈ 0.240 | 0.239862 | 1,800 unique states, 78 test episodes; stored train/validation/test episode sets disjoint |
| Relative-incentive within-state R² ≈ 0.784 | 0.783644 | 14,244 cells, 1,187 complete states, 48 episodes; fixed history and logit identity checked |
| Persistence steering contrast ≈ 1.9935 | 1.993510 | 224,343 intervention rows; complete target dose pairs; independent episode-weighted aggregation |
| Generic-return steering near zero | 0.002832; 95% bootstrap interval [-0.000974, 0.006827] | Same replay archive |
| Advantage steering near zero | -0.001172; 95% bootstrap interval [-0.006379, 0.004259] | Same replay archive |

The persistence-steering bootstrap interval is [1.977692, 2.010570], using 10,000 episode resamples with a new fixed seed. Interval endpoints need not exactly match the historical bootstrap seed. No bootstrap probability is reported as literally zero.

All eight large steering CSVs were downloaded from Git LFS and checked against their committed SHA-256 identifiers and byte counts. Input hashes, a compact state-effect export, and the independent arithmetic are included in `generated/original_audit/` and `scripts/audit_original_study.py`.

These checks upgrade the three quoted original numbers from merely attributed to independently recomputed from stored artifacts. They do not verify the original execution environment, pretraining data, hidden-state extraction, or absence of every possible upstream data leak. The original sequential task and later binary structured battery remain separate estimands and are not pooled.
