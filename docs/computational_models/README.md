# Frozen computational-model specifications

These documents are the mathematical authority for `computational-model-spec-v1`.
They freeze the operational models originally defined in `docs/PRD.md` (hypothesis
bank and sharing assumptions). Production code is tested against these documents;
the documents must not be reverse-generated from production code.

Shared conventions: the response is the signed semantic persistence logit (D_t).
Every non-null model is linear in its listed feature map,
(D_t=\beta_0+\sum_j\beta_j z_{tj}+\beta_m m_t), where (m_t) is a
response-mapping nuisance regressor. Numeric feature values and availability
indicators are standardized with training-only means and standard deviations.
Ridge penalties are selected from `[.001, .01, .1, 1, 10]`. `fully_shared`,
`task_specific`, and `hierarchical` sharing use, respectively, one coefficient
vector, independent task vectors, and a shared vector plus task deviations.

History arrays are ordered oldest to newest. A trace is
(K(x_{1:n})=\sum_{k=0}^{n-1}0.7^k x_{n-k}); the newest observation has weight
one. Empty histories are missing (value zero plus availability indicator zero),
not numeric zero. A counterfactual effect is always
(\Delta D^{CF}=D(x_{source})-D(x_{base})) under one frozen fit.

Changing any equation, ordering, constant, target, nuisance term, standardization,
or counterfactual sign requires a new specification version and new validation.
