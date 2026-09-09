# Manuscript lineage and integration decisions

User-supplied original: [Sisyphus in the loop: What Makes an LLM Persist?](https://apartresearch.com/project/sisyphus-in-the-loop-what-makes-an-llm-persist-yqd9), Apart Research, August 17, 2026.

[Full original PDF](https://framerforms.s3.us-east-1.amazonaws.com/form-uploads/32baa2cb-95ee-4054-87da-14f41117bd52-Sisyphus-in-the-loop_-What-Makes-an-LLM-Persist_.pdf). Original code: https://github.com/mohanwugupta/digital-minds-hackathon.

The PDF lists Mohan W. Gupta, Xingyu Shirley Liu, and Sandy Tanwisuth. The project landing page lists only Mohan; use the full PDF for historical attribution. This does not decide authorship/order of the expanded manuscript. The submission draft remains anonymous.

The rewrite develops the original question: does information decoded from hidden states causally determine persistence? Its narrative progresses from future-return decoding and a downstream decision positive control to computational history models, held-out neural interchange, specificity, and transfer. These are successive studies, not interchangeable measurements.

| Original material | Treatment in expanded manuscript |
|---|---|
| Sequential two-arm bandit with STOP | Retain as motivating prior study; distinguish logsumexp(A,B)-C from later binary semantic logits. |
| Future-return ridge test R2 .240 | Independently recomputed as .239862 from stored test predictions; original inference/training not rerun. |
| STOP/CONTINUE factorial within-state R2 .784 | Independently recomputed as .783644; do not pool 1,187 sequential states with later semantic contrasts. |
| Return/advantage steering nulls | Limit claims to fitted directions and calibrated intervention scale. Remove universal claims that future reward has no bearing on persistence. |
| Layer-31 persistence steering | Describe as decision-aligned positive control; this does not identify an upstream persistence computation. |
| Human waiting/foraging/quitting literature | Restore as hypothesis-generating literature, without inferring shared human/LLM mechanisms. |
| Goal-directedness, machine psychology, role-play, bandit and activation literature | Restore with primary-source checks. |
| Opening OpenAI/Hugging Face incident narrative | Do not carry forward without independently verified primary sources. It is unnecessary for the scientific argument. |
| Original references | Preserve source PDF; verified subset integrated now, remaining references require claim-level review. |

New RunPod experiment: fixed primary L28/rank2 controller, five optimization seeds, train-only conventional baselines, direct behavior-target DAS, and new signed history evaluation contrasts. This is a bounded follow-up, not a repeat of the full layer/rank search or a second-model replication. Fresh contrasts reuse one task/mapping template and are a narrow specificity diagnostic, not broad new task coverage.
