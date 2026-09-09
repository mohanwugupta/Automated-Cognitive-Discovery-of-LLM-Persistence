# Llama behavioral and causal replication protocol

Specified after the Yes/No question interface passed calibration and independent validation, before collecting these new outcomes. No pilot/validity observations enter behavioral fitting or neural selection. This is a bounded replication of history-guided discovery; it is not a reproduction of every Qwen theory search or the full EOS battery.

## Behavioral entry

Generate independent training (1,400 semantic situations; seed 93001), model-selection (490; 93002), and final evaluation (490; 93003) sets. Generate a threefold candidate pool and deterministically retain the first unique situations per task, excluding prior interface-validity situations and earlier phases. Both question polarities; retain all seven tasks and histories. Pin the approved checkpoint and validated question renderer. Compare immediate-state, choice-perseveration, outcome-history, and dual-history architectures under M3 shared-plus-task-deviation ridge, fixed alpha 1 and random-effect scale 0.5. No within-pair random splitting or automatic row-wise cross-validation.

Choose the best of outcome-history and dual-history by model-selection R²; retain all comparisons. The chosen history model must exceed immediate-state R² by at least 0.01 and attain overall R² ≥ 0.6 on model-selection and independent final evaluation. Recheck every inherited measurement gate on the final evaluation. Failure stops neural discovery; report the complete behavioral result without tuning gates.

## Conditional neural search

If behavioral entry passes, freeze the selected model. Generate separate matched-history train, selection, and final designs using seeds 93101, 93102, 93103. Use four backgrounds/task, both signed history changes of lengths 3/5, both polarities. Neural training and selection use bandit, debugging, foraging, and solvability only; effort, waiting, and information sampling are neural task holdouts. Behavioral calibration includes all tasks, so this is neural task transfer, not zero-shot behavioral prediction.

Search layers 8, 16, 24, 28 and ranks 2, 8 with one declared seed, three epochs, Adam learning rate 0.03 and mean-square activation-change penalty 1e-4. Select highest cognitive-target global recovery on the selection set; tie-break lower rank then lower layer. Freeze the winner, evaluate once on the final set, and retain every candidate's selection score. Additional fixed-configuration seeds may be reported separately; they do not replace the original winner.

Evaluate frozen cognitive-target and natural-effect recovery on all final pairs. Include 99 fresh layer/rank/norm-matched random subspaces and decision/output controls fitted to neural training states only. Use task-stratified background bootstrap and Holm correction across familiar/held-out random comparisons. Record actual norm errors. Neither response validity nor behavioral fit alone counts as a successful mechanistic replication.
