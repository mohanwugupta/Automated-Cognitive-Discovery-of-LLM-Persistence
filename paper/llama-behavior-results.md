# Llama response validation and behavioral replication

The original 4,060 arbitrary-label/order evaluations remain failed feasibility tests. The new interface changes the measurement to Yes/No answers about complementary proposed actions. This is a documented response-interface change, not a retroactive correction to earlier data.

## Interface validity

Three formats were compared on 210 new semantic situations, both question polarities (1,260 evaluations). The question format was selected using the fixed all-task validity rule. It then passed every inherited gate on 490 untouched situations (980 evaluations). Task mean polarity gaps ranged from 0.078 to 0.170, below the inherited 0.25 ceiling. The two phases total 2,240 evaluations.

## Separate behavioral entry test

The validated question format collected 1,400 training, 490 selection, and 490 final-test semantic situations, each in both polarities: 4,760 evaluations. Deterministic selection from a larger candidate pool excludes semantic duplicates across phases and prior interface-validity situations. No response-validity data enter the fits.

All models use shared and task-deviation M3 ridge, alpha 1 and deviation scale 0.5. Hyperparameters are fixed; only outcome-history versus dual-history architecture is selected using the selection set.

| Architecture | Selection R² | Final-test R² |
|---|---:|---:|
| Immediate state | 0.800677 | 0.821201 |
| Choice perseveration | 0.802659 | 0.822744 |
| Outcome history | 0.816537 | 0.833452 |
| Dual history | 0.819383 | 0.835387 |

Dual history passes the declared R² ≥ 0.6 and ≥0.01 improvement over immediate state on both independent phases. All final-test task-validity gates pass. The history increment is modest (0.0142 final-test R²), not evidence that history dominates all other factors. Behavioral calibration includes all seven tasks; a subsequent neural task holdout must not be described as zero-shot behavioral prediction.

Total completed Llama behavioral evaluations across the full sequence: **11,060**. The subsequent neural run is complete; see `llama-mechanistic-results.md` for its partial-replication result.

The compact model-prediction exports reproduce all scores using `scripts/analyze_llama_behavior.py`. Full interface and behavioral archives have SHA-256 values `0abb1f45f5954025ba9f449c07bb142d473037495a4e0f862f696513e18adacf` and `eb40792bd857b376e3f93bd24b53bc0c21cf48cc1d36efb5e180567cc0b40f86`, respectively.
