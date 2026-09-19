"""Human-readable replication report with component-level outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _yes_no(value: bool) -> str:
    return "PASS" if value else "FAIL"


def build_replication_report(results: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    model = results.get("model", {})
    interface = results.get("interface", {})
    behavior = results.get("behavior", {})
    mechanism = results.get("mechanism", {})
    specificity = results.get("specificity", {})
    optional = results.get("optional", {})
    generalization = mechanism.get("generalization", [])
    aggregate_rows = [
        row for row in generalization if row.get("task_family") in (None, "")
    ]
    test_rows = [row for row in aggregate_rows if row.get("split") == "neural_test"]
    holdout_rows = [
        row
        for row in aggregate_rows
        if row.get("split") == "neural_task_holdout"
    ]
    per_task_holdout_rows = [
        row
        for row in generalization
        if row.get("split") == "neural_task_holdout"
        and row.get("task_family") not in (None, "")
    ]
    test_cfr = mechanism.get(
        "neural_test_global_cfr",
        {row.get("theory"): row.get("global_cfr") for row in test_rows} or None,
    )
    holdout_cfr = mechanism.get(
        "neural_task_holdout_global_cfr",
        {row.get("theory"): row.get("global_cfr") for row in holdout_rows} or None,
    )
    test_ci = mechanism.get(
        "neural_test_ci",
        {row.get("theory"): [row.get("cfr_ci_low"), row.get("cfr_ci_high")] for row in test_rows},
    )
    holdout_ci = mechanism.get(
        "neural_task_holdout_ci",
        {
            row.get("theory"): [row.get("cfr_ci_low"), row.get("cfr_ci_high")]
            for row in holdout_rows
        },
    )
    if generalization:
        test_passed = bool(test_rows) and all(
            row.get("global_cfr", float("-inf")) > 0
            and row.get("cfr_ci_low", float("-inf")) > 0
            and row.get("correlation", float("-inf")) > 0
            for row in test_rows
        )
        holdout_passed = bool(holdout_rows) and all(
            row.get("global_cfr", float("-inf")) > 0
            and row.get("cfr_ci_low", float("-inf")) > 0
            and row.get("correlation", float("-inf")) > 0
            for row in holdout_rows
        )
        controller_passed = bool(mechanism.get("controllers"))
    else:
        test_passed = bool(
            test_cfr is not None
            and test_cfr > 0
            and test_ci[0] is not None
            and test_ci[0] > 0
        )
        holdout_passed = bool(
            holdout_cfr is not None
            and holdout_cfr > 0
            and holdout_ci[0] is not None
            and holdout_ci[0] > 0
        )
        controller_passed = test_cfr is not None and test_cfr > 0
    selected_models = (
        behavior.get("behavioral_survivor_set")
        or behavior.get("selected_models")
        or [behavior.get("selected_model")]
    )
    selected_models = [value for value in selected_models if value]
    controllers = mechanism.get("controllers", [])
    controller_by_theory = {row.get("theory"): row for row in controllers}
    specificity_status = str(
        specificity.get(
            "status", "pass" if specificity.get("random_control_passed") else "fail"
        )
    )
    specificity_summary = (
        specificity_status.replace("_", " ").upper()
        if specificity_status in {"not_run", "incomplete_controls"}
        else _yes_no(bool(specificity.get("random_control_passed")))
    )
    controller_summary = (
        ", ".join(
            f"{row.get('theory')}=L{row.get('layer')}/rank{row.get('rank')}"
            for row in controllers
        )
        if controllers
        else f"L{mechanism.get('layer')}/rank{mechanism.get('rank')}"
    )
    test_by_theory = {row.get("theory"): row.get("global_cfr") for row in test_rows}
    holdout_by_theory = {
        row.get("theory"): row.get("global_cfr") for row in holdout_rows
    }
    specificity_by_theory = specificity.get("by_theory", {})
    theory_rows = [
        "| Theory | Behavioral test R² / MSE | Layer/rank | Familiar CFR [95% CI] | Whole-task CFR [95% CI] | Mapping robust | Random comparison | Specificity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    heldout_behavior = behavior.get("heldout_by_theory", {})
    for theory in selected_models:
        controller = controller_by_theory.get(theory, {})
        test_row = next((row for row in test_rows if row.get("theory") == theory), {})
        holdout_row = next(
            (row for row in holdout_rows if row.get("theory") == theory), {}
        )
        behavioral_metric = heldout_behavior.get(theory, {})
        specific = specificity_by_theory.get(theory, {})
        mapping_rows = specific.get("response_mapping_robustness", [])
        random_rows = specific.get("random_comparison", [])
        theory_rows.append(
            "| "
            + " | ".join(
                (
                    str(theory),
                    (
                        f"{behavioral_metric.get('r2'):.6g} / "
                        f"{behavioral_metric.get('mse'):.6g}"
                        if behavioral_metric
                        else "not_run"
                    ),
                    (
                        f"L{controller.get('layer')}/rank{controller.get('rank')}"
                        if controller
                        else "not_run"
                    ),
                    (
                        f"{test_row.get('global_cfr'):.6g} "
                        f"[{test_row.get('cfr_ci_low'):.6g}, {test_row.get('cfr_ci_high'):.6g}]"
                        if test_row
                        else "not_run"
                    ),
                    (
                        f"{holdout_row.get('global_cfr'):.6g} "
                        f"[{holdout_row.get('cfr_ci_low'):.6g}, {holdout_row.get('cfr_ci_high'):.6g}]"
                        if holdout_row
                        else "not_run"
                    ),
                    (
                        "pass"
                        if mapping_rows
                        and all(row.get("passes_response_mapping_control") for row in mapping_rows)
                        else "fail"
                        if mapping_rows
                        else "not_run"
                    ),
                    (
                        "pass"
                        if random_rows
                        and all(row.get("passes_random_control") for row in random_rows)
                        else "fail"
                        if random_rows
                        else "not_run"
                    ),
                    (
                        "pass"
                        if specific.get("passed") is True
                        else "fail"
                        if specific.get("passed") is False
                        else "not_run"
                    ),
                )
            )
            + " |"
        )
    per_task_rows = [
        "| Theory | Held-out task | CFR | Correlation | Slope |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(
        per_task_holdout_rows,
        key=lambda item: (str(item.get("theory")), str(item.get("task_family"))),
    ):
        per_task_rows.append(
            f"| {row.get('theory')} | {row.get('task_family')} | "
            f"{row.get('global_cfr')} | {row.get('correlation')} | "
            f"{row.get('slope')} |"
        )
    specificity_detail_rows = [
        "| Theory | Split | Mapping CFR / gap | Random mean / max / p | Named controls | Shuffle controls |",
        "|---|---|---|---|---|---|",
    ]
    for theory in selected_models:
        specific = specificity_by_theory.get(theory, {})
        mapping_lookup = {
            row.get("split"): row
            for row in specific.get("response_mapping_robustness", [])
        }
        random_lookup = {
            row.get("split"): row for row in specific.get("random_comparison", [])
        }
        named_lookup = {
            row.get("split"): row
            for row in specific.get("informative_specificity_controls", [])
        }
        for split in sorted(set(mapping_lookup) | set(random_lookup) | set(named_lookup)):
            mapping = mapping_lookup.get(split, {})
            random = random_lookup.get(split, {})
            named = named_lookup.get(split, {})
            specificity_detail_rows.append(
                f"| {theory} | {split} | "
                f"{mapping.get('response_mapping_cfr', 'not_run')} / "
                f"{mapping.get('response_mapping_cfr_gap', 'not_run')} | "
                f"{random.get('random_mean_global_cfr', 'not_run')} / "
                f"{random.get('random_max_global_cfr', 'not_run')} / "
                f"{random.get('finite_sample_random_p', 'not_run')} | "
                f"{named.get('named_control_global_cfr', 'not_run')} | "
                f"source={named.get('shuffled_source_status', 'not_run')}; "
                f"target={named.get('shuffled_target_status', 'not_run')} |"
            )
    historical = results.get("historical_comparison", {})
    prospective_controller = controller_summary
    historical_rows = [
        "| Quantity | Historical Qwen | Prospective Qwen |",
        "|---|---|---|",
        f"| Behavioral theory status | {historical.get('behavioral_theory_status', 'not_available')} | {behavior.get('theory_status', 'not_run')} |",
        f"| Surviving theories | {historical.get('surviving_theories', 'not_available')} | {selected_models} |",
        f"| Selected layer/rank | {historical.get('selected_layer_rank', 'not_available')} | {prospective_controller} |",
        f"| Familiar-task CFR | {historical.get('familiar_task_cfr', 'not_available')} | {test_cfr} |",
        f"| Whole-task CFR | {historical.get('whole_task_cfr', 'not_available')} | {holdout_cfr} |",
        f"| Specificity | {historical.get('specificity', 'not_available')} | {specificity_summary} |",
    ]
    transfer_by_theory = {}
    for row in holdout_rows:
        transfer_by_theory[str(row.get("theory"))] = bool(
            row.get("global_cfr", float("-inf")) > 0
            and row.get("cfr_ci_low", float("-inf")) > 0
            and row.get("correlation", float("-inf")) > 0
        )
    interpretation_rows = []
    for theory in selected_models:
        if transfer_by_theory.get(theory) is True:
            interpretation_rows.extend(
                [
                    f"### {theory}: Outcome A — prospective Qwen retains whole-task transfer",
                    "",
                    "This supports model-dependent differences: prospective Qwen transfers under the modern harness where Gemma/Llama did not.",
                    "",
                ]
            )
        else:
            interpretation_rows.extend(
                [
                    f"### {theory}: Outcome B — prospective Qwen loses whole-task transfer",
                    "",
                    "This suggests the older task-general Qwen result depended on the historical pipeline/design rather than reflecting a robust universal controller.",
                    "",
                ]
            )
    interpretation_rows.append(
        "Do not repair or tune this outcome to agree with historical Qwen."
    )
    lines = [
        "# Replication report",
        "",
        "## Component summary",
        "",
        f"Measurement interface:                  {_yes_no(bool(interface.get('passed')))}",
        f"Behavioral history-sensitive structure: {_yes_no(bool(behavior.get('passed')))}",
        "Behavioral theory uniquely resolved:    "
        + ("YES" if behavior.get("theory_status") == "resolved" else "NO / UNRESOLVED"),
        f"Low-dimensional causal controller:      {_yes_no(controller_passed)}",
        f"Held-out example generalization:        {_yes_no(test_passed)}",
        f"Held-out task-family generalization:    {_yes_no(holdout_passed)}",
        "Specificity vs random controls:         " + specificity_summary,
        f"Abstraction identity:                   {str(optional.get('abstraction', 'not_run')).upper()}",
        "Far-OOD continuation generalization:    "
        + str(optional.get("ood", "not_run")).replace("_", " ").upper(),
        "",
        "## 1. Model identity",
        "",
        f"- Model: `{model.get('id', 'unknown')}`",
        f"- Revision: `{model.get('revision', 'unknown')}`",
        f"- Adapter: `{model.get('adapter', 'unknown')}`",
        "",
        "## 2. Measurement-interface result",
        "",
        f"Selected `{interface.get('selected')}`; approved tasks: {interface.get('approved_task_count', 0)}.",
        "",
        "## 3. Behavioral model comparison",
        "",
        f"Behavioral replication status: `{behavior.get('replication_status', 'pass' if behavior.get('passed') else 'not_run')}`.  ",
        f"Behavioral best model: `{behavior.get('best_model', behavior.get('selected_model', 'unknown'))}`.  ",
        f"Frozen behavioral survivor set: `{selected_models}`.",
        "",
        "## 4. Frozen behavioral theory status",
        "",
        f"Status: `{behavior.get('theory_status', 'unknown')}`. The behavior-only survivor set is frozen before neural fitting; neural CFR cannot change its membership.",
        "",
        *theory_rows,
        "",
        "### Per-task whole-task-holdout recovery",
        "",
        *per_task_rows,
        "",
        "## 5. Mechanistic search grid",
        "",
        f"Selected controller(s): `{controller_summary}`. "
        "The fresh-run grid is defined by relative depth, not a fixed Qwen layer.",
        "",
        "## 6. Selected controller",
        "",
        f"Endpoint: `{mechanism.get('endpoint_id')}`; metric: `{mechanism.get('metric_id')}`.",
        "",
        "## 7. Held-out within-task cognitive recovery",
        "",
        f"Global CFR: {test_cfr}; interval: {test_ci}.",
        "",
        "## 8. Held-out task-family cognitive recovery",
        "",
        f"Global CFR: {holdout_cfr}; interval: {holdout_ci}.",
        "",
        "## 9. Specificity controls",
        "",
        f"Matched random-subspace comparison: {specificity_summary}.",
        "",
        *specificity_detail_rows,
        "",
        "Detailed random-null and intervention rows are also recorded in `replication_results.json` and `specificity/`.",
        "",
        "## 10. Optional abstraction result",
        "",
        f"`{optional.get('abstraction', 'not_run')}`",
        "",
        "## 11. Optional OOD boundary result",
        "",
        f"`{optional.get('ood', 'not_run')}`",
        "",
        "## 12. Descriptive historical comparison",
        "",
        "Historical values are context only and were not used for tuning or selection.",
        "",
        *historical_rows,
        "",
        "## 13. Critical interpretation",
        "",
        *interpretation_rows,
        "",
        "## 14. Limitations",
        "",
        "Component outcomes remain separate. A failure or non-run optional boundary test does not "
        "retroactively change the structured-task result.",
        "",
        "## 15. Exact provenance hashes",
        "",
        "```json",
        json.dumps(dict(provenance), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_replication_report(
    path: str | Path, results: Mapping[str, Any], provenance: Mapping[str, Any]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_replication_report(results, provenance), encoding="utf-8")
    return path
