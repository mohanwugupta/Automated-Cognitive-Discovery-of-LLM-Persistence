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
    test_rows = [row for row in generalization if row.get("split") == "neural_test"]
    holdout_rows = [
        row for row in generalization if row.get("split") == "neural_task_holdout"
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
    selected_models = behavior.get("selected_models") or [behavior.get("selected_model")]
    selected_models = [value for value in selected_models if value]
    controllers = mechanism.get("controllers", [])
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
        f"Frozen history-sensitive model set: `{selected_models}`.",
        "",
        "## 4. Frozen behavioral theory status",
        "",
        f"Status: `{behavior.get('theory_status', 'unknown')}`. Selection is frozen before neural fitting.",
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
        "## 10. Optional abstraction result",
        "",
        f"`{optional.get('abstraction', 'not_run')}`",
        "",
        "## 11. Optional OOD boundary result",
        "",
        f"`{optional.get('ood', 'not_run')}`",
        "",
        "## 12. Limitations",
        "",
        "Component outcomes remain separate. A failure or non-run optional boundary test does not "
        "retroactively change the structured-task result.",
        "",
        "## 13. Exact provenance hashes",
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
