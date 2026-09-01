"""High-identifiability condition generation and causal dissociation matching."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pandas as pd

from cognitive_discovery.design.counterbalance import counterbalanced_mappings
from cognitive_discovery.design.manifests import semantic_hash
from cognitive_discovery.design.sampling import balanced_factor_assignment
from cognitive_discovery.experiments.contextual_history.factors import (
    CUE_RELIABILITY,
    PHASE_ORDER,
    validate_contextual_history,
)
from cognitive_discovery.ontology.histories import HistorySpec
from cognitive_discovery.ontology.task_schema import ConditionSpec


CONTRAST_FAMILIES = (
    "same_raw_different_context",
    "same_context_different_raw",
    "same_history_contribution",
    "same_history_different_evidence",
    "same_total_evidence",
    "same_effect_different_cause",
)
CONTEXTUAL_TASKS = ("bandit", "foraging", "debugging")
EFFECT_BINS = (
    "strong_negative",
    "moderate_negative",
    "near_zero",
    "moderate_positive",
    "strong_positive",
)


_OUTCOME_PATTERNS = (
    (-1, -1, -1),
    (-1, 0, 1),
    (0, 0, 0),
    (1, 0, -1),
    (1, 1, 1),
    (-1, 1, 1),
)
_ACTION_PATTERNS = (
    ("disengage", "disengage", "disengage"),
    ("continue", "disengage", "continue"),
    ("continue", "continue", "continue"),
    ("disengage", "continue", "continue"),
    ("continue", "continue", "disengage"),
    ("continue", "disengage", "disengage"),
)
_CONTEXT_OUTCOMES = (
    (-1, -1, -1),
    (-1, 1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, -1),
)


def _valence(outcomes) -> str:
    values = tuple(map(int, outcomes))
    if all(value > 0 for value in values):
        return "positive"
    if all(value < 0 for value in values):
        return "negative"
    if all(value == 0 for value in values):
        return "neutral"
    return "mixed"


def _history(outcomes, actions) -> HistorySpec:
    return HistorySpec(3, _valence(outcomes), tuple(actions), tuple(map(int, outcomes)))


def _context(variant: int, *, critical_id: str) -> dict:
    a_index = int(variant) % len(_CONTEXT_OUTCOMES)
    b_index = (int(variant) * 5 + 1) % len(_CONTEXT_OUTCOMES)
    a_outcomes = _CONTEXT_OUTCOMES[a_index]
    b_outcomes = _CONTEXT_OUTCOMES[b_index]
    reliability_name = tuple(CUE_RELIABILITY)[(int(variant) // 2) % 3]
    context_return = ("A", "B", "novel_C")[int(variant) % 3]
    change_point = (
        "change_point" if (int(variant) // 3) % 5 == 4 else "same_environment"
    )
    context = {
        "phase_order": PHASE_ORDER,
        "a_history_valence": _valence(a_outcomes),
        "b_history_valence": _valence(b_outcomes),
        "a_history_actions": _ACTION_PATTERNS[a_index],
        "a_history_outcomes": a_outcomes,
        "b_history_actions": _ACTION_PATTERNS[b_index],
        "b_history_outcomes": b_outcomes,
        "context_return": context_return,
        "cue_reliability": reliability_name,
        "cue_probability": CUE_RELIABILITY[reliability_name],
        "change_point": change_point,
        "cue_rule": (
            "A calibrated diagnostic marker identifies whether the current "
            "generator matches phase A, phase B, or a novel generator."
        ),
        "a_environment_rate": 0.80,
        "b_environment_rate": 0.20,
        "critical_contrast_id": critical_id,
    }
    validate_contextual_history(context)
    return context


def generate_candidate_conditions(config: dict):
    """Generate at least 50k legal semantic states from the existing grammar."""

    settings = config.get("design", {})
    count = int(settings.get("candidate_conditions", 50_000))
    minimum = int(settings.get("minimum_candidate_conditions", 50_000))
    if count < minimum:
        raise ValueError(f"abstraction candidate pool requires at least {minimum} rows")
    tasks = tuple(
        settings.get(
            "task_families",
            (
                "bandit",
                "foraging",
                "solvability",
                "information_sampling",
                "waiting",
                "effort",
                "debugging",
            ),
        )
    )
    seed = int(config.get("seed", 73001))
    factor_levels = settings.get("factor_levels", {})
    first_mapping = counterbalanced_mappings(
        tuple(config.get("response_labels", ("X", "Y")))
    )[0]
    variants_per_template = 108
    conditions, metadata, observed = [], [], set()
    attempt = 0
    maximum_attempts = count * 20
    while len(conditions) < count:
        if attempt >= maximum_attempts:
            raise RuntimeError("could not generate the requested unique candidate pool")
        task = tasks[attempt % len(tasks)]
        task_index = attempt // len(tasks)
        template_id, variant = divmod(task_index, variants_per_template)
        raw_variant = variant % 6
        context_variant = (variant // 6) % 6
        cost_variant = (variant // 36) % 3
        factors = balanced_factor_assignment(task, template_id, seed, factor_levels)
        factors["continuation_cost"] = ("low", "medium", "high")[cost_variant]
        history = _history(
            _OUTCOME_PATTERNS[raw_variant],
            _ACTION_PATTERNS[(raw_variant + context_variant) % 6],
        )
        candidate_id = f"abs-candidate-{attempt:07d}"
        context = (
            _context(context_variant, critical_id=candidate_id)
            if task in CONTEXTUAL_TASKS
            else None
        )
        condition = ConditionSpec(
            design_id="abstraction_discovery_v1_candidate",
            condition_id=candidate_id,
            paired_condition_id=candidate_id,
            task_family=task,
            semantic_factors=factors,
            factor_available={
                name: value is not None for name, value in factors.items()
            },
            history=history,
            response_mapping=first_mapping,
            environment_seed=seed * 1_000_003 + attempt,
            sampling_strategy="abstraction_identifiability_pool",
            split="candidate_pool",
            contextual_history=context,
        )
        digest = semantic_hash(condition)
        attempt += 1
        if digest in observed:
            continue
        observed.add(digest)
        conditions.append(condition)
        metadata.append(
            {
                "condition_id": candidate_id,
                "semantic_sha256": digest,
                "template_id": f"{task}:{template_id:06d}",
                "raw_variant": raw_variant,
                "context_variant": context_variant,
                "cost_variant": cost_variant,
                "contextual_condition": context is not None,
            }
        )
    return conditions, pd.DataFrame(metadata)


def _standardize(frame: pd.DataFrame, names=("O", "O_star", "A", "H", "E")):
    output = frame.copy()
    scales = {}
    for name in names:
        values = output[name].to_numpy(dtype=float)
        scale = float(np.nanstd(values))
        scales[name] = scale if scale > 1e-12 else 1.0
        output[f"{name}_z"] = (values - float(np.nanmean(values))) / scales[name]
    return output, scales


def _pair_record(left, right, family, score, **extra):
    row = {
        "contrast_family": family,
        "base_semantic_id": left.condition_id,
        "source_semantic_id": right.condition_id,
        "task_family": left.task_family,
        "identifiability_score": float(score),
    }
    for name in ("O", "O_star", "A", "H", "E"):
        row[f"delta_{name}"] = float(right[name] - left[name])
    row["upstream_distance"] = float(
        np.linalg.norm([row["delta_O"], row["delta_O_star"], row["delta_A"]])
    )
    row["predicted_E_delta"] = row["delta_E"]
    row.update(extra)
    return row


def _largest_pair(part: pd.DataFrame, columns) -> tuple[pd.Series, pd.Series, float]:
    sample = part.sort_values("condition_id").head(36)
    values = sample[list(columns)].to_numpy(dtype=float)
    differences = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    left_index, right_index = np.unravel_index(
        np.argmax(differences), differences.shape
    )
    return (
        sample.iloc[left_index],
        sample.iloc[right_index],
        float(differences[left_index, right_index]),
    )


def _effect_bins(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, q=5, labels=EFFECT_BINS).astype(str)


def _balanced_top(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(["base_semantic_id", "source_semantic_id"]).copy()
    frame["effect_bin"] = _effect_bins(frame.predicted_E_delta)
    quota = int(math.ceil(int(count) / len(EFFECT_BINS)))
    selected = []
    for effect_bin in EFFECT_BINS:
        selected.append(
            frame[frame.effect_bin == effect_bin]
            .sort_values("identifiability_score", ascending=False)
            .head(quota)
        )
    output = pd.concat(selected, ignore_index=True).sort_values(
        "identifiability_score", ascending=False
    )
    if len(output) < count:
        used = set(zip(output.base_semantic_id, output.source_semantic_id))
        remainder = frame[
            [
                (base, source) not in used
                for base, source in zip(
                    frame.base_semantic_id, frame.source_semantic_id
                )
            ]
        ].sort_values("identifiability_score", ascending=False)
        output = pd.concat([output, remainder.head(count - len(output))])
    return output.head(int(count)).reset_index(drop=True)


def select_identifiability_contrasts(
    candidate_variables: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Select balanced pairs that force neighboring abstraction levels apart."""

    settings = config.get("design", {})
    per_family = int(settings.get("contrasts_per_family", 200))
    if per_family < 200 and not bool(settings.get("allow_smoke_counts", False)):
        raise ValueError("each abstraction contrast family requires at least 200 pairs")
    frame, scales = _standardize(candidate_variables)
    proposals = {family: [] for family in CONTRAST_FAMILIES}

    contextual = frame[frame.contextual_condition.astype(bool)]
    for _, part in contextual.groupby(
        ["task_family", "template_id", "raw_variant", "cost_variant"]
    ):
        if len(part) < 2:
            continue
        left = part.loc[part.O_star.idxmin()]
        right = part.loc[part.O_star.idxmax()]
        change = abs(right.O_star_z - left.O_star_z)
        if abs(right.O - left.O) <= 1e-10 and change > 0.2:
            score = change - 0.35 * abs(right.H_z - left.H_z)
            proposals[CONTRAST_FAMILIES[0]].extend(
                (
                    _pair_record(left, right, CONTRAST_FAMILIES[0], score),
                    _pair_record(right, left, CONTRAST_FAMILIES[0], score),
                )
            )

    for _, part in contextual.groupby(
        ["task_family", "template_id", "context_variant", "cost_variant"]
    ):
        if len(part) < 2:
            continue
        sample = part.sort_values("condition_id").head(24)
        best = None
        for left_index in range(len(sample)):
            for right_index in range(left_index + 1, len(sample)):
                left, right = sample.iloc[left_index], sample.iloc[right_index]
                raw_change = abs(right.O_z - left.O_z)
                context_change = abs(right.O_star_z - left.O_star_z)
                score = raw_change - 2.0 * context_change
                if best is None or score > best[0]:
                    best = (score, left, right)
        if (
            best is not None
            and best[0] > 0
            and abs(best[2].O - best[1].O) > 0.50
            and abs(best[2].O_star - best[1].O_star) < 0.40
        ):
            score, left, right = best
            proposals[CONTRAST_FAMILIES[1]].extend(
                (
                    _pair_record(left, right, CONTRAST_FAMILIES[1], score),
                    _pair_record(right, left, CONTRAST_FAMILIES[1], score),
                )
            )

    width = float(settings.get("match_bin_width_sd", 0.12))
    frame["H_bin"] = np.round(frame.H_z / width).astype(int)
    frame["E_bin"] = np.round(frame.E_z / width).astype(int)
    for _, part in frame.groupby(["task_family", "H_bin", "E_bin"]):
        if len(part) < 2:
            continue
        left, right, distance = _largest_pair(part, ("O_z", "O_star_z", "A_z"))
        score = distance - abs(right.H_z - left.H_z) - abs(right.E_z - left.E_z)
        if distance > 0.35:
            proposals[CONTRAST_FAMILIES[2]].extend(
                (
                    _pair_record(left, right, CONTRAST_FAMILIES[2], score),
                    _pair_record(right, left, CONTRAST_FAMILIES[2], score),
                )
            )
    for _, part in frame.groupby(["task_family", "H_bin"]):
        if len(part) < 2:
            continue
        left, right = part.loc[part.E_z.idxmin()], part.loc[part.E_z.idxmax()]
        change = abs(right.E_z - left.E_z)
        if change > 0.4:
            score = change - abs(right.H_z - left.H_z)
            proposals[CONTRAST_FAMILIES[3]].extend(
                (
                    _pair_record(left, right, CONTRAST_FAMILIES[3], score),
                    _pair_record(right, left, CONTRAST_FAMILIES[3], score),
                )
            )
    for _, part in frame.groupby(["task_family", "E_bin"]):
        if len(part) < 2:
            continue
        left, right, distance = _largest_pair(part, ("O_z", "O_star_z", "A_z", "H_z"))
        score = distance - 2.0 * abs(right.E_z - left.E_z)
        if distance > 0.5:
            proposals[CONTRAST_FAMILIES[4]].extend(
                (
                    _pair_record(left, right, CONTRAST_FAMILIES[4], score),
                    _pair_record(right, left, CONTRAST_FAMILIES[4], score),
                )
            )

    history_routes, cost_routes = [], []
    for _, part in frame.groupby(
        ["task_family", "template_id", "context_variant", "cost_variant"]
    ):
        if len(part) < 2:
            continue
        left, right = part.loc[part.E.idxmin()], part.loc[part.E.idxmax()]
        if left.raw_variant != right.raw_variant:
            history_routes.extend(
                (
                    _pair_record(left, right, "route", abs(right.E_z - left.E_z)),
                    _pair_record(right, left, "route", abs(right.E_z - left.E_z)),
                )
            )
    for _, part in frame.groupby(
        ["task_family", "template_id", "raw_variant", "context_variant"]
    ):
        if part.cost_variant.nunique() < 2:
            continue
        left, right = part.loc[part.E.idxmin()], part.loc[part.E.idxmax()]
        cost_routes.extend(
            (
                _pair_record(left, right, "route", abs(right.E_z - left.E_z)),
                _pair_record(right, left, "route", abs(right.E_z - left.E_z)),
            )
        )
    history_frame, cost_frame = pd.DataFrame(history_routes), pd.DataFrame(cost_routes)
    match_index = 0
    if len(history_frame) and len(cost_frame):
        for task, histories in history_frame.groupby("task_family"):
            costs = cost_frame[cost_frame.task_family == task]
            if costs.empty:
                continue
            cost_effects = costs.predicted_E_delta.to_numpy(dtype=float)
            for history in (
                histories.sort_values("identifiability_score", ascending=False)
                .head(per_family * 4)
                .itertuples()
            ):
                index = int(np.argmin(np.abs(cost_effects - history.predicted_E_delta)))
                cost = costs.iloc[index]
                mismatch = abs(cost.predicted_E_delta - history.predicted_E_delta)
                group = f"effect-match-{match_index:06d}"
                match_index += 1
                score = (
                    min(abs(history.predicted_E_delta), abs(cost.predicted_E_delta))
                    - mismatch
                )
                proposals[CONTRAST_FAMILIES[5]].extend(
                    (
                        {
                            **history._asdict(),
                            "contrast_family": CONTRAST_FAMILIES[5],
                            "identifiability_score": score,
                            "cause_route": "history",
                            "effect_match_group": group,
                            "route_effect_mismatch": mismatch,
                        },
                        {
                            **cost.to_dict(),
                            "contrast_family": CONTRAST_FAMILIES[5],
                            "identifiability_score": score,
                            "cause_route": "continuation_cost",
                            "effect_match_group": group,
                            "route_effect_mismatch": mismatch,
                        },
                    )
                )

    selected = []
    for family in CONTRAST_FAMILIES[:5]:
        part = _balanced_top(pd.DataFrame(proposals[family]), per_family)
        if len(part) < per_family:
            raise RuntimeError(
                f"identifiability pool produced only {len(part)} {family} pairs"
            )
        selected.append(part)
    route_frame = pd.DataFrame(proposals[CONTRAST_FAMILIES[5]])
    if route_frame.empty:
        raise RuntimeError("no same-effect/different-cause matches were available")
    route_summary = (
        route_frame.groupby("effect_match_group", as_index=False)
        .agg(
            identifiability_score=("identifiability_score", "first"),
            route_effect_mismatch=("route_effect_mismatch", "first"),
            predicted_E_delta=("predicted_E_delta", "mean"),
            route_count=("cause_route", "nunique"),
        )
        .query("route_count == 2")
        .sort_values(
            ["route_effect_mismatch", "identifiability_score"],
            ascending=[True, False],
        )
        .head(per_family)
    )
    chosen_groups = set(route_summary.effect_match_group)
    route_frame = route_frame[route_frame.effect_match_group.isin(chosen_groups)].copy()
    route_frame["effect_bin"] = _effect_bins(route_frame.predicted_E_delta)
    if route_frame.effect_match_group.nunique() < per_family:
        raise RuntimeError("too few paired causal-route matches were available")
    selected.append(route_frame)
    manifest = pd.concat(selected, ignore_index=True)
    manifest["contrast_id"] = [
        f"abs-{family}-{index:06d}"
        for index, family in enumerate(manifest.contrast_family)
    ]
    manifest["pair_split"] = "abstraction_train"
    split_key = manifest.effect_match_group.fillna(manifest.contrast_id)
    group_order = {value: index for index, value in enumerate(pd.unique(split_key))}
    split_cycle = {
        0: "abstraction_test",
        1: "abstraction_validation",
        2: "abstraction_train",
        3: "abstraction_train",
        4: "abstraction_train",
    }
    manifest["pair_split"] = [
        split_cycle[group_order[value] % 5] for value in split_key
    ]
    manifest["H_match_tolerance"] = scales["H"] * width * 1.51
    manifest["E_match_tolerance"] = scales["E"] * width * 1.51
    manifest["candidate_variables_frozen"] = True
    validate_contrast_manifest(manifest)
    return manifest.reset_index(drop=True)


def validate_contrast_manifest(frame: pd.DataFrame) -> None:
    """Verify the defining equality/inequality for all six contrast families."""

    observed = set(frame.contrast_family)
    if observed != set(CONTRAST_FAMILIES):
        raise ValueError(
            "contrast manifest does not contain all preregistered families"
        )
    failures = []
    for family, part in frame.groupby("contrast_family"):
        if family == CONTRAST_FAMILIES[0]:
            valid = (part.delta_O.abs() <= 1e-10) & (part.delta_O_star.abs() > 0.20)
        elif family == CONTRAST_FAMILIES[1]:
            valid = (part.delta_O.abs() > 0.50) & (part.delta_O_star.abs() < 0.40)
        elif family == CONTRAST_FAMILIES[2]:
            valid = (part.delta_H.abs() <= part.H_match_tolerance) & (
                part.upstream_distance > 0.50
            )
        elif family == CONTRAST_FAMILIES[3]:
            valid = (part.delta_H.abs() <= part.H_match_tolerance) & (
                part.delta_E.abs() > 0.20
            )
        elif family == CONTRAST_FAMILIES[4]:
            valid = (part.delta_E.abs() <= part.E_match_tolerance) & (
                part.upstream_distance > 0.50
            )
        else:
            group_counts = part.groupby("effect_match_group").cause_route.nunique()
            valid_groups = set(group_counts[group_counts == 2].index)
            valid = part.effect_match_group.isin(valid_groups)
        if not bool(valid.all()):
            failures.append((family, int((~valid).sum())))
    if failures:
        raise ValueError(f"abstraction contrast validation failed: {failures}")
    if not frame.candidate_variables_frozen.astype(bool).all():
        raise ValueError("contrast selection used mutable computational variables")


def expand_selected_conditions(conditions_by_id: dict, selected_ids, config: dict):
    labels = tuple(config.get("response_labels", ("X", "Y")))
    mappings = counterbalanced_mappings(labels)
    output = []
    for semantic_id in sorted(set(map(str, selected_ids))):
        original = conditions_by_id[semantic_id]
        for index, mapping in enumerate(mappings):
            output.append(
                replace(
                    original,
                    design_id="abstraction_discovery_v1",
                    condition_id=f"{semantic_id}-m{index}",
                    paired_condition_id=semantic_id,
                    response_mapping=mapping,
                    sampling_strategy="abstraction_identifiability",
                    split="selected",
                )
            )
    return output


def expand_contrast_mappings(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        for mapping_index, mapping_id in enumerate(("continue_x", "continue_y")):
            rows.append(
                {
                    **row,
                    "semantic_contrast_id": row["contrast_id"],
                    "pair_id": f"{row['contrast_id']}:m{mapping_index}",
                    "base_condition_id": f"{row['base_semantic_id']}-m{mapping_index}",
                    "source_condition_id": f"{row['source_semantic_id']}-m{mapping_index}",
                    "response_mapping": mapping_id,
                }
            )
    return pd.DataFrame(rows)


def shuffle_abstraction_targets(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    seed: int,
    magnitude_strata: int = 5,
) -> pd.DataFrame:
    """Derange semantic targets within task/magnitude strata, keeping mappings paired."""

    required = {
        "semantic_contrast_id",
        "task_family",
        "response_mapping",
        prediction_column,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"shuffle columns are absent: {sorted(missing)}")
    output = frame.copy()
    semantic = (
        output.groupby(["semantic_contrast_id", "task_family"], as_index=False)[
            prediction_column
        ]
        .mean()
        .sort_values("semantic_contrast_id")
    )
    semantic["magnitude_stratum"] = -1
    semantic["shuffled_target_contrast_id"] = None
    semantic["shuffled_prediction"] = np.nan
    rng = np.random.default_rng(int(seed))
    for task, task_part in semantic.groupby("task_family"):
        q = min(int(magnitude_strata), max(1, len(task_part) // 2))
        strata = pd.qcut(
            task_part[prediction_column].abs().rank(method="first"),
            q=q,
            labels=False,
            duplicates="drop",
        )
        semantic.loc[task_part.index, "magnitude_stratum"] = np.asarray(
            strata, dtype=int
        )
        for _, part in semantic.loc[task_part.index].groupby("magnitude_stratum"):
            if len(part) < 2:
                part = semantic.loc[task_part.index]
            indices = part.index.to_numpy()
            shift = int(rng.integers(1, len(indices)))
            donors = np.roll(indices, shift)
            semantic.loc[indices, "shuffled_target_contrast_id"] = semantic.loc[
                donors, "semantic_contrast_id"
            ].to_numpy()
            semantic.loc[indices, "shuffled_prediction"] = semantic.loc[
                donors, prediction_column
            ].to_numpy(dtype=float)
    mapping = semantic.set_index("semantic_contrast_id")
    output["original_predicted_effect"] = output[prediction_column]
    output["shuffled_target_contrast_id"] = output.semantic_contrast_id.map(
        mapping.shuffled_target_contrast_id
    )
    output[prediction_column] = output.semantic_contrast_id.map(
        mapping.shuffled_prediction
    )
    output["magnitude_stratum"] = output.semantic_contrast_id.map(
        mapping.magnitude_stratum
    )
    if (
        output.semantic_contrast_id.astype(str)
        == output.shuffled_target_contrast_id.astype(str)
    ).any():
        raise RuntimeError("shuffled abstraction targets contain fixed points")
    before = np.sort(output.original_predicted_effect.to_numpy(dtype=float))
    after = np.sort(output[prediction_column].to_numpy(dtype=float))
    if not np.allclose(before, after):
        raise RuntimeError(
            "shuffled abstraction targets changed their marginal distribution"
        )
    for _, part in output.groupby("semantic_contrast_id"):
        if part.shuffled_target_contrast_id.nunique() != 1:
            raise RuntimeError("response mappings were not shuffled together")
    return output


def condition_rows(conditions) -> list[dict]:
    return [condition.to_dict() for condition in conditions]


def write_condition_jsonl(path, conditions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, default=list) + "\n"
            for row in condition_rows(conditions)
        ),
        encoding="utf-8",
    )
