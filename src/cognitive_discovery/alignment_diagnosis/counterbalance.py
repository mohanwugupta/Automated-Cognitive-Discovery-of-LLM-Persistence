"""Fully crossed timing design, native offline compiler, and SweetPea/SweetBean adapters."""

from collections import Counter
from itertools import product
import random

import numpy as np

from cognitive_discovery.cross_task.provenance import canonical_hash
from .design import _row


def design_spec(config):
    factors = {
        "correction": [0, 1],
        "history": ["favorable", "unfavorable"],
        "incentive_name": list(config["design"]["incentives"]),
        "scheduled_correction_decision": config["design"]["correction_decisions"],
    }
    return {
        "factors": factors,
        "crossing": list(factors),
        "backend": config["design"]["sampling_backend"],
        "constraints": [
            "complete crossing in every replicate",
            "correction twins share the same prefix and available rewards",
            "A/B prefix counts balanced across replicate pairs",
        ],
        "renderer": "role-preserving SweetBean-style event renderer",
        "events": ["controller_instruction", "action", "reward_feedback"],
        "max_decisions": 100,
        "absorbing_action": "C",
    }


def build_sweetpea_block(config):
    try:
        from sweetpea import CrossBlock, Factor
    except ImportError as error:
        raise RuntimeError(
            "SweetPea backend requires the optional sweetpea package"
        ) from error
    factors = [
        Factor(name, [str(v) for v in levels])
        for name, levels in design_spec(config)["factors"].items()
    ]
    return CrossBlock(factors, factors, [])


def _crossing(config, rng):
    spec = design_spec(config)
    if spec["backend"] == "sweetpea":
        block = build_sweetpea_block(config)
        from sweetpea import RandomGen, synthesize_trials

        state = random.getstate()
        try:
            random.seed(int(rng.integers(2**32)))
            sample = synthesize_trials(block, 1, RandomGen)[0]
        finally:
            random.setstate(state)
        return [dict(zip(sample, values)) for values in zip(*sample.values())]
    cells = [
        dict(zip(spec["factors"], values))
        for values in product(*spec["factors"].values())
    ]
    return [cells[i] for i in rng.permutation(len(cells))]


def crossed_episodes(config, *, phase, incentive=None):
    from copy import deepcopy

    c = deepcopy(config)
    if incentive is not None:
        c["design"]["incentives"] = {"stronger": incentive}
    seed = c["seeds"]["intervention" if phase == "intervention" else "design"]
    rng = np.random.default_rng(seed)
    rows = []
    for variant in range(c["design"][f"{phase}_variants"]):
        prefixes = {}
        for timing in c["design"]["correction_decisions"]:
            length = timing - 1
            prefix_rng = np.random.default_rng(
                seed + (variant // 2) * 1009 + timing * 100003
            )
            actions = (["A", "B"] * ((length + 1) // 2))[:length]
            prefix_rng.shuffle(actions)
            if variant % 2:
                actions = ["B" if a == "A" else "A" for a in actions]
            order = prefix_rng.permutation(length)
            favorable_successes = max(length // 2 + 1, round(0.75 * length))
            for history, successes in (
                ("favorable", favorable_successes),
                ("unfavorable", length - favorable_successes),
            ):
                rewards = np.full(length, -2)
                rewards[order[:successes]] = 3
                prefixes[timing, history] = (actions, rewards.tolist())
        for cell in _crossing(c, rng):
            timing = int(cell["scheduled_correction_decision"])
            history = cell["history"]
            name = cell["incentive_name"]
            actions, rewards = prefixes[timing, history]
            row = _row(
                variant=variant,
                history=history,
                actions=actions,
                rewards=rewards,
                incentive_name=name,
                incentive=c["design"]["incentives"][name],
                correction=int(cell["correction"]),
                phase=phase,
            )
            for key in ("pair_id", "trial_id"):
                row[key] = row[key].replace(f"_{variant}_", f"_{variant}_t{timing}_", 1)
            row.update(
                scheduled_correction_decision=timing,
                decision_index=timing,
                episode_max_decisions=100,
                episode_mode=True,
            )
            rows.append(row)
    return rows


def balance_audit(rows, config):
    chosen = [r for r in rows if r["trial_type"] == "choice"]
    counts = Counter(
        tuple(r[name] for name in design_spec(config)["crossing"]) for r in chosen
    )
    expected = int(np.prod([len(v) for v in design_spec(config)["factors"].values()]))
    expected_cells = set(product(*design_spec(config)["factors"].values()))
    arms = Counter(a for r in chosen for a in r["history_actions"])
    return {
        "backend": config["design"]["sampling_backend"],
        "cells": len(counts),
        "expected_cells": expected,
        "counts": [dict(cell=list(k), n=v) for k, v in sorted(counts.items())],
        "complete_crossing": set(counts) == expected_cells
        and len(set(counts.values())) == 1,
        "prefix_arm_counts": dict(arms),
        "arm_balanced": arms["A"] == arms["B"],
        "schedule_sha256": canonical_hash([r["trial_id"] for r in rows]),
    }


def build_sweetbean_preview(rows):
    """Optional human/LLM preview of critical states; local collection retains roles/logits."""
    try:
        from sweetbean import Block, Experiment
        from sweetbean.stimulus import Text
        from sweetbean.variable import TimelineVariable
    except ImportError as error:
        raise RuntimeError(
            "SweetBean preview requires the optional sweetbean package"
        ) from error
    timeline = [
        {
            "prompt": "\n\n".join(m["content"] for m in row["messages"]),
            "trial_id": row["trial_id"],
        }
        for row in rows
        if row["trial_type"] == "choice"
    ]
    stimulus = Text(text=TimelineVariable("prompt"), choices=["A", "B", "C"])
    return Experiment([Block([stimulus], timeline)])
