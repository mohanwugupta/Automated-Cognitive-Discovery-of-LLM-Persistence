"""Synthetic recovery of hierarchy variants at the seven-task sample size."""

from __future__ import annotations

import pandas as pd

from cognitive_discovery.models.fitting import regression_metrics
from cognitive_discovery.models.synthetic import (
    generate_hierarchical_teacher_data,
    generate_teacher_data,
)

from .random_effects import fit_hierarchical_model


def recover_hierarchy_variants(
    *,
    architecture="dual_history",
    n_train=700,
    n_test=280,
    seed=0,
) -> pd.DataFrame:
    rows = []
    for teacher in ("M1", "M2", "M3", "M4"):
        train, test = generate_hierarchical_teacher_data(
            teacher,
            n_train=int(n_train),
            n_test=int(n_test),
            seed=int(seed) + 101 * int(teacher[1]),
        )
        combined = pd.concat([train, test], ignore_index=True)
        for candidate in ("M1", "M2", "M3", "M4"):
            fit = fit_hierarchical_model(train, architecture, variant=candidate)
            rows.append(
                {
                    "teacher_architecture": architecture,
                    "teacher_variant": teacher,
                    "candidate_variant": candidate,
                    "evaluation": "known_task_interpolation",
                    **regression_metrics(test.persistence_logit, fit.predict(test)),
                }
            )
        for candidate in ("M1", "M3", "M4"):
            scores = []
            for heldout in sorted(combined.task_family.unique()):
                source = combined[combined.task_family != heldout]
                target = combined[combined.task_family == heldout]
                fit = fit_hierarchical_model(source, architecture, variant=candidate)
                scores.append(
                    regression_metrics(
                        target.persistence_logit,
                        fit.predict(target, include_random=False),
                    )["r2"]
                )
            rows.append(
                {
                    "teacher_architecture": architecture,
                    "teacher_variant": teacher,
                    "candidate_variant": candidate,
                    "evaluation": "strict_zero_shot_loto",
                    "r2": sum(scores) / len(scores),
                    "mse": float("nan"),
                    "correlation": float("nan"),
                }
            )
    for offset, teacher_architecture in enumerate(
        ("latent_context", "latent_motivation"), start=1
    ):
        train, test = generate_teacher_data(
            teacher_architecture,
            n_train=int(n_train),
            n_test=int(n_test),
            seed=int(seed) + 1000 + offset,
            noise=0.03,
        )
        for candidate_architecture in (
            "immediate_state",
            "choice_perseveration",
            "outcome_history",
            "dual_history",
            "dynamic_reevaluation",
            "latent_context",
            "latent_motivation",
        ):
            fit = fit_hierarchical_model(train, candidate_architecture, variant="M1")
            rows.append(
                {
                    "teacher_architecture": teacher_architecture,
                    "teacher_variant": "M1",
                    "candidate_variant": "M1",
                    "candidate_architecture": candidate_architecture,
                    "evaluation": "architecture_recovery",
                    **regression_metrics(test.persistence_logit, fit.predict(test)),
                }
            )
    result = pd.DataFrame(rows)
    result["candidate_architecture"] = result.candidate_architecture.fillna(
        architecture
    )
    best = (
        result.sort_values("r2", ascending=False)
        .groupby(
            ["teacher_architecture", "teacher_variant", "evaluation"],
            as_index=False,
        )
        .first()[
            [
                "teacher_architecture",
                "teacher_variant",
                "evaluation",
                "candidate_architecture",
                "candidate_variant",
            ]
        ]
        .rename(
            columns={
                "candidate_architecture": "best_candidate_architecture",
                "candidate_variant": "best_candidate",
            }
        )
    )
    result = result.merge(
        best,
        on=["teacher_architecture", "teacher_variant", "evaluation"],
        how="left",
    )
    result["teacher_recovered"] = (result.best_candidate == result.teacher_variant) & (
        result.best_candidate_architecture == result.teacher_architecture
    )
    return result
