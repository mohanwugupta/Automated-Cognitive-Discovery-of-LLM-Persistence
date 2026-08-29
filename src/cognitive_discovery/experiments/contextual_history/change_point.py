"""Operational descriptions of same-environment and change-point conditions."""

CHANGE_POINT_TEXT = {
    "same_environment": (
        "The outcome-generating mechanism has not changed; observations from the "
        "matching environment remain diagnostic."
    ),
    "change_point": (
        "A monitored regime transition changed the outcome-generating mechanism at "
        "the phase boundary; observations from the obsolete regime are no longer diagnostic."
    ),
}


def change_point_text(level: str) -> str:
    try:
        return CHANGE_POINT_TEXT[level]
    except KeyError as error:
        raise ValueError(f"unknown change-point level: {level}") from error
