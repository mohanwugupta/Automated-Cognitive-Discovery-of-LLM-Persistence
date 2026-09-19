"""Validation of owner-frozen, human-readable mathematical specifications."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS


REQUIRED_SECTIONS = (
    "## Status",
    "## Variables",
    "## Equations",
    "## History and initialization",
    "## Fitting and prediction",
    "## Counterfactual",
    "## Fixed constants",
)


class SpecificationError(ValueError):
    """Raised when a mathematical specification is absent or not frozen."""


def specification_directory(root: str | Path) -> Path:
    return Path(root) / "docs" / "computational_models"


def validate_specifications(root: str | Path, *, require_frozen: bool = True) -> dict[str, str]:
    """Return SHA-256 identities after checking every registered specification.

    The documents, rather than production feature code, are the authority.  A
    draft document is useful for review but cannot open Gate A.
    """

    directory = specification_directory(root)
    hashes: dict[str, str] = {}
    problems: list[str] = []
    for name in sorted(COGNITIVE_MODELS):
        path = directory / f"{name}.md"
        if not path.is_file():
            problems.append(f"missing:{name}")
            continue
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        absent = [section for section in REQUIRED_SECTIONS if section not in text]
        if absent:
            problems.append(f"incomplete:{name}:{','.join(absent)}")
        if require_frozen and "specification_status: frozen" not in text:
            problems.append(f"not_frozen:{name}")
        hashes[name] = hashlib.sha256(payload).hexdigest()
    if problems:
        raise SpecificationError("computational-model specification gate failed: " + "; ".join(problems))
    return hashes
