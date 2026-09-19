"""Leakage-safe task-transfer planning, metrics, geometry, and synthesis."""

from .metrics import summarize_transfer
from .splits import assign_transfer_splits, audit_transfer_leakage

__all__ = ["assign_transfer_splits", "audit_transfer_leakage", "summarize_transfer"]
