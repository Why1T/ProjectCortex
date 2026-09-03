"""Shared CLI helpers."""
from __future__ import annotations

from cortex.analysis.advisor import Advisor


def make_advisor() -> Advisor:
    return Advisor()
