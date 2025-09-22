"""Utilities for calculating time specific durations for LibriSQA data."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DURATION_MINUTES = 5.0


@dataclass
class TimeSpecific:
    """Helper to decide the target duration for generated audio segments.

    LibriSQA data is generated with a default target duration of five minutes.
    When the available audio span is shorter than this default window, we use
    half of the available span length instead (e.g. for a one minute span we
    generate a thirty second segment).
    """

    span_minutes: float
    default_minutes: float = DEFAULT_DURATION_MINUTES

    @property
    def duration_minutes(self) -> float:
        """Return the target duration in minutes for the current span."""
        if self.span_minutes < self.default_minutes:
            return self.span_minutes / 2.0
        return self.default_minutes

    @property
    def duration_seconds(self) -> float:
        """Return the target duration in seconds for the current span."""
        return self.duration_minutes * 60.0
