"""No-op search provider."""

from __future__ import annotations


class NoOpSearchProvider:
    """Always returns no results."""

    def search(self, query: str, max_results: int = 5):
        return []
