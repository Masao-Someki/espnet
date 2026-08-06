"""Search provider interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    """One web/document result."""

    title: str
    url: str
    snippet: str
    full_text: str | None = None


class SearchProvider(Protocol):
    """Search provider protocol."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search for documents."""
