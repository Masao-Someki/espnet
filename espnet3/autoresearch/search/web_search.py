"""Pluggable web search provider."""

from __future__ import annotations

import os
from urllib.parse import urlencode
from urllib.request import urlopen

from espnet3.autoresearch.search.provider import SearchResult


class WebSearchProvider:
    """Very small web-search adapter with optional providers."""

    def __init__(self, config):
        self.provider = str(getattr(config, "provider", "duckduckgo"))
        self.api_key_env = getattr(config, "api_key_env", None)
        self.max_results = int(getattr(config, "max_results", 5) or 5)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        max_results = int(max_results or self.max_results)
        if self.provider == "duckduckgo":
            params = urlencode({"q": query, "format": "json", "no_redirect": 1})
            try:
                with urlopen(f"https://api.duckduckgo.com/?{params}") as response:
                    payload = response.read().decode("utf-8", errors="replace")
            except Exception:
                return []
            return [
                SearchResult(
                    title=f"DuckDuckGo: {query}",
                    url="https://duckduckgo.com/",
                    snippet=payload[:1000],
                )
            ]
        api_key = os.environ.get(str(self.api_key_env or ""))
        if not api_key:
            return []
        return []
