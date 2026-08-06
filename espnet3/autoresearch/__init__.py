"""ESPnet3 AutoResearch package."""

from espnet3.autoresearch.agents.interface import (
    AgentClient,
    AgentRequest,
    AgentResponse,
)
from espnet3.autoresearch.core.context import StageContext
from espnet3.autoresearch.core.result import StageResult, StageStatus
from espnet3.autoresearch.core.stage import AutoResearchStage
from espnet3.autoresearch.search.provider import SearchProvider

__all__ = [
    "AgentClient",
    "AgentRequest",
    "AgentResponse",
    "AutoResearchStage",
    "SearchProvider",
    "StageContext",
    "StageResult",
    "StageStatus",
]
