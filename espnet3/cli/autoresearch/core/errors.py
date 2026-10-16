"""AutoResearch error hierarchy."""


class AutoResearchError(Exception):
    """Base class for AutoResearch exceptions."""


class GraphRoutingError(AutoResearchError):
    """Raised when graph routing is invalid."""


class StageLoadError(AutoResearchError):
    """Raised when a stage class cannot be imported."""


class MetricNotFoundError(AutoResearchError):
    """Raised when a configured metric cannot be extracted."""


class ConfigValidationError(AutoResearchError):
    """Raised when an AutoResearch config is invalid."""
