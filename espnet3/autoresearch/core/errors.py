"""AutoResearch error hierarchy."""


class AutoResearchError(Exception):
    """Base class for AutoResearch exceptions."""


class GraphRoutingError(AutoResearchError):
    """Raised when graph routing is invalid."""


class StageLoadError(AutoResearchError):
    """Raised when a stage class cannot be imported."""


class StageExecutionError(AutoResearchError):
    """Raised when a stage fails unexpectedly."""


class AgentResponseError(AutoResearchError):
    """Raised when an agent response cannot be parsed."""


class MetricNotFoundError(AutoResearchError):
    """Raised when a configured metric cannot be extracted."""


class JobSubmissionError(AutoResearchError):
    """Raised when a backend cannot submit a job."""


class StudyStateError(AutoResearchError):
    """Raised when persistent study state is inconsistent."""


class ConfigValidationError(AutoResearchError):
    """Raised when an AutoResearch config is invalid."""
