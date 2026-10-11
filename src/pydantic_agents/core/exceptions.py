"""Typed exception hierarchy for the Orgitration ecosystem."""


class OrgitrationError(Exception):
    """Base exception for all Orgitration errors."""


class ConfigurationError(OrgitrationError):
    """Invalid or missing configuration."""


class AgentNotFoundError(OrgitrationError):
    """Requested agent is not registered in the registry."""


class AmbiguousCapabilityError(OrgitrationError):
    """A capability is claimed by more than one agent and the caller did not pin an owner.

    Raised by ``AgentRegistry.query`` / ``instrumented_query`` when ``owner`` is
    omitted and the capability resolves to multiple agents. The exception carries
    the candidate owners so callers (or operators reading the trace) can pick one.
    """

    def __init__(self, capability: str, owners: list[str]):
        self.capability = capability
        self.owners = list(owners)
        super().__init__(
            f"Capability '{capability}' is claimed by multiple agents {self.owners}. "
            f"Pass owner=<agent_name> to disambiguate."
        )


class AgentExecutionError(OrgitrationError):
    """An agent's LLM call or tool execution failed."""

    def __init__(self, message, is_retryable=True):
        super().__init__(message)
        self.is_retryable = is_retryable


class PipelineError(OrgitrationError):
    """Pipeline-level failure that should abort the current run."""


class PipelineStepFailedError(PipelineError):
    """A specific pipeline step failed and cannot be retried."""


class ValidationError(OrgitrationError):
    """Code or data validation failed."""


class CheckpointError(OrgitrationError):
    """Checkpoint save/load/clear operation failed."""


class PathTraversalError(OrgitrationError):
    """Detected an attempt to write outside the allowed directory."""


class OrgitrationDependencyError(ImportError):
    """Raised when an optional dependency required by a runtime path is missing."""
