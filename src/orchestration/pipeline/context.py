from pydantic import BaseModel, Field


class ContextFrame(BaseModel):
    output: str = Field(description="Primary text summary of the agent's output")
    modified_files: list[str] = Field(default_factory=list, description="Paths of files created or modified")
    touched_symbols: list[str] = Field(default_factory=list, description="Codebase symbols touched")
    schema_changes: list[str] = Field(default_factory=list, description="Database schema migrations or updates")
    memory_notes: list[str] = Field(default_factory=list, description="Key insights or architectural decisions")
    command_traces: list[str] = Field(default_factory=list, description="Build/test command stdout or traces")
