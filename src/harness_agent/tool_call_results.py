"""Shared tool-execution result model used by the router and handler modules."""

from pydantic import BaseModel, Field

from harness_agent.content import ContentRef
from harness_agent.runtime import RuntimeToolResult


class ToolExecutionResult(BaseModel):
    """Tool call output plus any attachments the model should see."""

    result: RuntimeToolResult
    attachments: list[ContentRef] = Field(default_factory=list[ContentRef])
