"""SubAgentRecord and the SubAgentLookup Protocol consumed by
AgentTurnHandler when filtering agent.* tools out of child turns."""

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel


SubAgentStatus = Literal["running", "completed", "failed", "cancelled"]


class SubAgentRecord(BaseModel):
    id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str
    status: SubAgentStatus
    result: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SubAgentLookup(Protocol):
    async def get_by_child_conversation_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None: ...


class NullSubAgentLookup:
    """No-recursive-subagent lookup for callers that intentionally do
    not want to filter agent.* tools out of child turns. Returning None
    keeps the full agent tool surface visible.

    Use only in tests that do not exercise sub-agent recursion.
    Production wiring passes the real SubAgentService."""

    async def get_by_child_conversation_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None:
        return None
