"""JSONL session log writer.

Every user message, assistant message, and tool exchange in a conversation
appends one JSON line to `/workspace/sessions/<conversation_id>.jsonl`
inside the user's container. The `session.search` tool reads these files
to recall focused summaries of past conversations.

The writer subscribes to the same events `ConversationProjector` does, so
its writes happen in lockstep with the SQLite-backed conversation history.
"""

from __future__ import annotations

import json
from typing import Literal

from harness_agent.events import (
    AssistantTextProduced,
    EventBase,
    SessionLogAppendFailed,
    ToolCallCompleted,
    UserTextReceived,
)
from harness_agent.runtime import UserRuntime
from harness_agent.runtime.paths import safe_conversation_id_part
from harness_agent.turns import ConversationTurnCoordinator


__all__ = [
    "EventBatch",
    "SessionLogWriter",
    "safe_conversation_id_part",
]


EventBatch = tuple[EventBase, ...]


_Role = Literal["user", "assistant", "tool"]


class SessionLogWriter:
    """Append every conversation event to a per-conversation JSONL file.

    Append failures from the runtime are surfaced as a typed
    `SessionLogAppendFailed` event so observers can react instead of
    finding silent gaps in the JSONL log.
    """

    def __init__(
        self,
        *,
        runtime: UserRuntime,
        turn_coordinator: ConversationTurnCoordinator,
    ) -> None:
        self._runtime = runtime
        self._turn_coordinator = turn_coordinator

    async def handle_user_text(self, event: UserTextReceived) -> EventBatch:
        return await self._append(
            event.user_id,
            event.conversation_id,
            "user",
            {
                "role": "user",
                "source": event.source,
                "text": event.text,
                "occurred_at": event.occurred_at.isoformat(),
            },
        )

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        if not await self._is_current(event.conversation_id, event.generation):
            return ()
        return await self._append(
            event.user_id,
            event.conversation_id,
            "assistant",
            {
                "role": "assistant",
                "generation": event.generation,
                "text": event.text,
                "occurred_at": event.occurred_at.isoformat(),
            },
        )

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        if not await self._is_current(event.conversation_id, event.generation):
            return ()
        return await self._append(
            event.user_id,
            event.conversation_id,
            "tool",
            {
                "role": "tool",
                "tool_name": event.tool_name,
                "call_id": event.call_id,
                "input": event.input.model_dump(mode="json"),
                "exit_code": event.result.exit_code,
                "stdout": event.result.stdout,
                "stderr": event.result.stderr,
                "occurred_at": event.occurred_at.isoformat(),
            },
        )

    async def _is_current(self, conversation_id: str, generation: int) -> bool:
        return await self._turn_coordinator.is_current(conversation_id, generation)

    async def _append(
        self,
        user_id: str,
        conversation_id: str,
        role: _Role,
        record: dict[str, object],
    ) -> EventBatch:
        line = json.dumps(record, ensure_ascii=False)
        try:
            await self._runtime.append_session_log(user_id, conversation_id, line)
        except Exception as exc:
            return (
                SessionLogAppendFailed(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role=role,
                    error=str(exc) or type(exc).__name__,
                ),
            )
        return ()
