"""Per-tool-call futures and helpers keyed by `(conversation, generation, call)`."""

import asyncio

from harness_agent.events import EventBase, ToolCallCompleted, ToolCallRequested
from harness_agent.runtime.paths import safe_conversation_id_part


EventBatch = tuple[EventBase, ...]
ToolCallKey = tuple[str, int, str]


class ToolCallResultWaiter:
    """Awaits the `ToolCallCompleted` event for a previously registered call."""

    def __init__(self) -> None:
        self._pending: dict[ToolCallKey, asyncio.Future[ToolCallCompleted]] = {}

    def expect(self, event: ToolCallRequested) -> None:
        self._pending[tool_call_key(event)] = asyncio.get_running_loop().create_future()

    async def wait(self, event: ToolCallRequested) -> ToolCallCompleted:
        key = tool_call_key(event)
        future = self._pending[key]
        try:
            return await future
        finally:
            del self._pending[key]

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        key = tool_call_key(event)
        future = self._pending.get(key)
        if future is not None and not future.done():
            future.set_result(event)
        return ()


def tool_call_key(event: ToolCallRequested | ToolCallCompleted) -> ToolCallKey:
    return (event.conversation_id, event.generation, event.call_id)


def tool_result_spill_path(event: ToolCallRequested) -> str:
    # WHY `/` between components: safe_conversation_id_part percent-escapes
    # any `/` inside the encoded conversation_id / call_id, so `/` only ever
    # appears as our separator. A flat `-` join would collide on tuples like
    # (conv="cli-1", gen=2, call="a") vs (conv="cli", gen=1, call="2-a")
    # because `-` is in the percent-encoding safe set.
    return (
        "/workspace/content/tool-results/"
        f"{safe_conversation_id_part(event.conversation_id)}/"
        f"{event.generation}/"
        f"{safe_conversation_id_part(event.call_id)}.txt"
    )
