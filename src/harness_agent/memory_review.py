"""Background memory review.

After every `nudge_interval` user turns the service fires an asyncio task
that runs a shadow LLM turn restricted to the `memory` tool. The shadow
turn never emits `AssistantTextProduced`; on completion it emits
`MemoryReviewCompleted(actions, note)` for telemetry.

Memory mutations from the review go through the same event bus as
foreground tool calls: the service publishes a `ToolCallRequested` event
with `generation=0` (review-scope marker) and awaits the matching
`ToolCallCompleted` via `ToolCallResultWaiter`. The conversation projector
and session log writer both skip non-current generations, so review-scope
mutations are auditable on the event bus but never leak into the
conversation transcript or the JSONL session log.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from uuid import uuid4

from harness_agent.bus import EventBus
from harness_agent.events import (
    AssistantTextProduced,
    EventBase,
    MemoryReviewCompleted,
    ToolCallCompleted,
    ToolCallRequested,
)
from harness_agent.llm import (
    AssistantToolCallMessage,
    LlmClient,
    LlmMessage,
    LlmRequest,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.tool_executor import ToolCallResultWaiter
from harness_agent.tools import MemoryToolInput, ToolRegistry
from harness_agent.turns import ConversationTurnCoordinator


logger = logging.getLogger(__name__)


EventBatch = tuple[EventBase, ...]

REVIEW_GENERATION = 0


_REVIEW_SYSTEM_PROMPT = (
    "You are reviewing a conversation to decide what is worth persisting "
    "into long-term memory. You have one tool — `memory` — and you may "
    "call it zero or more times. Save what would prevent the user from "
    "having to repeat themselves later: user persona, preferences, "
    "communication style, environment facts, conventions. Write "
    "declarative facts, not imperatives. Do not save task progress or "
    "completed-work logs. If nothing is worth saving, reply with the "
    "literal text 'Nothing to save.' and stop."
)

_REVIEW_USER_INSTRUCTION = (
    "Review the conversation above and consider saving to memory if "
    "appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, "
    "their work style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the `memory` tool (target "
    "'user' for who the user is, target 'memory' for environment / "
    "conventions / tool quirks). If nothing is worth saving, reply with "
    "'Nothing to save.' and stop."
)


class MemoryReviewService:
    """Counts user turns per conversation; spawns shadow reviews on threshold.

    Counter is a `defaultdict[str, int]` keyed by `conversation_id`. It
    increments on each `AssistantTextProduced` of the current generation
    and resets to 0 when (a) the threshold fires, or (b) a foreground
    memory tool call for the current generation completes. Review-scoped
    (generation 0) memory completions are intentionally ignored — the
    conversation may have advanced while the review was in flight and
    resetting there would silently drop a newer foreground increment.
    Stale-generation foreground completions are also ignored so this
    service stays in lockstep with `ConversationProjector` and
    `SessionLogWriter`.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        llm: LlmClient,
        tool_results: ToolCallResultWaiter,
        projection: SQLiteConversationProjection,
        tool_registry: ToolRegistry,
        turn_coordinator: ConversationTurnCoordinator,
        nudge_interval: int = 10,
        max_iterations: int = 5,
    ) -> None:
        self._bus = bus
        self._llm = llm
        self._tool_results = tool_results
        self._projection = projection
        self._tool_registry = tool_registry
        self._turn_coordinator = turn_coordinator
        self._nudge_interval = nudge_interval
        self._max_iterations = max_iterations
        self._counters: dict[str, int] = defaultdict(int)
        self._memory_used_in_generation: set[tuple[str, int]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        if not await self._is_current(event.conversation_id, event.generation):
            return ()
        # If a foreground memory tool call already fired in this same
        # generation, the assistant's wrap-up text must NOT re-increment
        # the counter. Without this guard, a foreground memory tool call
        # followed by the assistant's final text could cross the threshold
        # immediately and spawn a review of the work the model just did.
        key = (event.conversation_id, event.generation)
        if key in self._memory_used_in_generation:
            self._memory_used_in_generation.discard(key)
            return ()
        self._counters[event.conversation_id] += 1
        if self._counters[event.conversation_id] >= self._nudge_interval:
            self._counters[event.conversation_id] = 0
            self._spawn(event.user_id, event.conversation_id)
        return ()

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        if event.tool_name != "memory":
            return ()
        if event.generation == REVIEW_GENERATION:
            # Review-scope mutation finished. Do NOT touch the counter:
            # the conversation may have advanced while the review was in
            # flight, and a newer foreground AssistantTextProduced may
            # have already incremented the counter. Resetting here would
            # silently drop that increment and delay the next review by
            # one whole interval.
            return ()
        if not await self._is_current(event.conversation_id, event.generation):
            # Stale foreground completion — the conversation has moved
            # on to a newer generation. Mirror ConversationProjector and
            # ignore it.
            return ()
        self._counters[event.conversation_id] = 0
        self._memory_used_in_generation.add(
            (event.conversation_id, event.generation)
        )
        return ()

    async def wait_until_idle(self) -> None:
        """Wait for all in-flight review tasks to settle. Test helper."""
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _spawn(self, user_id: str, conversation_id: str) -> None:
        task = asyncio.create_task(
            self._run_review(user_id, conversation_id),
            name=f"memory-review:{conversation_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_review(self, user_id: str, conversation_id: str) -> None:
        try:
            history = await self._projection.list_llm_messages(conversation_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "memory review history fetch failed for %s: %s",
                conversation_id,
                exc,
            )
            return
        if not history:
            return
        try:
            memory_tool = self._tool_registry.by_name("memory")
        except KeyError:
            logger.warning("memory tool not registered; skipping review")
            return
        messages: list[LlmMessage] = list(history) + [
            UserMessage(text=_REVIEW_USER_INSTRUCTION)
        ]
        actions: list[str] = []
        note: str | None = None
        for _ in range(self._max_iterations):
            request = LlmRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=REVIEW_GENERATION,
                system=_REVIEW_SYSTEM_PROMPT,
                messages=messages,
                tools=[memory_tool],
            )
            try:
                response = await self._llm.respond(request)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("memory review llm call failed: %s", exc)
                return
            if response.kind == "assistant_text":
                note = response.text.strip() or None
                break
            if response.kind == "tool_call":
                if response.name != "memory":
                    logger.warning(
                        "memory review attempted unauthorised tool %s",
                        response.name,
                    )
                    break
                if not isinstance(response.input, MemoryToolInput):
                    logger.warning(
                        "memory review got non-memory input payload: %r",
                        response.input,
                    )
                    break
                requested = ToolCallRequested(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    generation=REVIEW_GENERATION,
                    call_id=uuid4().hex,
                    tool_name="memory",
                    input=response.input,
                )
                self._tool_results.expect(requested)
                await self._bus.publish(requested)
                completed = await self._tool_results.wait(requested)
                result = completed.result
                actions.append(
                    f"{response.input.action} {response.input.target}"
                    if result.exit_code == 0
                    else f"{response.input.action} {response.input.target} (rejected)"
                )
                messages = list(messages) + [
                    AssistantToolCallMessage(
                        call_id=requested.call_id,
                        name=response.name,
                        arguments=response.input.model_dump(mode="json"),
                    ),
                    ToolResultMessage(
                        call_id=requested.call_id,
                        name=response.name,
                        content=result.render_for_llm("memory"),
                    ),
                ]
                continue
        if actions or note:
            await self._bus.publish(
                MemoryReviewCompleted(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    actions=actions,
                    note=note,
                )
            )

    async def _is_current(self, conversation_id: str, generation: int) -> bool:
        return await self._turn_coordinator.is_current(conversation_id, generation)
