"""Cross-cutting tests for the memory plumbing.

Covers spots that the per-module test files don't hit:
- MEMORY.md content reaches the LLM system prompt.
- ToolRegistry's include flags actually filter `memory` / `session.search`.
- `safe_conversation_id_part` is injective for inputs that look similar
  after the old lossy regex (`tg:456` vs `tg-456` vs `tg/456`).
- `truncate_around_terms` centres on matches and falls back when there
  are none.
- `MemoryReviewService` ignores memory tool completions for stale
  generations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder, Skill
from harness_agent.events import (
    AssistantTextProduced,
    EventBase,
    MemoryReviewCompleted,
    ToolCallCompleted,
    ToolCallRequested,
)
from harness_agent.llm import (
    AssistantText,
    FakeLlmClient,
)
from harness_agent.memory_review import MemoryReviewService
from harness_agent.memory_service import MemoryService
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import RuntimeToolResult
from harness_agent.runtime.fake import FakeUserRuntime
from harness_agent.runtime.paths import safe_conversation_id_part
from harness_agent.session_search_service import truncate_around_terms
from harness_agent.store import SQLiteEventStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    MemoryToolInput,
    default_tool_registry,
)
from harness_agent.turns import ConversationTurnCoordinator


# ---------- MEMORY.md is reachable in the system prompt ----------


@pytest.mark.asyncio
async def test_context_builder_includes_memory_md_content() -> None:
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(
            soul="SOUL",
            agents="AGENTS",
            user="USER",
            tools="TOOLS",
            memory="Project uses uv for env management.\n§\nCI runs pyright strict.",
        ),
        skills=(),
    )
    builder = ContextBuilder(runtime=runtime)
    context = await builder.build("u:1")
    assert "Project uses uv for env management." in context.system
    assert "CI runs pyright strict." in context.system


@pytest.mark.asyncio
async def test_context_builder_omits_empty_memory_payload() -> None:
    """When AgentFileSet.memory is empty the assembled system prompt
    must NOT contain a stray double newline marking a missing block."""
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T")
    )
    builder = ContextBuilder(runtime=runtime)
    context = await builder.build("u:1")
    # The three set files appear in order with `\n\n` between them; no
    # blank slot for a missing memory block.
    assert "S\n\nA\n\nU\n\nT" in context.system
    # ...but the user-facing guidance that names the file does appear.
    assert "Persistent memory:" in context.system


# ---------- ToolRegistry include flags ----------


def test_default_registry_includes_memory_and_session_search() -> None:
    names = {tool.name for tool in default_tool_registry().tools}
    assert "memory" in names
    assert "session.search" in names


def test_registry_can_exclude_memory_only() -> None:
    names = {
        tool.name for tool in default_tool_registry(include_memory=False).tools
    }
    assert "memory" not in names
    assert "session.search" in names


def test_registry_can_exclude_session_search_only() -> None:
    names = {
        tool.name
        for tool in default_tool_registry(include_session_search=False).tools
    }
    assert "session.search" not in names
    assert "memory" in names


def test_registry_can_exclude_both() -> None:
    names = {
        tool.name
        for tool in default_tool_registry(
            include_memory=False,
            include_session_search=False,
        ).tools
    }
    assert "memory" not in names
    assert "session.search" not in names
    # Other tools still present.
    assert "shell.exec" in names


# ---------- safe_conversation_id_part injectivity ----------


def test_safe_conversation_id_part_distinguishes_colon_slash_dash() -> None:
    """Distinct conversation_ids that share characters after lossy stripping
    must still map to distinct on-disk filenames."""
    encoded = {
        safe_conversation_id_part(raw)
        for raw in ("tg:456", "tg/456", "tg-456", "tg.456")
    }
    assert len(encoded) == 4


def test_safe_conversation_id_part_roundtrip_for_simple_ids() -> None:
    # IDs already in the safe set must pass through unchanged.
    assert safe_conversation_id_part("conv-abc") == "conv-abc"
    assert safe_conversation_id_part("u_1.session-2") == "u_1.session-2"


# ---------- truncate_around_terms ----------


def test_truncate_around_terms_returns_input_when_under_budget() -> None:
    text = "short transcript"
    assert truncate_around_terms(text, ["deploy"], max_chars=1000) == text


def test_truncate_around_terms_centers_on_match_position() -> None:
    head = "alpha " * 200  # 1200 chars
    tail = "zeta " * 200  # 1000 chars
    body = head + "deployment marker " + tail
    result = truncate_around_terms(body, ["deployment"], max_chars=400)
    assert len(result) <= 400 + len("…[truncated]\n") + len("\n…[truncated]")
    assert "deployment marker" in result


def test_truncate_around_terms_falls_back_to_head_when_no_match() -> None:
    body = "x" * 10_000
    result = truncate_around_terms(body, ["needle"], max_chars=200)
    assert result.startswith("x" * 200)
    assert "[truncated]" in result


# ---------- Stale-generation guard in MemoryReviewService ----------


@pytest.mark.asyncio
async def test_review_ignores_memory_completion_for_stale_generation(
    tmp_path: Path,
) -> None:
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    coordinator = ConversationTurnCoordinator()
    # Open conversation, request gen=1 then gen=2; gen=1 is now stale.
    await coordinator.request_generation("conv-1")
    await coordinator.request_generation("conv-1")
    await projection.append_user_message(
        user_id="alex", conversation_id="conv-1", text="hi"
    )
    await projection.append_assistant_message(
        user_id="alex", conversation_id="conv-1", generation=2, text="hello"
    )

    executor = ToolCallExecutor(
        runtime=runtime,
        memory_service=MemoryService(runtime=runtime),
    )
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)

    llm = FakeLlmClient([AssistantText(text="Nothing to save.")])
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        tool_results=tool_results,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=1,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    # A stale memory completion arrives — must NOT reset counter or
    # consume the next assistant-text increment.
    await bus.publish(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,  # stale; current is 2
            call_id="t-stale",
            tool_name="memory",
            input=MemoryToolInput(
                action="add", target="user", content="stale"
            ),
            result=RuntimeToolResult(stdout="{}"),
        )
    )
    # Now a real current-gen assistant text fires — counter should
    # increment from 0 to 1, threshold hits, review runs.
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=2, text="reply"
        )
    )
    await service.wait_until_idle()
    reviews = [
        event
        for event in await store.list_events()
        if isinstance(event, MemoryReviewCompleted)
    ]
    assert len(reviews) == 1
    assert reviews[0].note == "Nothing to save."


def _collect(events: list[EventBase], cls: type) -> list:  # type: ignore[type-arg]
    return [event for event in events if isinstance(event, cls)]
