import asyncio
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
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
    LlmToolCall,
)
from harness_agent.memory_review import MemoryReviewService
from harness_agent.memory_service import MemoryService
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import RuntimeToolResult
from harness_agent.runtime.fake import FakeUserRuntime
from harness_agent.store import SQLiteEventStore
from harness_agent.session_search_service import SessionSearchService
from harness_agent.tool_executor import ToolCallExecutor
from harness_agent.tools import MemoryToolInput, default_tool_registry
from harness_agent.turns import ConversationTurnCoordinator


def tool_executor_for_test(
    *,
    runtime,
    memory_service=None,
    session_search=None,
    session_search_llm=None,
    **kwargs,
):
    return ToolCallExecutor(
        runtime=runtime,
        memory_service=memory_service or MemoryService(runtime=runtime),
        session_search=session_search
        or SessionSearchService(
            runtime=runtime,
            llm=session_search_llm or FakeLlmClient([]),
        ),
        **kwargs,
    )


async def _coord_at(conversation_id: str, generation: int) -> ConversationTurnCoordinator:
    coord = ConversationTurnCoordinator()
    for _ in range(generation):
        await coord.request_generation(conversation_id)
    return coord


async def _advance(coord: ConversationTurnCoordinator, conversation_id: str) -> int:
    return await coord.request_generation(conversation_id)


async def _seed_conversation(
    projection: SQLiteConversationProjection,
    conversation_id: str,
) -> None:
    await projection.append_user_message(
        user_id="alex",
        conversation_id=conversation_id,
        text="My name is Alex and I always want concise answers.",
    )
    await projection.append_assistant_message(
        user_id="alex",
        conversation_id=conversation_id,
        generation=1,
        text="Noted.",
    )


def _collect_review_events(events: list[EventBase]) -> list[MemoryReviewCompleted]:
    return [event for event in events if isinstance(event, MemoryReviewCompleted)]


def _wire_executor(bus: EventBus, runtime: FakeUserRuntime) -> ToolCallExecutor:
    executor = tool_executor_for_test(
        runtime=runtime,
        memory_service=MemoryService(runtime=runtime),
    )
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)
    return executor


@pytest.mark.asyncio
async def test_review_fires_after_threshold_and_persists_memory(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="t-1",
                name="memory",
                input=MemoryToolInput(
                    action="add",
                    target="user",
                    content="User prefers concise answers.",
                ),
            ),
            AssistantText(text="Saved."),
        ]
    )
    coordinator = ConversationTurnCoordinator()
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=2,
        max_iterations=3,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    await _advance(coordinator, "conv-1")  # gen 1
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="step1"
        )
    )
    await _advance(coordinator, "conv-1")  # gen 2
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=2, text="step2"
        )
    )
    await service.wait_until_idle()

    raw = await runtime.read_memory_file("alex", "user")
    assert "User prefers concise answers." in raw
    review_events = _collect_review_events(await store.list_events())
    assert len(review_events) == 1
    assert review_events[0].actions == ["add user"]


@pytest.mark.asyncio
async def test_review_records_nothing_to_save_note(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    llm = FakeLlmClient([AssistantText(text="Nothing to save.")])
    coordinator = await _coord_at("conv-1", 1)
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=1,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)

    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="ok"
        )
    )
    await service.wait_until_idle()

    review_events = _collect_review_events(await store.list_events())
    assert len(review_events) == 1
    assert review_events[0].actions == []
    assert review_events[0].note == "Nothing to save."
    raw = await runtime.read_memory_file("alex", "user")
    assert raw == ""


@pytest.mark.asyncio
async def test_review_counter_resets_on_memory_tool_call(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    llm = FakeLlmClient([])  # would raise if review fires
    coordinator = ConversationTurnCoordinator()
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=2,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    await _advance(coordinator, "conv-1")  # gen 1
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="step1"
        )
    )
    await bus.publish(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            call_id="t-1",
            tool_name="memory",
            input=MemoryToolInput(
                action="add", target="user", content="User is Alex."
            ),
            result=RuntimeToolResult(stdout="{}"),
        )
    )
    await _advance(coordinator, "conv-1")  # gen 2
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=2, text="step2"
        )
    )
    await service.wait_until_idle()
    assert _collect_review_events(await store.list_events()) == []


@pytest.mark.asyncio
async def test_counters_are_per_conversation(tmp_path: Path) -> None:
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-A")
    await _seed_conversation(projection, "conv-B")

    llm = FakeLlmClient([])
    coordinator = ConversationTurnCoordinator()
    await coordinator.request_generation("conv-A")
    await coordinator.request_generation("conv-B")
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=2,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)

    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-A", generation=1, text="step"
        )
    )
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-B", generation=1, text="step"
        )
    )
    await service.wait_until_idle()
    assert _collect_review_events(await store.list_events()) == []


@pytest.mark.asyncio
async def test_foreground_memory_tool_call_suppresses_next_assistant_increment(
    tmp_path: Path,
) -> None:
    """Regression: when the foreground turn called the memory tool, the
    assistant's final text of the SAME generation must not re-increment
    the counter and immediately re-fire a review. Otherwise nudge_interval=1
    would mean "review every turn that wrote memory", which negates the
    purpose of the foreground-write counter reset."""
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    llm = FakeLlmClient([])  # would raise if review fires
    coordinator = ConversationTurnCoordinator()
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=1,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    # Foreground turn: model called memory.add, then emitted assistant text.
    await _advance(coordinator, "conv-1")  # gen 1
    await bus.publish(
        ToolCallCompleted(
            user_id="alex",
            conversation_id="conv-1",
            generation=1,
            call_id="t-fg",
            tool_name="memory",
            input=MemoryToolInput(
                action="add", target="user", content="User is Alex."
            ),
            result=RuntimeToolResult(stdout="{}"),
        )
    )
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="Saved."
        )
    )
    await service.wait_until_idle()
    assert _collect_review_events(await store.list_events()) == []
    # The NEXT generation's assistant text should resume counting and fire
    # the threshold normally.
    review_llm = FakeLlmClient([AssistantText(text="Nothing to save.")])
    service._llm = review_llm  # swap in a real LLM so the review can run  # type: ignore[attr-defined]
    await _advance(coordinator, "conv-1")  # gen 2
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=2, text="next turn"
        )
    )
    await service.wait_until_idle()
    review_events = _collect_review_events(await store.list_events())
    assert len(review_events) == 1


@pytest.mark.asyncio
async def test_in_flight_review_does_not_clobber_newer_foreground_increment(
    tmp_path: Path,
) -> None:
    """Race: a foreground turn advances the counter while a review is in
    flight. When the review finishes its (review-scope) memory write,
    its ToolCallCompleted must NOT reset the counter — otherwise the
    newer increment is silently dropped and the next foreground review
    is delayed by a full interval."""
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    gate = asyncio.Event()
    saved_input = MemoryToolInput(
        action="add", target="user", content="User likes pytest."
    )

    class GatedReviewLlm:
        def __init__(self) -> None:
            self.calls = 0

        async def respond(self, request):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                # First call inside the review fork: pause until the
                # test has nudged the conversation forward.
                await gate.wait()
                return LlmToolCall(
                    call_id="t-review",
                    name="memory",
                    input=saved_input,
                )
            return AssistantText(text="Done.")

    llm = GatedReviewLlm()
    coordinator = ConversationTurnCoordinator()
    service = MemoryReviewService(
        bus=bus,
        llm=llm,  # type: ignore[arg-type]
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=2,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    # gen=1, gen=2 → threshold (2) hit, review #1 spawned. Counter resets to 0.
    await _advance(coordinator, "conv-1")
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="a"
        )
    )
    await _advance(coordinator, "conv-1")
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=2, text="b"
        )
    )
    # Give the spawned review one event-loop tick to start awaiting the gate.
    await asyncio.sleep(0)
    # gen=3 fires while review #1 is still gated → counter advances to 1.
    await _advance(coordinator, "conv-1")
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=3, text="c"
        )
    )
    # Release the review. It will publish ToolCallRequested(gen=0) for the
    # memory write and then return assistant text.
    gate.set()
    await service.wait_until_idle()
    # gen=4 fires; with the bug fixed the counter is at 1, this turns it
    # into 2 ≥ threshold → review #2 spawns. With the old (buggy) code
    # review #1's ToolCallCompleted would have reset the counter to 0 and
    # this gen=4 would only push it to 1, no second review.
    await _advance(coordinator, "conv-1")
    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=4, text="d"
        )
    )
    await service.wait_until_idle()

    reviews = [
        event
        for event in await store.list_events()
        if isinstance(event, MemoryReviewCompleted)
    ]
    assert len(reviews) == 2


@pytest.mark.asyncio
async def test_review_memory_mutations_are_emitted_on_event_bus(
    tmp_path: Path,
) -> None:
    """Review-scoped memory writes go through ToolCallRequested /
    ToolCallCompleted, so the exact mutation input and result are
    auditable on the event bus."""
    runtime = FakeUserRuntime()
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    _wire_executor(bus, runtime)
    await _seed_conversation(projection, "conv-1")

    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="t-1",
                name="memory",
                input=MemoryToolInput(
                    action="add",
                    target="memory",
                    content="Project uses uv for env management.",
                ),
            ),
            AssistantText(text="Saved."),
        ]
    )
    coordinator = await _coord_at("conv-1", 1)
    service = MemoryReviewService(
        bus=bus,
        llm=llm,
        projection=projection,
        tool_registry=default_tool_registry(),
        turn_coordinator=coordinator,
        nudge_interval=1,
    )
    bus.subscribe(AssistantTextProduced, service.handle_assistant_text)
    bus.subscribe(ToolCallCompleted, service.handle_tool_call_completed)

    await bus.publish(
        AssistantTextProduced(
            user_id="alex", conversation_id="conv-1", generation=1, text="step"
        )
    )
    await service.wait_until_idle()

    events = await store.list_events()
    memory_requested = [
        event
        for event in events
        if isinstance(event, ToolCallRequested) and event.tool_name == "memory"
    ]
    memory_completed = [
        event
        for event in events
        if isinstance(event, ToolCallCompleted) and event.tool_name == "memory"
    ]
    assert len(memory_requested) == 1
    assert len(memory_completed) == 1
    assert memory_requested[0].generation == 0  # review-scope marker
    assert memory_completed[0].generation == 0
    assert isinstance(memory_requested[0].input, MemoryToolInput)
    assert memory_requested[0].input.content == "Project uses uv for env management."
