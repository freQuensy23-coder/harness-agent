import asyncio
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    AssistantTextProduced,
    SubAgentCancelled,
    SubAgentCompleted,
    SubAgentFailed,
    SubAgentStarted,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.llm import (
    AssistantMessage,
    AssistantText,
    AssistantToolCallMessage,
    FakeLlmClient,
    LlmRequest,
    LlmToolCall,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentResultWaiter, SubAgentService
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    AgentResultInput,
    AgentRunInput,
    AgentSpawnInput,
    FileWriteInput,
    default_tool_registry,
)


class NeverReturningLlmClient(FakeLlmClient):
    def __init__(self) -> None:
        super().__init__([])

    async def respond(self, request: LlmRequest) -> AssistantText:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_agent_run_subagent_can_write_file_and_return_result(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="delegate",
                name="agent.run",
                input=AgentRunInput(
                    prompt="write /workspace/subagent.txt with delegated",
                    name="writer",
                ),
            ),
            LlmToolCall(
                call_id="child-write",
                name="file.write",
                input=FileWriteInput(
                    path="/workspace/subagent.txt",
                    content="delegated",
                ),
            ),
            AssistantText(text="child wrote delegated"),
            AssistantText(text="parent saw child result"),
        ]
    )
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    sub_agent_results = SubAgentResultWaiter()
    sub_agents = SubAgentService(
        bus=bus,
        store=sub_agent_store,
        result_waiter=sub_agent_results,
    )
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        sub_agents=sub_agents,
    )
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=tool_results,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(AssistantTextProduced, sub_agent_results.handle_assistant_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:parent",
            source="cli",
            text="delegate file write",
        )
    )

    assert runtime.file_write_calls == [
        FileWriteInput(path="/workspace/subagent.txt", content="delegated")
    ]
    events = await store.list_events()
    assert [event.type for event in events if event.type.startswith("subagent.")] == [
        "subagent.started",
        "subagent.completed",
    ]
    started = [event for event in events if event.type == "subagent.started"][0]
    completed = [event for event in events if event.type == "subagent.completed"][0]
    assert started == SubAgentStarted(
        agent_id=started.agent_id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=started.child_conversation_id,
        parent_call_id="delegate",
        name="writer",
        id=started.id,
        occurred_at=started.occurred_at,
    )
    assert completed == SubAgentCompleted(
        agent_id=started.agent_id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=started.child_conversation_id,
        result="child wrote delegated",
        id=completed.id,
        occurred_at=completed.occurred_at,
    )

    parent_messages = await projection.list_llm_messages("cli:parent")
    assert parent_messages[0] == UserMessage(text="delegate file write")
    assert parent_messages[-1] == AssistantMessage(text="parent saw child result")
    parent_tool_result = [
        message
        for message in parent_messages
        if message.kind == "tool_result" and message.name == "agent.run"
    ][0]
    assert '"status": "completed"' in parent_tool_result.content
    assert "child wrote delegated" in parent_tool_result.content

    child_messages = await projection.list_llm_messages(started.child_conversation_id)
    assert child_messages == [
        UserMessage(text="write /workspace/subagent.txt with delegated"),
        AssistantToolCallMessage(
            call_id="child-write",
            name="file.write",
            arguments={"path": "/workspace/subagent.txt", "content": "delegated"},
        ),
        ToolResultMessage(
            call_id="child-write",
            name="file.write",
            content="file.write stdout:\nstderr:\n\nexit_code: 0",
        ),
        AssistantMessage(text="child wrote delegated"),
    ]


@pytest.mark.asyncio
async def test_agent_result_returns_unknown_agent_as_tool_error(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="missing",
                name="agent.result",
                input=AgentResultInput(agent_id="missing-agent"),
            ),
            AssistantText(text="reported"),
        ]
    )
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    sub_agent_results = SubAgentResultWaiter()
    sub_agents = SubAgentService(
        bus=bus,
        store=sub_agent_store,
        result_waiter=sub_agent_results,
    )
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        sub_agents=sub_agents,
    )
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=tool_results,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection).handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:parent",
            source="cli",
            text="check missing agent",
        )
    )

    completed = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ][0]
    assert completed.result.exit_code == 1
    assert completed.result.stderr == "Unknown sub-agent: missing-agent\n"


@pytest.mark.asyncio
async def test_agent_spawn_tool_input_round_trips_through_event_store(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    spawn_input = AgentSpawnInput(prompt="write report", name="writer", timeout_seconds=12.5)
    requested = ToolCallRequested(
        user_id="u:1",
        conversation_id="cli:parent",
        generation=1,
        call_id="spawn",
        tool_name="agent.spawn",
        input=spawn_input,
    )
    completed = ToolCallCompleted(
        user_id="u:1",
        conversation_id="cli:parent",
        generation=1,
        call_id="spawn",
        tool_name="agent.spawn",
        input=spawn_input,
        result=RuntimeToolResult(stdout="spawned"),
    )

    await store.append(requested)
    await store.append(completed)

    stored_requested, stored_completed = await store.list_events()
    assert stored_requested == requested
    assert stored_completed == completed


@pytest.mark.asyncio
async def test_spawn_list_and_cancel_subagent(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    sub_agents = SubAgentService(
        bus=bus,
        store=sub_agent_store,
        result_waiter=SubAgentResultWaiter(),
    )

    record = await sub_agents.spawn(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        parent_call_id="spawn",
        input=AgentSpawnInput(
            prompt="keep working",
            name="worker",
            timeout_seconds=60,
        ),
    )
    await wait_for_event(store, "subagent.started")

    running = await sub_agents.list_for_parent(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        include_completed=False,
    )
    assert [item.id for item in running] == [record.id]

    cancelled = await sub_agents.cancel(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
    )

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert await sub_agent_store.get(record.id) == cancelled
    assert await sub_agents.list_for_parent(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        include_completed=False,
    ) == []
    assert await sub_agents.list_for_parent(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        include_completed=True,
    ) == [cancelled]

    events = await store.list_events()
    assert [event.type for event in events if event.type.startswith("subagent.")] == [
        "subagent.started",
        "subagent.cancelled",
    ]
    started = [event for event in events if event.type == "subagent.started"][0]
    cancelled_event = [event for event in events if event.type == "subagent.cancelled"][0]
    assert started == SubAgentStarted(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=record.child_conversation_id,
        parent_call_id="spawn",
        name="worker",
        id=started.id,
        occurred_at=started.occurred_at,
    )
    assert cancelled_event == SubAgentCancelled(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=record.child_conversation_id,
        id=cancelled_event.id,
        occurred_at=cancelled_event.occurred_at,
    )


@pytest.mark.asyncio
async def test_subagent_timeout_is_failed_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    bus = EventBus(store)
    hanging_llm = NeverReturningLlmClient()
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(
            runtime=FakeUserRuntime(
                agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
            )
        ),
        llm=hanging_llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=ToolCallResultWaiter(),
    )
    sub_agent_results = SubAgentResultWaiter()
    sub_agents = SubAgentService(
        bus=bus,
        store=SQLiteSubAgentStore(tmp_path / "subagents.sqlite3"),
        result_waiter=sub_agent_results,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(AssistantTextProduced, sub_agent_results.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    record = await sub_agents.run(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        parent_call_id="run",
        input=AgentRunInput(
            prompt="never answered",
            name="worker",
            timeout_seconds=0.01,
        ),
    )

    assert record.status == "failed"
    assert record.error
    assert hanging_llm.requests[0].conversation_id == record.child_conversation_id
    events = await store.list_events()
    assert [event.type for event in events] == [
        "subagent.started",
        "user.text.received",
        "agent.turn.requested",
        "subagent.failed",
    ]
    failed = [event for event in events if event.type == "subagent.failed"][0]
    assert failed == SubAgentFailed(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=record.child_conversation_id,
        error=record.error,
        id=failed.id,
        occurred_at=failed.occurred_at,
    )


@pytest.mark.asyncio
async def test_subagent_result_and_cancel_are_parent_scoped(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    sub_agents = SubAgentService(
        bus=bus,
        store=sub_agent_store,
        result_waiter=SubAgentResultWaiter(),
    )

    record = await sub_agents.spawn(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        parent_call_id="spawn",
        input=AgentSpawnInput(
            prompt="keep working",
            name="worker",
            timeout_seconds=60,
        ),
    )

    assert await sub_agents.result(
        agent_id=record.id,
        user_id="u:2",
        parent_conversation_id="cli:parent",
    ) is None
    assert await sub_agents.result(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:other",
    ) is None
    assert await sub_agents.cancel(
        agent_id=record.id,
        user_id="u:2",
        parent_conversation_id="cli:parent",
    ) is None
    assert await sub_agent_store.get(record.id) == record

    cancelled = await sub_agents.cancel(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
    )

    assert cancelled is not None
    assert cancelled.status == "cancelled"


async def wait_for_event(store: SQLiteEventStore, event_type: str) -> None:
    for _ in range(100):
        if event_type in [event.type for event in await store.list_events()]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"missing event: {event_type}")
