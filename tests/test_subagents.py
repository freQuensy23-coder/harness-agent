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
    SubAgentRequested,
    SubAgentStarted,
    SubAgentTimedOut,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import (
    AgentTurnHandler,
    ConversationProjector,
    SUB_AGENT_SYSTEM_PREFIX,
    SUB_AGENT_TASK_PREFIX,
)
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
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentService
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    AgentResultInput,
    AgentRunInput,
    AgentSpawnInput,
    FileWriteInput,
    default_tool_registry,
)


def _wire_sub_agents(bus: EventBus, sub_agents: SubAgentService) -> None:
    bus.subscribe(SubAgentRequested, sub_agents.handle_requested)
    bus.subscribe(SubAgentStarted, sub_agents.handle_started)
    bus.subscribe(SubAgentTimedOut, sub_agents.handle_timed_out)
    bus.subscribe(AssistantTextProduced, sub_agents.handle_assistant_text)
    bus.subscribe(SubAgentCompleted, sub_agents.handle_completed)
    bus.subscribe(SubAgentFailed, sub_agents.handle_failed)
    bus.subscribe(SubAgentCancelled, sub_agents.handle_cancelled)


class NeverReturningLlmClient(FakeLlmClient):
    def __init__(self) -> None:
        super().__init__([])

    async def respond(self, request: LlmRequest) -> AssistantText:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class RaisingChildLlmClient(FakeLlmClient):
    def __init__(self, *, parent_responses: list, child_error: str) -> None:
        super().__init__(list(parent_responses))
        self._child_error = child_error

    async def respond(self, request: LlmRequest) -> AssistantText:
        self.requests.append(request)
        if ":subagent:" in request.conversation_id:
            raise RuntimeError(self._child_error)
        if not self._responses:
            raise AssertionError("FakeLlmClient has no queued response")
        return self._responses.pop(0)


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
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
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
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    _wire_sub_agents(bus, sub_agents)
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
        "subagent.requested",
        "subagent.started",
        "subagent.completed",
    ]
    requested = [event for event in events if event.type == "subagent.requested"][0]
    started = [event for event in events if event.type == "subagent.started"][0]
    completed = [event for event in events if event.type == "subagent.completed"][0]
    assert requested.agent_id == started.agent_id == completed.agent_id
    assert requested.child_conversation_id == started.child_conversation_id
    assert started == SubAgentStarted(
        agent_id=started.agent_id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
        child_conversation_id=started.child_conversation_id,
        parent_call_id="delegate",
        name="writer",
        prompt="write /workspace/subagent.txt with delegated",
        timeout_seconds=AgentRunInput.model_fields["timeout_seconds"].default,
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
async def test_subagent_turn_filters_agent_tools_and_injects_system_prompt(
    tmp_path: Path,
) -> None:
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
                    prompt="finish the task and reply",
                    name="writer",
                ),
            ),
            AssistantText(text="child final answer"),
            AssistantText(text="parent answer"),
        ]
    )
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
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
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    _wire_sub_agents(bus, sub_agents)
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
            text="hello",
        )
    )

    parent_request = llm.requests[0]
    child_request = llm.requests[1]
    final_parent_request = llm.requests[-1]

    parent_tool_names = {tool.name for tool in parent_request.tools}
    child_tool_names = {tool.name for tool in child_request.tools}

    assert "agent.run" in parent_tool_names
    assert "agent.spawn" in parent_tool_names
    assert {name for name in child_tool_names if name.startswith("agent.")} == set()

    assert SUB_AGENT_SYSTEM_PREFIX.format(name="writer") in child_request.system
    assert SUB_AGENT_TASK_PREFIX in child_request.system
    assert "finish the task and reply" in child_request.system
    assert SUB_AGENT_SYSTEM_PREFIX.format(name="writer") not in parent_request.system
    assert SUB_AGENT_SYSTEM_PREFIX.format(name="writer") not in final_parent_request.system


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
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
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
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    _wire_sub_agents(bus, sub_agents)
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
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
    _wire_sub_agents(bus, sub_agents)

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
    subagent_events = [event.type for event in events if event.type.startswith("subagent.")]
    assert subagent_events == [
        "subagent.requested",
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
        prompt="keep working",
        timeout_seconds=60,
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
    sub_agents = SubAgentService(
        bus=bus,
        store=SQLiteSubAgentStore(tmp_path / "subagents.sqlite3"),
    )
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
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    _wire_sub_agents(bus, sub_agents)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    record = await sub_agents.run(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        parent_call_id="run",
        input=AgentRunInput(
            prompt="never answered",
            name="worker",
            timeout_seconds=0.5,
        ),
    )

    assert record.status == "failed"
    assert record.error
    assert hanging_llm.requests[0].conversation_id == record.child_conversation_id
    events = await store.list_events()
    assert [event.type for event in events] == [
        "subagent.requested",
        "subagent.started",
        "user.text.received",
        "agent.turn.requested",
        "agent.generation.started",
        "subagent.timed_out",
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
async def test_subagent_child_turn_error_publishes_failed_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = RaisingChildLlmClient(
        parent_responses=[
            LlmToolCall(
                call_id="delegate",
                name="agent.run",
                input=AgentRunInput(
                    prompt="do work",
                    name="writer",
                    timeout_seconds=60,
                ),
            ),
            AssistantText(text="parent saw failure"),
        ],
        child_error="upstream model exploded",
    )
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
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
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    _wire_sub_agents(bus, sub_agents)
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
            text="delegate",
        )
    )

    events = await store.list_events()
    subagent_events = [event.type for event in events if event.type.startswith("subagent.")]
    assert subagent_events == [
        "subagent.requested",
        "subagent.started",
        "subagent.failed",
    ]
    failed_event = [event for event in events if event.type == "subagent.failed"][0]
    assert "upstream model exploded" in failed_event.error
    stored = await sub_agent_store.get(failed_event.agent_id)
    assert stored is not None
    assert stored.status == "failed"
    assert "upstream model exploded" in (stored.error or "")
    assert failed_event.agent_id not in sub_agents._child_tasks
    assert failed_event.agent_id not in sub_agents._timeout_tasks
    parent_messages = await projection.list_llm_messages("cli:parent")
    assert parent_messages[-1] == AssistantMessage(text="parent saw failure")
    parent_tool_result = [
        message
        for message in parent_messages
        if message.kind == "tool_result" and message.name == "agent.run"
    ][0]
    assert '"status": "failed"' in parent_tool_result.content
    assert "upstream model exploded" in parent_tool_result.content


@pytest.mark.asyncio
async def test_subagent_result_and_cancel_are_parent_scoped(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
    _wire_sub_agents(bus, sub_agents)

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
    stored = await sub_agent_store.get(record.id)
    assert stored is not None
    assert stored.id == record.id
    assert stored.status == "running"

    cancelled = await sub_agents.cancel(
        agent_id=record.id,
        user_id="u:1",
        parent_conversation_id="cli:parent",
    )

    assert cancelled is not None
    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_spawn_returns_running_snapshot_even_if_child_completes_fast(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient([AssistantText(text="instant child reply")])
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=tool_results,
        sub_agent_lookup=sub_agents,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    _wire_sub_agents(bus, sub_agents)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    snapshot = await sub_agents.spawn(
        user_id="u:1",
        parent_conversation_id="cli:parent",
        parent_call_id="spawn",
        input=AgentSpawnInput(
            prompt="do it",
            name="worker",
            timeout_seconds=60,
        ),
    )

    assert snapshot.status == "running"
    assert snapshot.result is None

    stored = None
    for _ in range(200):
        stored = await sub_agent_store.get(snapshot.id)
        if stored is not None and stored.status != "running":
            break
        await asyncio.sleep(0.005)
    assert stored is not None
    assert stored.status == "completed"
    assert stored.result == "instant child reply"


@pytest.mark.asyncio
async def test_sub_agent_lookup_is_scoped_by_user_id(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    sub_agent_store = SQLiteSubAgentStore(tmp_path / "subagents.sqlite3")
    sub_agents = SubAgentService(bus=bus, store=sub_agent_store)
    _wire_sub_agents(bus, sub_agents)

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

    same_user = await sub_agents.get_by_child_conversation_id(
        user_id="u:1",
        conversation_id=record.child_conversation_id,
    )
    other_user = await sub_agents.get_by_child_conversation_id(
        user_id="u:2",
        conversation_id=record.child_conversation_id,
    )

    assert same_user is not None
    assert same_user.id == record.id
    assert other_user is None


@pytest.mark.asyncio
async def test_old_subagent_started_rows_load_without_new_fields(tmp_path: Path) -> None:
    import json

    import aiosqlite

    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    await store._ensure_schema()
    legacy_payload = {
        "id": "abc123",
        "type": "subagent.started",
        "occurred_at": "2025-01-01T00:00:00+00:00",
        "agent_id": "agent-1",
        "user_id": "u:1",
        "parent_conversation_id": "cli:parent",
        "child_conversation_id": "cli:parent:subagent:agent-1",
        "parent_call_id": "call-1",
        "name": "writer",
    }
    async with aiosqlite.connect(store._path) as db:
        await db.execute(
            "insert into events (id, type, occurred_at, payload) values (?, ?, ?, ?)",
            (
                "abc123",
                "subagent.started",
                "2025-01-01T00:00:00+00:00",
                json.dumps(legacy_payload),
            ),
        )
        await db.commit()

    events = await store.list_events()
    assert len(events) == 1
    started = events[0]
    assert started.type == "subagent.started"
    assert started.prompt == ""
    assert started.timeout_seconds == 0.0


async def wait_for_event(store: SQLiteEventStore, event_type: str) -> None:
    for _ in range(100):
        if event_type in [event.type for event in await store.list_events()]:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"missing event: {event_type}")
