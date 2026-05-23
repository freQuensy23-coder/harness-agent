from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness_agent import events
from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder, system_prompt_with_tools
from harness_agent.events import (
    AgentTurnRequested,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.memory_service import MemoryService
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall, ToolResultMessage
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SQLiteSubAgentStore, SubAgentService
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.session_search_service import SessionSearchService
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    AgentRunInput,
    FileEditInput,
    FileListInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    ScheduleOnceInput,
    TaskCreateInput,
    ToolSpec,
    WebFetchInput,
    default_tool_registry,
    parse_llm_tool_input,
)
from harness_agent.web_fetch import HttpxWebFetcher

import httpx


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


@pytest.mark.asyncio
async def test_file_write_read_edit_and_shell_exec_are_routed(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        files={"/workspace/agent/SOUL.md": "S", "/workspace/agent/AGENTS.md": "A", "/workspace/agent/USER.md": "U", "/workspace/agent/TOOLS.md": "T"},
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
        shell_results=[RuntimeToolResult(stdout="hi\n")],
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="write",
                name="file.write",
                input=FileWriteInput(path="/workspace/hello.py", content="print('bye')\n"),
            ),
            LlmToolCall(
                call_id="edit",
                name="file.edit",
                input=FileEditInput(path="/workspace/hello.py", old="bye", new="hi"),
            ),
            LlmToolCall(
                call_id="read",
                name="file.read",
                input=FileReadInput(path="/workspace/hello.py"),
            ),
            LlmToolCall(
                call_id="run",
                name="shell.exec",
                input=ShellExecInput(command="python hello.py"),
            ),
            AssistantText(text="done"),
        ]
    )
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=runtime, task_store=task_store)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection, turn_coordinator=coordinator).handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, agent_turn_handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(
        AgentTurnRequested,
        agent_turn_handler.handle_agent_turn,
    )

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:1",
            source="cli",
            text="write edit read run",
        )
    )

    assert runtime.file_write_calls[0].path == "/workspace/hello.py"
    assert runtime.shell_exec_calls == [ShellExecInput(command="python hello.py")]
    completed = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ]
    assert [event.tool_name for event in completed] == [
        "file.write",
        "file.edit",
        "file.read",
        "shell.exec",
    ]


@pytest.mark.asyncio
async def test_agent_turn_publishes_generation_started_before_llm_request(tmp_path: Path) -> None:
    assert hasattr(events, "AgentGenerationStarted")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient([AssistantText(text="done")])
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:1",
            source="cli",
            text="hello",
        )
    )

    event_types = [event.type for event in await store.list_events()]
    assert event_types == [
        "user.text.received",
        "agent.turn.requested",
        "agent.generation.started",
        "assistant.text.produced",
    ]


@pytest.mark.asyncio
async def test_agent_turn_system_prompt_matches_executor_registry_disabled_groups(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient([AssistantText(text="done")])
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=runtime)
    handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=tool_executor.tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    bus.subscribe(
        UserTextReceived,
        ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text,
    )
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:disabled-tools",
            source="cli",
            text="hello",
        )
    )

    request = llm.requests[0]
    tool_names = {tool.name for tool in request.tools}
    assert "web.fetch" not in tool_names
    assert not any(name.startswith("task.") for name in tool_names)
    assert not any(name.startswith("schedule.") for name in tool_names)
    assert not any(name.startswith("agent.") for name in tool_names)
    assert "memory" in tool_names
    assert "session.search" in tool_names
    assert "- web.fetch fetches HTTP/HTTPS text." not in request.system
    assert "- task.* manages the conversation checklist." not in request.system
    assert "schedule.once" not in request.system
    assert "- agent.* runs sub-agents" not in request.system
    assert "- memory writes durable notes" in request.system
    assert "- session.search recalls focused summaries" in request.system
    assert "Persistent memory:" in request.system
    assert "- shell.exec runs commands in /workspace." in request.system
    assert "- skill.* reads enabled markdown skills." in request.system


@pytest.mark.asyncio
async def test_task_tool_is_persisted(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="task",
                name="task.create",
                input=TaskCreateInput(title="ship harness"),
            ),
            AssistantText(text="tracked"),
        ]
    )
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=runtime, task_store=task_store)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection, turn_coordinator=coordinator).handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, agent_turn_handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(
        AgentTurnRequested,
        agent_turn_handler.handle_agent_turn,
    )

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:1",
            source="cli",
            text="track this",
        )
    )

    tasks = await task_store.list(
        user_id="u:1",
        conversation_id="cli:1",
        include_stopped=True,
    )
    assert [(task.title, task.status) for task in tasks] == [("ship harness", "pending")]


@pytest.mark.asyncio
async def test_tool_exception_completes_with_error_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=FakeUserRuntime())
    requested = ToolCallRequested(
        user_id="u:1",
        conversation_id="cli:errors",
        generation=1,
        call_id="agent-run",
        tool_name="agent.run",
        input=AgentRunInput(prompt="work"),
    )

    tool_results.expect(requested)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)

    await bus.publish(requested)

    completed = await tool_results.wait(requested)
    assert completed.result.exit_code == 1
    assert completed.result.stderr == "Unsupported tool call: agent.run"
    assert completed.result.stdout == ""
    assert [event.type for event in await store.list_events()] == [
        "tool.call.requested",
        "tool.call.error",
        "tool.call.completed",
    ]


@pytest.mark.asyncio
async def test_oversized_tool_output_is_saved_to_workspace_content() -> None:
    runtime = FakeUserRuntime(shell_results=[RuntimeToolResult(stdout="x" * 30_001)])
    executor = tool_executor_for_test(runtime=runtime, max_model_output_chars=20_000)

    (completed,) = await executor.handle_tool_call_requested(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:large",
            generation=1,
            call_id="long-output",
            tool_name="shell.exec",
            input=ShellExecInput(command="yes"),
        )
    )

    assert completed.result.exit_code == 0
    assert completed.result.stderr == ""
    assert (
        "truncated, because it is too long. Full result saved to "
        "/workspace/content/tool-results/cli%3Alarge-1-long-output.txt"
    ) in completed.result.stdout
    assert "You can read it in chunks with file.read" in completed.result.stdout
    assert len(runtime.content_write_calls) == 1
    path, content = runtime.content_write_calls[0]
    assert path == "/workspace/content/tool-results/cli%3Alarge-1-long-output.txt"
    saved = content.decode("utf-8")
    assert saved.startswith("shell.exec stdout:\n")
    assert "x" * 30_001 in saved


@pytest.mark.asyncio
async def test_oversized_tool_output_paths_are_injective_for_unsafe_ids() -> None:
    """Before the fix, the lossy `re.sub("[^A-Za-z0-9._-]+", "-")` encoder
    collapsed `:`, `/`, ` ` all to `-`, so distinct conversation_ids
    like `tg:big`, `tg/big`, `tg-big` wrote their oversized results to
    the same spill file. The second write silently overwrote the first
    and could expose another conversation's saved output."""
    runtime = FakeUserRuntime(
        shell_results=[
            RuntimeToolResult(stdout="x" * 30_001),
            RuntimeToolResult(stdout="y" * 30_001),
            RuntimeToolResult(stdout="z" * 30_001),
        ]
    )
    executor = tool_executor_for_test(runtime=runtime, max_model_output_chars=20_000)

    for cid in ("tg:big", "tg/big", "tg-big"):
        await executor.handle_tool_call_requested(
            ToolCallRequested(
                user_id="u:1",
                conversation_id=cid,
                generation=1,
                call_id="c",
                tool_name="shell.exec",
                input=ShellExecInput(command="yes"),
            )
        )

    paths = {path for path, _ in runtime.content_write_calls}
    assert len(paths) == 3


def test_file_listing_tools_do_not_advertise_hard_result_caps() -> None:
    assert "max_results" not in default_tool_registry().by_name("file.list").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.glob").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.grep").parameters_schema()["properties"]
    assert "max_results" not in FileListInput.model_fields


def test_schedule_once_input_accepts_legacy_immediate_and_single_time_selector() -> None:
    assert parse_llm_tool_input("schedule.once", {"message": "now"}) == ScheduleOnceInput(
        message="now"
    )
    assert parse_llm_tool_input(
        "schedule.once",
        {"message": "later", "delay_seconds": 60},
    ) == ScheduleOnceInput(message="later", delay_seconds=60)
    assert parse_llm_tool_input(
        "schedule.once",
        {"message": "at time", "run_at_utc": "2026-05-22T10:00:00+00:00"},
    ) == ScheduleOnceInput(
        message="at time",
        run_at_utc="2026-05-22T10:00:00+00:00",
    )
    with pytest.raises(ValueError, match="provide at most one"):
        parse_llm_tool_input(
            "schedule.once",
            {
                "message": "ambiguous",
                "run_at_utc": "2026-05-22T10:00:00+00:00",
                "delay_seconds": 60,
            },
        )


def test_default_tool_registry_can_disable_optional_tool_groups() -> None:
    registry = default_tool_registry(
        include_web_fetch=False,
        include_tasks=False,
        include_schedules=False,
        include_agents=False,
    )

    tool_names = {tool.name for tool in registry.tools}
    assert "web.fetch" not in tool_names
    assert not any(name.startswith("task.") for name in tool_names)
    assert not any(name.startswith("schedule.") for name in tool_names)
    assert not any(name.startswith("agent.") for name in tool_names)
    assert {"shell.exec", "file.read", "skill.list"}.issubset(tool_names)


def test_tool_executor_registry_matches_configured_dependencies(tmp_path: Path) -> None:
    minimal = tool_executor_for_test(runtime=FakeUserRuntime()).tool_registry()
    minimal_names = {tool.name for tool in minimal.tools}
    assert "web.fetch" not in minimal_names
    assert not any(name.startswith("task.") for name in minimal_names)
    assert not any(name.startswith("schedule.") for name in minimal_names)
    assert not any(name.startswith("agent.") for name in minimal_names)
    assert "memory" in minimal_names
    assert "session.search" in minimal_names

    sub_agents = SubAgentService(
        bus=EventBus(SQLiteEventStore(tmp_path / "subagent-events.sqlite3")),
        store=SQLiteSubAgentStore(tmp_path / "subagents.sqlite3"),
    )
    configured = tool_executor_for_test(
        runtime=FakeUserRuntime(),
        task_store=SQLiteTaskStore(tmp_path / "tasks.sqlite3"),
        schedule_store=SQLiteScheduleStore(
            tmp_path / "schedules.sqlite3",
            now=lambda: datetime(2026, 5, 22, 10, 0, tzinfo=UTC),
        ),
        web_fetcher=HttpxWebFetcher(llm=FakeLlmClient([])),
        sub_agents=sub_agents,
    ).tool_registry()

    configured_names = {tool.name for tool in configured.tools}
    assert "web.fetch" in configured_names
    assert {"task.create", "task.update"}.issubset(configured_names)
    assert {"schedule.once", "schedule.cancel"}.issubset(configured_names)
    assert {"agent.run", "agent.cancel"}.issubset(configured_names)


def test_system_prompt_with_tools_mentions_configured_mcp_tools() -> None:
    prompt = system_prompt_with_tools(
        "base",
        [
            ToolSpec(
                name="mcp.local.echo",
                description="Echo text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ],
    )

    assert "- mcp.* calls configured MCP tools." in prompt


@pytest.mark.asyncio
async def test_web_fetch_converts_html_to_markdown_and_filters_with_secondary_model() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html>
              <head><title>Ignored chrome</title><script>alert('x')</script></head>
              <body>
                <nav>Navigation chrome</nav>
                <article>
                  <h1>Important Title</h1>
                  <p>Use the <a href="/docs">docs</a> first.</p>
                </article>
              </body>
            </html>
            """,
        )
    )
    llm = FakeLlmClient([AssistantText(text="Use the docs first.")])

    result = await HttpxWebFetcher(llm=llm, transport=transport).fetch(
        WebFetchInput(
            url="https://example.test/article",
            prompt="Find the main instruction.",
        )
    )

    assert result == RuntimeToolResult(stdout="Use the docs first.")
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request.tools == []
    assert request.messages[0].text.startswith("URL: https://example.test/article")
    assert "Question: Find the main instruction." in request.messages[0].text
    assert "# Important Title" in request.messages[0].text
    assert "[docs](https://example.test/docs)" in request.messages[0].text
    assert "<script>" not in request.messages[0].text
    assert "<h1>" not in request.messages[0].text
    assert "Ignored chrome" not in request.messages[0].text
    assert "Navigation chrome" not in request.messages[0].text


@pytest.mark.asyncio
async def test_web_fetch_result_does_not_expose_raw_markdown_to_main_history(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="fetch",
                name="web.fetch",
                input=WebFetchInput(
                    url="https://example.test/article",
                    prompt="Extract only the key sentence.",
                ),
            ),
            AssistantText(text="done"),
        ]
    )
    web_llm = FakeLlmClient([AssistantText(text="Filtered key sentence.")])
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<main><h1>Raw Page Title</h1><p>Filtered key sentence.</p></main>",
        )
    )
    runtime = FakeUserRuntime(agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"))
    handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    tool_executor = tool_executor_for_test(
        runtime=runtime,
        web_fetcher=HttpxWebFetcher(llm=web_llm, transport=transport),
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection, turn_coordinator=coordinator).handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:web-fetch",
            source="cli",
            text="read https://example.test/article",
        )
    )

    history = await projection.list_llm_messages("cli:web-fetch")
    tool_results_in_history = [
        message for message in history if isinstance(message, ToolResultMessage)
    ]
    assert len(tool_results_in_history) == 1
    assert "Filtered key sentence." in tool_results_in_history[0].content
    assert "Raw Page Title" not in tool_results_in_history[0].content
