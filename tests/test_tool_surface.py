import asyncio
from pathlib import Path

import pytest

from harness_agent import events
from harness_agent.subagents import NullSubAgentLookup, SubAgentService
from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall, ToolResultMessage
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.store import SQLiteEventStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    AgentRunInput,
    FileEditInput,
    FileListInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    TaskCreateInput,
    TaskGetInput,
    WebFetchInput,
    default_tool_registry,
)
from harness_agent.web_fetch import HttpxWebFetcher, WebFetchExtractionWaiter

import httpx


@pytest.mark.asyncio
async def test_file_write_read_edit_and_shell_exec_are_routed(tmp_path: Path, browser_use_service: BrowserUseService, web_fetch_waiter: WebFetchExtractionWaiter, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
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
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime, task_store=task_store, browser_use_service=browser_use_service, bus=bus, web_fetch_waiter=web_fetch_waiter, schedule_store=schedule_store, sub_agents=sub_agents)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=tool_results, sub_agent_lookup=NullSubAgentLookup())
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
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
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection, sub_agent_lookup=NullSubAgentLookup())
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
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
async def test_task_tool_is_persisted(tmp_path: Path, browser_use_service: BrowserUseService, web_fetch_waiter: WebFetchExtractionWaiter, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
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
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime, task_store=task_store, browser_use_service=browser_use_service, bus=bus, web_fetch_waiter=web_fetch_waiter, schedule_store=schedule_store, sub_agents=sub_agents)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_results=tool_results, sub_agent_lookup=NullSubAgentLookup())
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
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
async def test_tool_exception_completes_with_error_event(
    tmp_path: Path,
    browser_use_service: BrowserUseService,
    web_fetch_waiter: WebFetchExtractionWaiter,
    task_store: SQLiteTaskStore,
    schedule_store: SQLiteScheduleStore,
    sub_agents: SubAgentService,
) -> None:
    """Any tool handler raising must surface as a ToolCallError event
    *and* a ToolCallCompleted with exit_code=1, so the model sees the
    failure and the audit log records the original exception."""
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(
        runtime=FakeUserRuntime(),
        browser_use_service=browser_use_service,
        bus=bus,
        web_fetch_waiter=web_fetch_waiter,
        task_store=task_store,
        schedule_store=schedule_store,
        sub_agents=sub_agents,
    )
    # task.get against an id that doesn't exist raises KeyError inside
    # the handler. handle_tool_call_requested must catch it and emit
    # ToolCallError + ToolCallCompleted.
    requested = ToolCallRequested(
        user_id="u:1",
        conversation_id="cli:errors",
        generation=1,
        call_id="task-get",
        tool_name="task.get",
        input=TaskGetInput(task_id="does-not-exist"),
    )

    tool_results.expect(requested)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)

    await bus.publish(requested)

    completed = await tool_results.wait(requested)
    assert completed.result.exit_code == 1
    assert "does-not-exist" in completed.result.stderr
    assert completed.result.stdout == ""
    assert [event.type for event in await store.list_events()] == [
        "tool.call.requested",
        "tool.call.error",
        "tool.call.completed",
    ]


@pytest.mark.asyncio
async def test_oversized_tool_output_is_saved_to_workspace_content(
    tmp_path: Path, browser_use_service: BrowserUseService
, web_fetch_waiter: WebFetchExtractionWaiter, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
    runtime = FakeUserRuntime(shell_results=[RuntimeToolResult(stdout="x" * 30_001)])
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    executor = ToolCallExecutor(
        runtime=runtime,
        bus=bus,
        max_model_output_chars=20_000,
        browser_use_service=browser_use_service, web_fetch_waiter=web_fetch_waiter, task_store=task_store, schedule_store=schedule_store, sub_agents=sub_agents)

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
        "/workspace/content/tool-results/cli-large-1-long-output.txt"
    ) in completed.result.stdout
    assert "You can read it in chunks with file.read" in completed.result.stdout
    assert len(runtime.content_write_calls) == 1
    path, content = runtime.content_write_calls[0]
    assert path == "/workspace/content/tool-results/cli-large-1-long-output.txt"
    saved = content.decode("utf-8")
    assert saved.startswith("shell.exec stdout:\n")
    assert "x" * 30_001 in saved
    # Successful spill must be persisted as a typed event *after* the write,
    # so the audit log only ever points at files that actually exist.
    persisted = await store.list_events()
    spilled = [e for e in persisted if e.type == "tool.result.spilled"]
    assert len(spilled) == 1
    assert spilled[0].workspace_path == path
    assert spilled[0].rendered_size_bytes == len(content)
    spill_failed = [e for e in persisted if e.type == "tool.result.spill_failed"]
    assert spill_failed == []
    # Returned ToolCallCompleted carries the truncated pointer.
    assert path in completed.result.stdout


@pytest.mark.asyncio
async def test_oversized_tool_output_spill_failure_emits_spill_failed_event(
    tmp_path: Path, browser_use_service: BrowserUseService
, web_fetch_waiter: WebFetchExtractionWaiter, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
    class _BrokenWriteRuntime(FakeUserRuntime):
        async def write_content_file(
            self,
            user_id: str,
            path: str,
            content: bytes,
        ) -> RuntimeToolResult:
            return RuntimeToolResult(stderr="disk full", exit_code=1)

    runtime = _BrokenWriteRuntime(shell_results=[RuntimeToolResult(stdout="x" * 30_001)])
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    executor = ToolCallExecutor(
        runtime=runtime,
        bus=bus,
        max_model_output_chars=20_000,
        browser_use_service=browser_use_service, web_fetch_waiter=web_fetch_waiter, task_store=task_store, schedule_store=schedule_store, sub_agents=sub_agents)

    emitted = await executor.handle_tool_call_requested(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:disk",
            generation=1,
            call_id="oversized",
            tool_name="shell.exec",
            input=ShellExecInput(command="yes"),
        )
    )

    error_events = [e for e in emitted if e.type == "tool.call.error"]
    assert len(error_events) == 1
    assert "disk full" in error_events[0].error
    persisted = await store.list_events()
    spill_failed = [e for e in persisted if e.type == "tool.result.spill_failed"]
    spilled = [e for e in persisted if e.type == "tool.result.spilled"]
    assert len(spill_failed) == 1
    assert spilled == []
    assert spill_failed[0].error == "disk full"
    assert spill_failed[0].workspace_path == (
        "/workspace/content/tool-results/cli-disk-1-oversized.txt"
    )


@pytest.mark.asyncio
async def test_web_fetch_non_2xx_response_emits_failed_event() -> None:
    """A non-2xx HTTP response must become a typed
    WebFetchExtractionFailed -- never a silent retry or empty success."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(404, text="not found")
    )
    llm = FakeLlmClient([])  # LLM must not be touched on HTTP failure
    fetcher = HttpxWebFetcher(llm=llm, transport=transport)

    emitted = await fetcher.handle_extraction_requested(
        events.WebFetchExtractionRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=3,
            call_id="fetch-404",
            url="https://example.test/missing",
            prompt="Find something.",
            max_bytes=20_000,
        )
    )

    assert len(emitted) == 1
    failed = emitted[0]
    assert isinstance(failed, events.WebFetchExtractionFailed)
    assert failed.error == "HTTP 404"
    assert failed.conversation_id == "cli:1"
    assert failed.call_id == "fetch-404"
    assert llm.requests == []


@pytest.mark.asyncio
async def test_web_fetch_truncates_oversized_markdown_before_llm_extraction() -> None:
    """Content larger than max_bytes must be truncated before being
    passed to the LLM, and the LLM must see the truncation marker so
    its answer is not silently based on a complete page."""
    big_body = "x" * 5_000
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=big_body,
        )
    )
    llm = FakeLlmClient([AssistantText(text="truncated answer")])
    fetcher = HttpxWebFetcher(llm=llm, transport=transport)

    emitted = await fetcher.handle_extraction_requested(
        events.WebFetchExtractionRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=3,
            call_id="fetch-big",
            url="https://example.test/big",
            prompt="Answer the question.",
            max_bytes=1_000,
        )
    )

    assert len(emitted) == 1
    completed = emitted[0]
    assert isinstance(completed, events.WebFetchExtractionCompleted)
    assert completed.answer == "truncated answer"
    assert len(llm.requests) == 1
    sent_text = llm.requests[0].messages[0].text
    # LLM must see truncation marker AND must not see the full 5_000 bytes.
    assert "[Content truncated before analysis.]" in sent_text
    # Extract the markdown block we sent and assert it was clipped to
    # max_bytes; the URL "example.test" contributes one x to the full
    # request text, so we count only inside the markdown body.
    markdown_body = sent_text.split("Markdown:\n", 1)[1]
    body_only = markdown_body.split("\n\n[Content truncated before analysis.]")[0]
    assert len(body_only) <= 1_000
    assert body_only == "x" * 1_000


@pytest.mark.asyncio
async def test_executor_translates_web_fetch_failed_event_into_tool_error_result(
    tmp_path: Path, browser_use_service: BrowserUseService
, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
    """When the extraction handler emits WebFetchExtractionFailed, the
    tool executor must surface it as a ToolCallCompleted with exit_code=1
    and the error in stderr -- not a silent success."""
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    waiter = WebFetchExtractionWaiter()
    bus.subscribe(events.WebFetchExtractionCompleted, waiter.handle_completed)
    bus.subscribe(events.WebFetchExtractionFailed, waiter.handle_failed)

    async def emit_failure(event: events.WebFetchExtractionRequested) -> tuple:
        return (
            events.WebFetchExtractionFailed(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                call_id=event.call_id,
                error="upstream said no",
            ),
        )

    bus.subscribe(events.WebFetchExtractionRequested, emit_failure)

    executor = ToolCallExecutor(
        runtime=FakeUserRuntime(),
        bus=bus,
        browser_use_service=browser_use_service,
        web_fetch_waiter=waiter, task_store=task_store, schedule_store=schedule_store, sub_agents=sub_agents)

    (completed,) = await executor.handle_tool_call_requested(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:fetch",
            generation=2,
            call_id="fetch-fail",
            tool_name="web.fetch",
            input=WebFetchInput(url="https://example.test", prompt="x"),
        )
    )
    assert completed.tool_name == "web.fetch"
    assert completed.result.exit_code == 1
    assert completed.result.stderr == "upstream said no"
    assert completed.result.stdout == ""


@pytest.mark.asyncio
async def test_web_fetch_waiter_isolates_concurrent_calls_with_same_call_id() -> None:
    """Two web.fetch calls from different conversations can share the
    same call_id (the model picks it). The waiter must key by the full
    (conversation_id, generation, call_id) tuple; otherwise one resolution
    can deliver the wrong answer or hang the other caller forever."""
    waiter = WebFetchExtractionWaiter()
    waiter.expect(conversation_id="cli:alice", generation=1, call_id="fetch-1")
    waiter.expect(conversation_id="cli:bob", generation=2, call_id="fetch-1")

    alice_task = asyncio.create_task(
        waiter.wait(conversation_id="cli:alice", generation=1, call_id="fetch-1")
    )
    bob_task = asyncio.create_task(
        waiter.wait(conversation_id="cli:bob", generation=2, call_id="fetch-1")
    )

    await waiter.handle_completed(
        events.WebFetchExtractionCompleted(
            user_id="u:1",
            conversation_id="cli:bob",
            generation=2,
            call_id="fetch-1",
            answer="bob answer",
        )
    )
    bob_result = await asyncio.wait_for(bob_task, timeout=1.0)
    assert isinstance(bob_result, events.WebFetchExtractionCompleted)
    assert bob_result.answer == "bob answer"
    assert not alice_task.done(), "alice's waiter must not be resolved by bob's event"

    await waiter.handle_completed(
        events.WebFetchExtractionCompleted(
            user_id="u:1",
            conversation_id="cli:alice",
            generation=1,
            call_id="fetch-1",
            answer="alice answer",
        )
    )
    alice_result = await asyncio.wait_for(alice_task, timeout=1.0)
    assert isinstance(alice_result, events.WebFetchExtractionCompleted)
    assert alice_result.answer == "alice answer"


def test_file_listing_tools_do_not_advertise_hard_result_caps() -> None:
    assert "max_results" not in default_tool_registry().by_name("file.list").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.glob").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.grep").parameters_schema()["properties"]
    assert "max_results" not in FileListInput.model_fields


@pytest.mark.asyncio
async def test_file_list_returns_all_entries_without_truncation(tmp_path: Path) -> None:
    """file.list must surface every directory entry; the runtime must
    not append a `| head -n N` or similar truncation. We assert this by
    feeding back 500 paths via a recording docker runner and checking
    they all land in the tool result intact."""
    from harness_agent.runtime import (
        DockerProcessResult,
        DockerUserRuntime,
        InMemorySpawnedProcessStore,
    )
    from harness_agent.tools import FileListInput

    expected_paths = [f"/workspace/dir/file_{i:04d}.txt" for i in range(500)]
    stdout = "\n".join(expected_paths) + "\n"

    class _Runner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def run(self, argv: list[str], **_: object) -> DockerProcessResult:
            self.calls.append(argv)
            return DockerProcessResult(stdout=stdout, stderr="", exit_code=0)

    runner = _Runner()
    runtime = DockerUserRuntime(
        runner=runner,  # type: ignore[arg-type]
        spawned_process_store=InMemorySpawnedProcessStore(),
    )

    result = await runtime.file_list("u:1", FileListInput(path="/workspace/dir"))
    assert result.stdout == stdout
    returned = [line for line in result.stdout.splitlines() if line]
    assert returned == expected_paths
    # The command sent to docker must not contain a truncation primitive.
    joined_argv = " ".join(runner.calls[0])
    for forbidden in ("head -n", "| head", " | head"):
        assert forbidden not in joined_argv, f"truncation detected: {forbidden!r} in {joined_argv}"


@pytest.mark.asyncio
async def test_file_glob_returns_all_matches_without_truncation(tmp_path: Path) -> None:
    from harness_agent.runtime import (
        DockerProcessResult,
        DockerUserRuntime,
        InMemorySpawnedProcessStore,
    )
    from harness_agent.tools import FileGlobInput

    expected = [f"/workspace/src/mod_{i:04d}.py" for i in range(500)]
    stdout = "\n".join(expected) + "\n"

    class _Runner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def run(self, argv: list[str], **_: object) -> DockerProcessResult:
            self.calls.append(argv)
            return DockerProcessResult(stdout=stdout, stderr="", exit_code=0)

    runner = _Runner()
    runtime = DockerUserRuntime(
        runner=runner,  # type: ignore[arg-type]
        spawned_process_store=InMemorySpawnedProcessStore(),
    )

    result = await runtime.file_glob("u:1", FileGlobInput(pattern="**/*.py"))
    returned = [line for line in result.stdout.splitlines() if line]
    assert returned == expected
    joined_argv = " ".join(runner.calls[0])
    assert "head" not in joined_argv


@pytest.mark.asyncio
async def test_file_grep_returns_all_matches_without_truncation(tmp_path: Path) -> None:
    from harness_agent.runtime import (
        DockerProcessResult,
        DockerUserRuntime,
        InMemorySpawnedProcessStore,
    )
    from harness_agent.tools import FileGrepInput

    expected = [
        f"/workspace/src/file_{i:04d}.py:{i}:TODO investigate this"
        for i in range(500)
    ]
    stdout = "\n".join(expected) + "\n"

    class _Runner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def run(self, argv: list[str], **_: object) -> DockerProcessResult:
            self.calls.append(argv)
            return DockerProcessResult(stdout=stdout, stderr="", exit_code=0)

    runner = _Runner()
    runtime = DockerUserRuntime(
        runner=runner,  # type: ignore[arg-type]
        spawned_process_store=InMemorySpawnedProcessStore(),
    )

    result = await runtime.file_grep(
        "u:1", FileGrepInput(pattern="TODO", path="/workspace/src")
    )
    returned = [line for line in result.stdout.splitlines() if line]
    assert returned == expected
    joined_argv = " ".join(runner.calls[0])
    for forbidden in ("head -n", " | head", "--max-count"):
        assert forbidden not in joined_argv


@pytest.mark.asyncio
async def test_web_fetch_converts_html_to_markdown_and_filters_with_secondary_model(
    tmp_path: Path,
) -> None:
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
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    fetcher = HttpxWebFetcher(llm=llm, transport=transport)
    completed_events: list = []
    failed_events: list = []
    bus.subscribe(
        events.WebFetchExtractionCompleted,
        lambda e: completed_events.append(e) or asyncio.sleep(0),  # type: ignore[arg-type]
    )
    bus.subscribe(
        events.WebFetchExtractionFailed,
        lambda e: failed_events.append(e) or asyncio.sleep(0),  # type: ignore[arg-type]
    )

    emitted = await fetcher.handle_extraction_requested(
        events.WebFetchExtractionRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=7,
            call_id="fetch-1",
            url="https://example.test/article",
            prompt="Find the main instruction.",
            max_bytes=20_000,
        )
    )

    assert len(emitted) == 1
    completed = emitted[0]
    assert isinstance(completed, events.WebFetchExtractionCompleted)
    assert completed.answer == "Use the docs first."
    # Real conversation context flows into the LLM audit row.
    assert completed.user_id == "u:1"
    assert completed.conversation_id == "cli:1"
    assert completed.generation == 7
    assert completed.call_id == "fetch-1"

    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request.user_id == "u:1"
    assert request.conversation_id == "cli:1"
    assert request.generation == 7
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
async def test_web_fetch_result_does_not_expose_raw_markdown_to_main_history(tmp_path: Path, browser_use_service: BrowserUseService, web_fetch_waiter: WebFetchExtractionWaiter, task_store: SQLiteTaskStore, schedule_store: SQLiteScheduleStore, sub_agents: SubAgentService) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    bus = EventBus(store)
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
        tool_results=tool_results, sub_agent_lookup=NullSubAgentLookup())
    fetcher = HttpxWebFetcher(llm=web_llm, transport=transport)
    web_fetch_waiter = WebFetchExtractionWaiter()
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        web_fetch_waiter=web_fetch_waiter,
        browser_use_service=browser_use_service,
        bus=bus, task_store=task_store, schedule_store=schedule_store, sub_agents=sub_agents)
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection).handle_tool_call_completed)
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)
    bus.subscribe(events.WebFetchExtractionRequested, fetcher.handle_extraction_requested)
    bus.subscribe(events.WebFetchExtractionCompleted, web_fetch_waiter.handle_completed)
    bus.subscribe(events.WebFetchExtractionFailed, web_fetch_waiter.handle_failed)

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
