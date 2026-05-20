from pathlib import Path

import pytest

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
    WebFetchInput,
    default_tool_registry,
)
from harness_agent.web_fetch import HttpxWebFetcher

import httpx


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
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime, task_store=task_store)
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
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime, task_store=task_store)
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
    tool_executor = ToolCallExecutor(runtime=FakeUserRuntime())
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
    assert completed.result.stderr == "sub-agent service is not configured"
    assert completed.result.stdout == ""
    assert [event.type for event in await store.list_events()] == [
        "tool.call.requested",
        "tool.call.error",
        "tool.call.completed",
    ]


@pytest.mark.asyncio
async def test_oversized_tool_output_is_saved_to_workspace_content() -> None:
    runtime = FakeUserRuntime(shell_results=[RuntimeToolResult(stdout="x" * 30_001)])
    executor = ToolCallExecutor(runtime=runtime, max_model_output_chars=20_000)

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


def test_file_listing_tools_do_not_advertise_hard_result_caps() -> None:
    assert "max_results" not in default_tool_registry().by_name("file.list").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.glob").parameters_schema()["properties"]
    assert "max_results" not in default_tool_registry().by_name("file.grep").parameters_schema()["properties"]
    assert "max_results" not in FileListInput.model_fields


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


@pytest.mark.asyncio
async def test_web_fetch_result_does_not_expose_raw_markdown_to_main_history(tmp_path: Path) -> None:
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
        tool_results=tool_results,
    )
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        web_fetcher=HttpxWebFetcher(llm=web_llm, transport=transport),
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection).handle_tool_call_completed)
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
