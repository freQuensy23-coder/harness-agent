from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SubAgentRecord
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    AgentSpawnInput,
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    TaskCreateInput,
    WebFetchInput,
    default_tool_registry,
)
import harness_agent.web_fetch as web_fetch


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
async def test_tool_exception_completes_as_failed_result_and_error_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    executor = ToolCallExecutor(runtime=FakeUserRuntime())
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)

    await bus.publish(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=1,
            call_id="spawn",
            tool_name="agent.spawn",
            input=AgentSpawnInput(prompt="do work"),
        )
    )

    events = await store.list_events()
    assert [event.type for event in events] == [
        "tool.call.requested",
        "tool.call.error",
        "tool.call.completed",
    ]
    assert getattr(events[1], "error") == "sub-agent service is not configured"
    completed = events[2]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result.exit_code == 1
    assert completed.result.stderr == "sub-agent service is not configured"


@pytest.mark.asyncio
async def test_oversized_tool_output_is_saved_to_workspace_data(tmp_path: Path) -> None:
    long_output = "x" * 30_001
    runtime = FakeUserRuntime(shell_results=[RuntimeToolResult(stdout=long_output)])
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    executor = ToolCallExecutor(runtime=runtime)
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)

    await bus.publish(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=1,
            call_id="run",
            tool_name="shell.exec",
            input=ShellExecInput(command="generate-output"),
        )
    )

    completed = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ][0]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result.exit_code == 0
    assert long_output not in completed.result.stdout
    assert completed.result.stderr == ""
    assert completed.result.stdout.startswith("truncated, because it is too long.")
    assert "Full result saved to /workspace/content/tool-results/" in completed.result.stdout
    assert len(runtime.content_write_calls) == 1
    saved_path, saved_content = runtime.content_write_calls[0]
    assert saved_path.startswith("/workspace/content/tool-results/")
    assert saved_path.endswith(".txt")
    assert b"shell.exec stdout:\n" in saved_content
    assert long_output.encode("utf-8") in saved_content


def test_file_scan_tool_inputs_do_not_expose_builtin_result_limits() -> None:
    for input_model in (FileGlobInput, FileGrepInput, FileListInput):
        assert "max_results" not in input_model.model_json_schema()["properties"]


def test_web_fetch_requires_an_extraction_prompt() -> None:
    with pytest.raises(ValidationError):
        WebFetchInput(url="https://example.com")


@pytest.mark.asyncio
async def test_web_fetch_filters_markdown_through_subagent(tmp_path: Path) -> None:
    fetcher = FakeMarkdownFetcher(
        markdown="# Pricing\n\nThe pro plan costs $20.",
        final_url="https://example.com/pricing",
    )
    sub_agents = FakeWebFetchSubAgents(result="The pro plan costs $20.")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    executor = ToolCallExecutor(
        runtime=FakeUserRuntime(),
        web_fetcher=fetcher,
        sub_agents=sub_agents,
    )
    bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)

    await bus.publish(
        ToolCallRequested(
            user_id="u:1",
            conversation_id="cli:1",
            generation=1,
            call_id="fetch",
            tool_name="web.fetch",
            input=WebFetchInput(
                url="http://example.com/pricing",
                prompt="Find the pro plan price.",
            ),
        )
    )

    completed = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ][0]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.result.stdout == "The pro plan costs $20."
    assert completed.result.exit_code == 0
    assert fetcher.inputs == [
        WebFetchInput(url="http://example.com/pricing", prompt="Find the pro plan price.")
    ]
    assert sub_agents.inputs[0].name == "web-fetch"
    assert "Find the pro plan price." in sub_agents.inputs[0].prompt
    assert "https://example.com/pricing" in sub_agents.inputs[0].prompt
    assert "# Pricing" in sub_agents.inputs[0].prompt
    assert "# Pricing" not in completed.result.stdout


def test_html_to_markdown_extracts_readable_content() -> None:
    markdown = web_fetch.html_to_markdown(
        """
        <html>
          <head><title>Ignored</title><script>bad()</script></head>
          <body>
            <nav>Menu</nav>
            <article>
              <h1>Release Notes</h1>
              <p>Hello <a href="/docs">docs</a></p>
              <ul><li>One</li><li>Two</li></ul>
            </article>
          </body>
        </html>
        """,
        base_url="https://example.com/base/",
    )

    assert "# Release Notes" in markdown
    assert "Hello [docs](https://example.com/docs)" in markdown
    assert "- One" in markdown
    assert "- Two" in markdown
    assert "bad()" not in markdown
    assert "Menu" not in markdown


@dataclass
class FakeFetchedPage:
    url: str
    markdown: str


class FakeMarkdownFetcher:
    def __init__(self, *, markdown: str, final_url: str) -> None:
        self._markdown = markdown
        self._final_url = final_url
        self.inputs: list[WebFetchInput] = []

    async def fetch_markdown(self, input: WebFetchInput) -> FakeFetchedPage:
        self.inputs.append(input)
        return FakeFetchedPage(url=self._final_url, markdown=self._markdown)


class FakeWebFetchSubAgents:
    def __init__(self, *, result: str) -> None:
        self._result = result
        self.inputs = []

    async def run(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input,
    ) -> SubAgentRecord:
        self.inputs.append(input)
        now = datetime.now(UTC)
        return SubAgentRecord(
            id="agent-web-fetch",
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            child_conversation_id=f"{parent_conversation_id}:subagent:agent-web-fetch",
            parent_call_id=parent_call_id,
            name=input.name,
            prompt=input.prompt,
            status="completed",
            result=self._result,
            created_at=now,
            updated_at=now,
        )
