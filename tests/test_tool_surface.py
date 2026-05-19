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
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.store import SQLiteEventStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    FileEditInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    TaskCreateInput,
    default_tool_registry,
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
