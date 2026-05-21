import asyncio
import json
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.compaction import ContextCompactionConfig, ContextCompactor
from harness_agent.context import AgentFileSet, ContextBuilder, Skill
from harness_agent.events import (
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    CliTextReceived,
    ContextCompacting,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import (
    AgentTurnHandler,
    ConversationProjector,
    IdentityHandler,
    TelegramReplyHandler,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import (
    AssistantText,
    AssistantMessage,
    FakeLlmClient,
    LlmClient,
    LlmRequest,
    LlmToolCall,
    AssistantToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from harness_agent.llm_audit import AuditedLlmClient, SQLiteLlmAuditStore
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import (
    FakeUserRuntime,
    RuntimeToolResult,
)
from harness_agent.store import SQLiteEventStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import ShellExecInput, default_tool_registry


@pytest.mark.asyncio
async def test_telegram_say_hi_builds_context_from_runtime_and_replies(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        files={
            "/workspace/agent/SOUL.md": "SOUL: answer tersely.",
            "/workspace/agent/AGENTS.md": "AGENTS: obey the user.",
            "/workspace/agent/USER.md": "USER: Alex.",
            "/workspace/agent/TOOLS.md": "TOOLS: use tools only when needed.",
        },
        skills=[
            Skill(
                name="shell-work",
                description="Use shell commands in the workspace.",
                body="Shell commands must stay inside /workspace.",
            )
        ],
    )
    llm = FakeLlmClient([AssistantText(text="Hi.")])
    replies: list[tuple[int, str]] = []

    bus = EventBus(store)
    identity_handler = IdentityHandler(StaticIdentityResolver())
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
    )
    bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(
        AssistantTextProduced,
        TelegramReplyHandler(replies).handle_assistant_text,
    )

    await bus.publish(
        TelegramTextReceived(
            telegram_user_id=123,
            telegram_chat_id=456,
            telegram_message_id=999,
            text="Say hi",
        )
    )

    assert replies == [(456, "Hi.")]
    assert [event.type for event in await store.list_events()] == [
        "telegram.text.received",
        "user.text.received",
        "agent.turn.requested",
        "agent.generation.started",
        "assistant.text.produced",
    ]
    assert await projection.list_messages("tg:456") == [
        ("user", "Say hi"),
        ("assistant", "Hi."),
    ]

    request = llm.requests[0]
    assert request.system == "\n\n".join(
        [
            "SOUL: answer tersely.",
            "AGENTS: obey the user.",
            "USER: Alex.",
            "TOOLS: use tools only when needed.",
            "Skill: shell-work\nUse shell commands in the workspace.\nShell commands must stay inside /workspace.",
            "\n".join(
                [
                    "Tools:",
                    "- shell.exec runs commands in /workspace.",
                    "- shell.spawn starts long-running commands in /workspace.",
                    "- shell.read reads spawned command output.",
                    "- shell.kill stops spawned commands.",
                    "- file.read reads files under /workspace.",
                    "- file.write writes files under /workspace.",
                    "- file.edit replaces exact text in one file.",
                    "- file.multi_edit applies exact replacements to one file.",
                    "- file.glob finds files under /workspace.",
                    "- file.grep searches files under /workspace.",
                    "- file.list lists paths under /workspace.",
                    "- web.fetch fetches HTTP/HTTPS text.",
                    "- task.* manages the conversation checklist.",
                    "- schedule.once schedules one future synthetic user message.",
                    "- schedule.cron schedules recurring synthetic user messages.",
                    "- schedule.list and schedule.cancel manage scheduled messages.",
                    "- skill.* reads enabled markdown skills.",
                    "- agent.* runs sub-agents that can use workspace file and shell tools.",
                ]
            ),
            "Runtime: tools run in the user's workspace. Container details are not part of the model context.",
            "Incoming Telegram files are saved under /workspace/content. Use file.read for saved text files. Images are also attached to multimodal user messages when available.",
            "Do not use sleep, wait, or long-running bash commands to schedule future work. Use schedule.once or schedule.cron.",
        ]
    )
    assert request.messages == [UserMessage(text="Say hi")]
    assert [tool.name for tool in request.tools] == [
        "shell.exec",
        "shell.spawn",
        "shell.read",
        "shell.kill",
        "file.read",
        "file.write",
        "file.edit",
        "file.multi_edit",
        "file.glob",
        "file.grep",
        "file.list",
        "web.fetch",
        "task.create",
        "task.get",
        "task.list",
        "task.update",
        "task.stop",
        "schedule.once",
        "schedule.cron",
        "schedule.list",
        "schedule.cancel",
        "skill.list",
        "skill.read",
        "agent.run",
        "agent.spawn",
        "agent.result",
        "agent.list",
        "agent.cancel",
    ]
    assert not any("docker" in tool.name for tool in request.tools)
    assert runtime.read_agent_files_calls == ["u:123"]
    assert runtime.list_skills_calls == ["u:123"]
    assert runtime.shell_exec_calls == []


@pytest.mark.asyncio
async def test_tool_call_executes_in_runtime_without_exposing_docker(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(
            soul="SOUL",
            agents="AGENTS",
            user="USER",
            tools="TOOLS",
        ),
        shell_results=[
            RuntimeToolResult(
                stdout="/workspace\n",
                stderr="",
                exit_code=0,
            )
        ],
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="call_1",
                name="shell.exec",
                input=ShellExecInput(command="pwd", cwd="/workspace"),
            ),
            AssistantText(text="done"),
        ]
    )

    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime)
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
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(
        AgentTurnRequested,
        agent_turn_handler.handle_agent_turn,
    )

    await bus.publish(
        UserTextReceived(
            user_id="u_1",
            conversation_id="tg:456",
            source="telegram",
            text="run pwd",
            reply_target=None,
        )
    )

    assert [event.type for event in await store.list_events()] == [
        "user.text.received",
        "agent.turn.requested",
        "agent.generation.started",
        "tool.call.requested",
        "tool.call.completed",
        "agent.generation.started",
        "assistant.text.produced",
    ]
    assert (await store.list_events())[3].type == "tool.call.requested"
    assert runtime.shell_exec_calls == [ShellExecInput(command="pwd", cwd="/workspace")]
    assert llm.requests[0].tools[0].name == "shell.exec"
    assert not any("docker" in request.system.lower() for request in llm.requests)
    assert llm.requests[1].messages[-2:] == [
        AssistantToolCallMessage(
            call_id="call_1",
            name="shell.exec",
            arguments={"command": "pwd", "cwd": "/workspace", "timeout_seconds": 60},
        ),
        ToolResultMessage(
            call_id="call_1",
            name="shell.exec",
            content="shell.exec stdout:\n/workspace\nstderr:\n\nexit_code: 0",
        ),
    ]


@pytest.mark.asyncio
async def test_identity_handler_turns_telegram_event_into_user_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    identity_handler = IdentityHandler(StaticIdentityResolver())
    bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)

    await bus.publish(
        TelegramTextReceived(
            telegram_user_id=321,
            telegram_chat_id=654,
            telegram_message_id=111,
            text="Say hi",
        )
    )

    assert await store.list_events() == [
        TelegramTextReceived(
            telegram_user_id=321,
            telegram_chat_id=654,
            telegram_message_id=111,
            text="Say hi",
            id=(await store.list_events())[0].id,
            occurred_at=(await store.list_events())[0].occurred_at,
        ),
        UserTextReceived(
            user_id="u:321",
            conversation_id="tg:654",
            source="telegram",
            text="Say hi",
            reply_target={"kind": "telegram", "chat_id": 654},
            id=(await store.list_events())[1].id,
            occurred_at=(await store.list_events())[1].occurred_at,
        ),
    ]


@pytest.mark.asyncio
async def test_identity_handler_turns_cli_event_into_user_event(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    identity_handler = IdentityHandler(StaticIdentityResolver())
    bus.subscribe(CliTextReceived, identity_handler.handle_cli_text)

    await bus.publish(
        CliTextReceived(
            cli_user_id="123",
            conversation_id="cli:123",
            request_id="req_1",
            text="Say hi",
        )
    )

    assert await store.list_events() == [
        CliTextReceived(
            cli_user_id="123",
            conversation_id="cli:123",
            request_id="req_1",
            text="Say hi",
            id=(await store.list_events())[0].id,
            occurred_at=(await store.list_events())[0].occurred_at,
        ),
        UserTextReceived(
            user_id="u:123",
            conversation_id="cli:123",
            source="cli",
            text="Say hi",
            reply_target={"kind": "cli", "request_id": "req_1"},
            id=(await store.list_events())[1].id,
            occurred_at=(await store.list_events())[1].occurred_at,
        ),
    ]


@pytest.mark.asyncio
async def test_agent_turn_uses_persisted_conversation_history(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = FakeLlmClient([AssistantText(text="first"), AssistantText(text="second")])
    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:history",
            source="cli",
            text="one",
        )
    )
    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:history",
            source="cli",
            text="two",
        )
    )

    assert llm.requests[0].messages == [UserMessage(text="one")]
    assert llm.requests[1].messages == [
        UserMessage(text="one"),
        AssistantMessage(text="first"),
        UserMessage(text="two"),
    ]


@pytest.mark.asyncio
async def test_tool_calls_and_results_are_persisted_in_llm_history(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
        shell_results=[RuntimeToolResult(stdout="/workspace\n")],
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="call_1",
                name="shell.exec",
                input=ShellExecInput(command="pwd", cwd="/workspace"),
            ),
            AssistantText(text="done"),
            AssistantText(text="second"),
        ]
    )
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        tool_results=tool_results,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:history-tools",
            source="cli",
            text="run pwd",
        )
    )
    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:history-tools",
            source="cli",
            text="again",
        )
    )

    exchange = [
        AssistantToolCallMessage(
            call_id="call_1",
            name="shell.exec",
            arguments={"command": "pwd", "cwd": "/workspace", "timeout_seconds": 60},
        ),
        ToolResultMessage(
            call_id="call_1",
            name="shell.exec",
            content="shell.exec stdout:\n/workspace\nstderr:\n\nexit_code: 0",
        ),
    ]
    assert await projection.list_llm_messages("cli:history-tools") == [
        UserMessage(text="run pwd"),
        *exchange,
        AssistantMessage(text="done"),
        UserMessage(text="again"),
        AssistantMessage(text="second"),
    ]
    assert llm.requests[2].messages == [
        UserMessage(text="run pwd"),
        *exchange,
        AssistantMessage(text="done"),
        UserMessage(text="again"),
    ]


@pytest.mark.asyncio
async def test_tool_history_bytes_are_identical_in_next_llm_request(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    audit_store = SQLiteLlmAuditStore(tmp_path / "llm.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
        shell_results=[RuntimeToolResult(stdout="byte-proof\n")],
    )
    inner_llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="call_1",
                name="shell.exec",
                input=ShellExecInput(command="printf byte-proof", cwd="/workspace"),
            ),
            AssistantText(text="done"),
            AssistantText(text="hi"),
        ]
    )
    llm = AuditedLlmClient(inner=inner_llm, store=audit_store)
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        tool_results=tool_results,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:bytes",
            source="cli",
            text="run tool",
        )
    )
    tool_history_bytes = await projection.list_tool_history_json("cli:bytes")

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:bytes",
            source="cli",
            text="say hi",
        )
    )

    requests = await audit_store.list_requests("cli:bytes")
    second_turn_messages = requests[2].message_json
    second_turn_tool_history = [
        message
        for message in second_turn_messages
        if json.loads(message)["kind"] in {"assistant_tool_call", "tool_result"}
    ]
    assert second_turn_tool_history == tool_history_bytes


@pytest.mark.asyncio
async def test_new_message_supersedes_running_turn_before_tool_side_effect(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
        shell_results=[RuntimeToolResult(stdout="should-not-run\n")],
    )
    llm = BlockingToolThenTextLlm()
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    replies: list[str] = []
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(AssistantTextProduced, capture_current_reply(replies, coordinator))

    first = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:race",
                source="cli",
                text="old",
            )
        )
    )
    await llm.first_started.wait()
    second = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:race",
                source="cli",
                text="new",
            )
        )
    )
    await wait_until_generation(coordinator, conversation_id="cli:race", generation=2)
    llm.release_first.set()
    await first
    await second

    events = await store.list_events()
    assert [event.type for event in events if event.type == "agent.turn.superseded"] == [
        "agent.turn.superseded"
    ]
    superseded = [event for event in events if event.type == "agent.turn.superseded"][0]
    assert superseded == AgentTurnSuperseded(
        user_id="u:1",
        conversation_id="cli:race",
        generation=1,
        superseded_by=2,
        reason="newer_user_message",
        id=superseded.id,
        occurred_at=superseded.occurred_at,
    )
    assert runtime.shell_exec_calls == []
    assert replies == ["latest"]
    assert llm.requests[1].messages == [UserMessage(text="old"), UserMessage(text="new")]


@pytest.mark.asyncio
async def test_context_compaction_archives_summary_and_keeps_last_two_messages(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = FakeLlmClient(
        [
            AssistantText(
                text="<analysis>scratch thoughts</analysis>\n<summary>compact summary</summary>"
            ),
            AssistantText(text="done"),
        ]
    )
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:compact",
        text="old user",
    )
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:compact",
        generation=1,
        text="old assistant",
    )
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:compact",
        text="recent user",
    )

    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        compactor=ContextCompactor(
            projection=projection,
            runtime=runtime,
            llm=llm,
            config=ContextCompactionConfig(max_tokens_per_model=20, reserve_tokens=0),
            estimate_tokens=lambda request: 20,
        ),
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:compact",
            source="cli",
            text="latest user",
        )
    )

    assert [event.type for event in await store.list_events()] == [
        "user.text.received",
        "agent.turn.requested",
        "context.compacting",
        "agent.generation.started",
        "assistant.text.produced",
    ]
    compacting = [event for event in await store.list_events() if event.type == "context.compacting"][0]
    assert compacting == ContextCompacting(
        user_id="u:1",
        conversation_id="cli:compact",
        generation=1,
        token_estimate=20,
        threshold=20,
        keep_last_messages=2,
        id=compacting.id,
        occurred_at=compacting.occurred_at,
    )
    assert llm.requests[0].tools == []
    assert llm.requests[0].messages == [
        UserMessage(text="old user"),
        AssistantMessage(text="old assistant"),
    ]
    assert llm.requests[1].messages == [
        UserMessage(text="Previous conversation summary:\ncompact summary"),
        UserMessage(text="recent user"),
        UserMessage(text="latest user"),
    ]
    assert all(
        "scratch thoughts" not in message.text
        for message in llm.requests[1].messages
        if hasattr(message, "text")
    )
    assert runtime.file_write_calls[0].path.startswith("/workspace/.old-sessions/")
    archive_lines = runtime.file_write_calls[0].content.splitlines()
    assert [json.loads(line)["item_kind"] for line in archive_lines] == [
        "user",
        "assistant",
    ]
    all_records = await projection.list_all_context_items("cli:compact")
    assert [record.item_kind for record in all_records] == [
        "user",
        "assistant",
        "user",
        "user",
        "context_summary",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_context_compaction_runs_between_tool_calls_and_keeps_tools_available(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
        shell_results=[
            RuntimeToolResult(stdout="one\n"),
            RuntimeToolResult(stdout="two\n"),
        ],
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="call_1",
                name="shell.exec",
                input=ShellExecInput(command="printf one", cwd="/workspace"),
            ),
            LlmToolCall(
                call_id="call_2",
                name="shell.exec",
                input=ShellExecInput(command="printf two", cwd="/workspace"),
            ),
            AssistantText(text="<summary>tool summary</summary>"),
            AssistantText(text="done"),
        ]
    )
    estimator = SequenceEstimator([0, 0, 20])

    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        tool_results=tool_results,
        compactor=ContextCompactor(
            projection=projection,
            runtime=runtime,
            llm=llm,
            config=ContextCompactionConfig(
                max_tokens_per_model=20,
                reserve_tokens=0,
                keep_last_messages=1,
            ),
            estimate_tokens=estimator,
        ),
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:compact-tools",
        text="prior task",
    )
    await projection.append_tool_exchange(
        user_id="u:1",
        conversation_id="cli:compact-tools",
        generation=1,
        call_id="prior_call",
        tool_name="shell.exec",
        input=ShellExecInput(command="prep", cwd="/workspace"),
        result=RuntimeToolResult(stdout="ready\n"),
    )
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:compact-tools",
        generation=1,
        text="prior done",
    )

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:compact-tools",
            source="cli",
            text="run two tools",
        )
    )

    assert [request.tools == [] for request in llm.requests] == [
        False,
        False,
        True,
        False,
    ]
    assert [m.kind for m in llm.requests[2].messages] == [
        "user",
        "assistant_tool_call",
        "tool_result",
        "assistant",
    ]
    assert llm.requests[3].messages[-2:] == [
        AssistantToolCallMessage(
            call_id="call_2",
            name="shell.exec",
            arguments={
                "command": "printf two",
                "cwd": "/workspace",
                "timeout_seconds": 60,
            },
        ),
        ToolResultMessage(
            call_id="call_2",
            name="shell.exec",
            content="shell.exec stdout:\ntwo\nstderr:\n\nexit_code: 0",
        ),
    ]
    assert [call.command for call in runtime.shell_exec_calls] == [
        "printf one",
        "printf two",
    ]
    assert [event.type for event in await store.list_events()].count("context.compacting") == 1


@pytest.mark.asyncio
async def test_new_user_message_during_compaction_uses_fresh_history_without_stale_summary(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = BlockingSummaryLlm()
    estimator = SequenceEstimator([20, 0])
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        compactor=ContextCompactor(
            projection=projection,
            runtime=runtime,
            llm=llm,
            config=ContextCompactionConfig(max_tokens_per_model=20, reserve_tokens=0),
            estimate_tokens=estimator,
        ),
    )
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:compact-race",
        text="ancient user",
    )
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:compact-race",
        generation=1,
        text="ancient assistant",
    )
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:compact-race",
        text="old user",
    )
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:compact-race",
        generation=1,
        text="old assistant",
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    first = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:compact-race",
                source="cli",
                text="stale user",
            )
        )
    )
    await llm.summary_started.wait()
    second = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:compact-race",
                source="cli",
                text="fresh user",
            )
        )
    )
    await wait_until_generation(coordinator, conversation_id="cli:compact-race", generation=2)
    llm.release_summary.set()
    await first
    await second

    assert runtime.file_write_calls == []
    assert "Previous conversation summary" not in [
        message.text for message in llm.requests[1].messages if message.kind == "user"
    ]
    assert llm.requests[1].messages == [
        UserMessage(text="ancient user"),
        AssistantMessage(text="ancient assistant"),
        UserMessage(text="old user"),
        AssistantMessage(text="old assistant"),
        UserMessage(text="stale user"),
        UserMessage(text="fresh user"),
    ]
    assert [record.item_kind for record in await projection.list_all_context_items("cli:compact-race")] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "user",
        "assistant",
    ]
    assert [event.type for event in await store.list_events()].count("agent.turn.superseded") == 1


@pytest.mark.asyncio
async def test_compaction_skipped_when_exactly_n_user_messages_with_first_at_index_zero(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first user")
    await projection.append_tool_exchange(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        call_id="a",
        tool_name="shell.exec",
        input=ShellExecInput(command="ls", cwd="/workspace"),
        result=RuntimeToolResult(stdout="ok\n"),
    )
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second user")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient([])
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=2,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is None
    assert llm.requests == []


@pytest.mark.asyncio
async def test_compaction_falls_back_to_safe_boundary_when_user_messages_are_too_few(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="only user")
    await projection.append_tool_exchange(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        call_id="a",
        tool_name="shell.exec",
        input=ShellExecInput(command="ls", cwd="/workspace"),
        result=RuntimeToolResult(stdout="ok\n"),
    )
    await projection.append_tool_exchange(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        call_id="b",
        tool_name="shell.exec",
        input=ShellExecInput(command="pwd", cwd="/workspace"),
        result=RuntimeToolResult(stdout="/workspace\n"),
    )
    runtime = FakeUserRuntime()
    llm = FakeLlmClient([AssistantText(text="<summary>tool churn summary</summary>")])
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=2,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    assert [r.item_kind for r in snapshot.compacted_records] == [
        "user",
        "assistant_tool_call",
        "tool_result",
    ]
    assert [r.item_kind for r in snapshot.tail_records] == [
        "assistant_tool_call",
        "tool_result",
    ]


@pytest.mark.asyncio
async def test_compaction_retries_when_model_omits_summary_tag(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient(
        [
            AssistantText(text="<analysis>forgot the summary tag</analysis>"),
            AssistantText(
                text="<analysis>retry</analysis>\n<summary>retry summary</summary>"
            ),
        ]
    )
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )

    assert draft.summary == "retry summary"
    assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_compaction_strips_analysis_from_fallback_when_summary_tag_missing(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient(
        [
            AssistantText(text="<analysis>noise A</analysis> body one"),
            AssistantText(text="<analysis>noise B</analysis> body two"),
        ]
    )
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )

    assert draft.summary == "body two"
    assert "noise" not in draft.summary
    assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_compaction_fallback_uses_raw_text_when_strip_leaves_nothing(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient(
        [
            AssistantText(text="<analysis>only this</analysis>"),
            AssistantText(text="<analysis>still only this</analysis>"),
        ]
    )
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )

    assert draft.summary == "<analysis>still only this</analysis>"


@pytest.mark.asyncio
async def test_compaction_preserves_literal_summary_tag_inside_summary_body(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient(
        [
            AssistantText(
                text=(
                    "<analysis>thinking</analysis>\n"
                    "<summary>kept code: <summary>literal</summary> remains</summary>"
                )
            ),
        ]
    )
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )

    assert draft.summary == "kept code: <summary>literal</summary> remains"


@pytest.mark.asyncio
async def test_compaction_picks_trailing_summary_when_analysis_quotes_a_summary_tag(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime()
    llm = FakeLlmClient(
        [
            AssistantText(
                text=(
                    "<analysis>user quoted &lt;summary&gt;fake nested&lt;/summary&gt; "
                    "and also a literal <summary>fake nested</summary> inside analysis"
                    "</analysis>\n<summary>real trailing summary</summary>"
                )
            ),
        ]
    )
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )

    assert draft.summary == "real trailing summary"


@pytest.mark.asyncio
async def test_compaction_retries_archive_write_on_transient_failure(tmp_path: Path) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime(
        file_write_results=[
            RuntimeToolResult(stderr="disk pressure", exit_code=1),
            RuntimeToolResult(stderr="disk pressure", exit_code=1),
            RuntimeToolResult(),
        ],
    )
    llm = FakeLlmClient([AssistantText(text="<summary>archive me</summary>")])
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )
    await compactor.commit_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        draft=draft,
    )

    assert [call.path for call in runtime.file_write_calls] == [draft.archive_path] * 3
    summary_rows = [
        record
        for record in await projection.list_all_context_items("c")
        if record.item_kind == "context_summary"
    ]
    assert len(summary_rows) == 1


@pytest.mark.asyncio
async def test_compaction_commits_summary_when_archive_write_persistently_fails(
    tmp_path: Path,
) -> None:
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="first")
    await projection.append_user_message(user_id="u:1", conversation_id="c", text="second")
    runtime = FakeUserRuntime(
        file_write_results=[
            RuntimeError("docker exec failed"),
            RuntimeToolResult(stderr="still failing", exit_code=1),
            RuntimeError("docker exec failed"),
        ],
    )
    llm = FakeLlmClient([AssistantText(text="<summary>archive me</summary>")])
    compactor = ContextCompactor(
        projection=projection,
        runtime=runtime,
        llm=llm,
        config=ContextCompactionConfig(
            max_tokens_per_model=10,
            reserve_tokens=0,
            keep_last_messages=1,
        ),
    )

    snapshot = await compactor.create_snapshot(conversation_id="c")
    assert snapshot is not None
    draft = await compactor.create_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        snapshot=snapshot,
    )
    await compactor.commit_draft(
        user_id="u:1",
        conversation_id="c",
        generation=1,
        draft=draft,
    )

    assert len(runtime.file_write_calls) == 3
    summary_rows = [
        record
        for record in await projection.list_all_context_items("c")
        if record.item_kind == "context_summary"
    ]
    assert len(summary_rows) == 1
    assert summary_rows[0].text == "archive me"


class BlockingToolThenTextLlm(LlmClient):
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[LlmRequest] = []

    async def respond(self, request: LlmRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
            return LlmToolCall(
                call_id="stale",
                name="shell.exec",
                input=ShellExecInput(command="echo stale"),
            )
        return AssistantText(text="latest")


class BlockingSummaryLlm(LlmClient):
    def __init__(self) -> None:
        self.summary_started = asyncio.Event()
        self.release_summary = asyncio.Event()
        self.requests: list[LlmRequest] = []

    async def respond(self, request: LlmRequest):
        self.requests.append(request)
        if request.tools == []:
            self.summary_started.set()
            await self.release_summary.wait()
            return AssistantText(text="<summary>stale summary</summary>")
        return AssistantText(text="latest")


class SequenceEstimator:
    def __init__(self, estimates: list[int]) -> None:
        self._estimates = list(estimates)

    def __call__(self, request: LlmRequest) -> int:
        if not self._estimates:
            raise AssertionError("SequenceEstimator has no queued estimate")
        return self._estimates.pop(0)


def capture_current_reply(
    replies: list[str],
    coordinator: ConversationTurnCoordinator,
):
    async def handler(event: AssistantTextProduced) -> tuple:
        if await coordinator.is_current(event.conversation_id, event.generation):
            replies.append(event.text)
        return ()

    return handler


async def wait_until_generation(
    coordinator: ConversationTurnCoordinator,
    *,
    conversation_id: str,
    generation: int,
) -> None:
    while not await coordinator.is_current(conversation_id, generation):
        await asyncio.sleep(0.01)
