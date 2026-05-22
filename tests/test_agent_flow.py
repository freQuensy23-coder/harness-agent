import asyncio
import json
from pathlib import Path

import pytest

from harness_agent.app import HarnessApp
from harness_agent.bus import EventBus
from harness_agent.compaction import CompactionConfig, CompactionService
from harness_agent.config import (
    DatabaseConfig,
    DockerConfig,
    HarnessConfig,
    LlmConfig,
    RuntimeConfig,
)
from harness_agent.context import AgentFileSet, ContextBuilder, Skill
from harness_agent.events import (
    AgentTurnRequested,
    AgentTurnSuperseded,
    AssistantTextProduced,
    CliTextReceived,
    CompactionRequested,
    CompactionSnapshotReady,
    CompactionSummaryReady,
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
    LlmResponse,
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
from harness_agent.tool_executor import ToolCallExecutor
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
                    "- agent.* runs sub-agents that can use workspace, web, task, schedule, skill, and MCP tools but cannot spawn further sub-agents.",
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
    tool_executor = ToolCallExecutor(runtime=runtime)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
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
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
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
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
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


@pytest.mark.asyncio
async def test_runner_publishes_compaction_requested_when_estimate_over_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = FakeLlmClient([AssistantText(text="ok")])
    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    compaction_config = CompactionConfig(max_tokens_per_model=1_000, reserve_tokens=0)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        compaction_config=compaction_config,
    )

    monkeypatch.setattr(
        "harness_agent.handlers.estimate_request_tokens",
        lambda _request: compaction_config.threshold + 1,
    )

    requested_events: list[CompactionRequested] = []

    async def record_compaction_requested(event: CompactionRequested) -> tuple:
        requested_events.append(event)
        # Synthetic summary so re-read can drop the old user row.
        all_records = await projection.list_all_context_items(event.conversation_id)
        compacted = [r.sequence for r in all_records]
        await projection.append_compacted_context_if_unchanged(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            summary="synthetic",
            archive_path=f"/workspace/.old-sessions/{event.compaction_id}.jsonl",
            compacted_sequences=compacted,
            tail_sequences=[],
            snapshot_max_sequence=max(r.sequence for r in all_records),
        )
        return ()

    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(CompactionRequested, record_compaction_requested)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:compact",
            source="cli",
            text="hello",
        )
    )

    assert len(requested_events) == 1
    requested = requested_events[0]
    assert requested.user_id == "u:1"
    assert requested.conversation_id == "cli:compact"
    assert requested.generation == 1
    assert requested.compaction_id  # uuid hex, non-empty


@pytest.mark.asyncio
async def test_runner_re_reads_messages_after_compaction_chain_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = FakeLlmClient([AssistantText(text="ok")])
    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    compaction_config = CompactionConfig(max_tokens_per_model=1_000, reserve_tokens=0)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        compaction_config=compaction_config,
    )

    monkeypatch.setattr(
        "harness_agent.handlers.estimate_request_tokens",
        lambda _request: compaction_config.threshold + 1,
    )

    async def synthetic_compaction(event: CompactionRequested) -> tuple:
        all_records = await projection.list_all_context_items(event.conversation_id)
        compacted = [r.sequence for r in all_records]
        await projection.append_compacted_context_if_unchanged(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            summary="prior chat condensed",
            archive_path=f"/workspace/.old-sessions/{event.compaction_id}.jsonl",
            compacted_sequences=compacted,
            tail_sequences=[],
            snapshot_max_sequence=max(r.sequence for r in all_records),
        )
        return ()

    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(CompactionRequested, synthetic_compaction)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:reread",
            source="cli",
            text="please summarize this",
        )
    )

    # After the compaction cascade returns, the runner re-reads messages.
    # The LLM call sees only the synthetic summary (the prior user row was
    # compacted into the summary by the synthetic handler).
    assert len(llm.requests) == 1
    assert llm.requests[0].messages == [
        UserMessage(text="Previous conversation summary:\nprior chat condensed"),
    ]


@pytest.mark.asyncio
async def test_runner_does_not_publish_compaction_when_under_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    llm = FakeLlmClient([AssistantText(text="ok")])
    bus = EventBus(store)
    conversation_projector = ConversationProjector(projection)
    compaction_config = CompactionConfig(max_tokens_per_model=1_000, reserve_tokens=0)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        compaction_config=compaction_config,
    )

    monkeypatch.setattr(
        "harness_agent.handlers.estimate_request_tokens",
        lambda _request: compaction_config.threshold - 1,
    )

    requested_events: list[CompactionRequested] = []

    async def record_compaction_requested(event: CompactionRequested) -> tuple:
        requested_events.append(event)
        return ()

    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(CompactionRequested, record_compaction_requested)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:no-compact",
            source="cli",
            text="hello",
        )
    )

    assert requested_events == []
    # The runner still made its single LLM call with the unmodified messages.
    assert len(llm.requests) == 1
    assert llm.requests[0].messages == [UserMessage(text="hello")]


@pytest.mark.asyncio
async def test_supersede_during_compaction_does_not_commit_stale_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )
    runner_llm = FakeLlmClient(
        [AssistantText(text="reply gen 1"), AssistantText(text="reply gen 2")]
    )
    compaction_llm = BlockingSummaryLlm()
    bus = EventBus(store)
    coordinator = ConversationTurnCoordinator()
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    compaction_config = CompactionConfig(max_tokens_per_model=1_000, reserve_tokens=0)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=runner_llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
        compaction_config=compaction_config,
    )
    compaction_service = CompactionService(
        projection=projection,
        llm=compaction_llm,
        config=compaction_config,
    )

    # First turn is over threshold; second turn is well under (so it does
    # not retrigger compaction).
    estimate_responses = iter([compaction_config.threshold + 1, 0])
    monkeypatch.setattr(
        "harness_agent.handlers.estimate_request_tokens",
        lambda _request: next(estimate_responses),
    )

    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(AssistantTextProduced, conversation_projector.handle_assistant_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)
    bus.subscribe(CompactionRequested, compaction_service.handle_requested)
    bus.subscribe(CompactionSnapshotReady, compaction_service.handle_snapshot_ready)
    bus.subscribe(CompactionSummaryReady, compaction_service.handle_summary_ready)

    # Pre-seed conversation history so the boundary algorithm has enough
    # user messages to pick a non-trivial boundary.
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:supersede",
        text="prior-1",
    )
    await projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:supersede",
        generation=0,
        text="prior-a",
    )
    await projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:supersede",
        text="prior-2",
    )

    first = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:supersede",
                source="cli",
                text="old",
            )
        )
    )
    # Wait until the compaction LLM call begins — the cascade is mid-flight.
    await compaction_llm.first_started.wait()

    second = asyncio.create_task(
        bus.publish(
            UserTextReceived(
                user_id="u:1",
                conversation_id="cli:supersede",
                source="cli",
                text="new",
            )
        )
    )
    # Wait until the second turn has bumped generation (via request_generation
    # in handle_user_text) — projector also appended its user row, bumping
    # max(sequence) so the in-flight CAS will fail.
    await wait_until_generation(
        coordinator, conversation_id="cli:supersede", generation=2
    )
    compaction_llm.release_first.set()
    await first
    await second

    events = await store.list_events()
    event_types = [event.type for event in events]
    # The cascade emitted CompactionConflicted (CAS lost), not Committed.
    assert "compaction.conflicted" in event_types
    assert "compaction.committed" not in event_types

    # No context_summary row was written.
    rows = await projection.list_all_context_items("cli:supersede")
    assert all(r.item_kind != "context_summary" for r in rows)

    # The second turn's LLM request saw the full uncompacted history.
    # runner_llm.requests[0] was the first (stale) turn; [1] is gen 2.
    assert len(runner_llm.requests) >= 2
    second_turn_request = runner_llm.requests[-1]
    assert second_turn_request.messages == [
        UserMessage(text="prior-1"),
        AssistantMessage(text="prior-a"),
        UserMessage(text="prior-2"),
        UserMessage(text="old"),
        UserMessage(text="new"),
    ]


class BlockingSummaryLlm(LlmClient):
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.requests: list[LlmRequest] = []

    async def respond(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
        return AssistantText(text="<summary>stale summary</summary>")


@pytest.mark.asyncio
async def test_e2e_compaction_cascade_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLlmClient(
        [
            AssistantText(text="<summary>condensed earlier turns</summary>"),
            AssistantText(text="hello after compaction"),
        ]
    )
    fake_runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="SOUL", agents="AGENTS", user="USER", tools="TOOLS"),
    )

    class _StubDockerRuntime(FakeUserRuntime):
        def __init__(self, **_kwargs: object) -> None:
            super().__init__()

        # Delegate every operation to the shared fake_runtime so tests can
        # inspect a single instance.
        async def read_agent_files(self, user_id):  # type: ignore[override]
            return await fake_runtime.read_agent_files(user_id)

        async def list_skills(self, user_id):  # type: ignore[override]
            return await fake_runtime.list_skills(user_id)

        async def list_mcp_servers(self, user_id):  # type: ignore[override]
            return await fake_runtime.list_mcp_servers(user_id)

        async def write_content_file(self, user_id, path, content):  # type: ignore[override]
            return await fake_runtime.write_content_file(user_id, path, content)

        async def read_file_bytes(self, user_id, path, max_bytes):  # type: ignore[override]
            return await fake_runtime.read_file_bytes(user_id, path, max_bytes)

        async def shell_exec(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.shell_exec(user_id, input)

        async def shell_spawn(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.shell_spawn(user_id, input)

        async def shell_read(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.shell_read(user_id, input)

        async def shell_kill(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.shell_kill(user_id, input)

        async def file_read(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_read(user_id, input)

        async def file_write(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_write(user_id, input)

        async def file_edit(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_edit(user_id, input)

        async def file_multi_edit(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_multi_edit(user_id, input)

        async def file_glob(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_glob(user_id, input)

        async def file_grep(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_grep(user_id, input)

        async def file_list(self, user_id, input):  # type: ignore[override]
            return await fake_runtime.file_list(user_id, input)

    class _StubOpenAIClient(LlmClient):
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def respond(self, request: LlmRequest) -> LlmResponse:
            return await fake_llm.respond(request)

    monkeypatch.setattr("harness_agent.app.DockerUserRuntime", _StubDockerRuntime)
    monkeypatch.setattr("harness_agent.app.OpenAIResponsesClient", _StubOpenAIClient)
    # Force the runner to estimate over threshold on its first turn so the
    # compaction cascade is exercised end-to-end through HarnessApp wiring.
    threshold_holder: dict[str, int] = {}
    real_estimate = "harness_agent.handlers.estimate_request_tokens"

    def _over_threshold(_request: LlmRequest) -> int:
        return threshold_holder["value"] + 1

    monkeypatch.setattr(real_estimate, _over_threshold)

    config = HarnessConfig(
        database=DatabaseConfig(path=tmp_path / "harness.sqlite3"),
        llm=LlmConfig(
            base_url="http://example.invalid",
            api_key="not-used",
            model="fake-model",
            max_tokens_per_model=1_000,
            compaction_reserve_tokens=0,
            compaction_keep_last_user_messages=2,
        ),
        runtime=RuntimeConfig(docker=DockerConfig()),
    )

    app = HarnessApp(config=config)
    threshold_holder["value"] = app._config.llm.max_tokens_per_model - app._config.llm.compaction_reserve_tokens  # noqa: SLF001

    # Pre-seed prior conversation history so the boundary algorithm
    # (keep_last_user_messages=2) has room to pick a real boundary.
    await app.projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:e2e",
        text="prior-1",
    )
    await app.projection.append_assistant_message(
        user_id="u:1",
        conversation_id="cli:e2e",
        generation=0,
        text="prior-a",
    )
    await app.projection.append_user_message(
        user_id="u:1",
        conversation_id="cli:e2e",
        text="prior-2",
    )

    await app.bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:e2e",
            source="cli",
            text="new question",
        )
    )

    events = await app.event_store.list_events()
    event_types = [event.type for event in events]
    assert event_types == [
        "user.text.received",
        "agent.turn.requested",
        "compaction.requested",
        "compaction.snapshot.ready",
        "compaction.summary.ready",
        "compaction.committed",
        "agent.generation.started",
        "assistant.text.produced",
    ]

    compaction_committed_index = event_types.index("compaction.committed")
    assistant_index = event_types.index("assistant.text.produced")
    assert compaction_committed_index < assistant_index

    # The CompactionCommitted event names the archive path, and the
    # CompactionArchiveHandler must have invoked file_write with the same path
    # before the next turn produced AssistantTextProduced.
    committed_event = events[compaction_committed_index]
    assert committed_event.type == "compaction.committed"
    archive_path = committed_event.archive_path  # type: ignore[union-attr]
    assert archive_path.startswith("/workspace/.old-sessions/")
    assert fake_runtime.file_write_calls, "archive handler did not call file_write"
    written_paths = [call.path for call in fake_runtime.file_write_calls]
    assert archive_path in written_paths

    final_text = events[assistant_index].text  # type: ignore[union-attr]
    assert final_text == "hello after compaction"

    # Two LLM calls: one for the compaction summary, one for the assistant reply.
    assert len(fake_llm.requests) == 2
