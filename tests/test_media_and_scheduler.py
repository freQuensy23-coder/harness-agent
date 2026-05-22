import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.content import ContentRef
from harness_agent.context import AgentFileSet, ContextBuilder
from harness_agent.events import (
    AgentTurnRequested,
    InboundAttachment,
    ScheduledMessageDue,
    TelegramReplyTarget,
    TelegramTextReceived,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import (
    AgentTurnHandler,
    ContentIngestionHandler,
    ConversationProjector,
    IdentityHandler,
)
from harness_agent.identity import StaticIdentityResolver
from harness_agent.llm import (
    AssistantText,
    FakeLlmClient,
    LlmToolCall,
    ToolResultMessage,
    UserMessage,
    message_to_openai,
)
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime
from harness_agent.scheduler import SchedulerDueHandler, SchedulerPump, SQLiteScheduleStore
from harness_agent.store import SQLiteEventStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor
from harness_agent.tools import (
    FileReadInput,
    ScheduleCronInput,
    ScheduleOnceInput,
    default_tool_registry,
)


def test_runtime_layer_has_no_llm_context_import() -> None:
    runtime_package = Path("src/harness_agent/runtime")
    for module in sorted(runtime_package.glob("*.py")):
        assert "harness_agent.llm" not in module.read_text(encoding="utf-8"), module


@pytest.mark.asyncio
async def test_telegram_image_is_saved_and_passed_to_llm_context(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    image_bytes = b"image-bytes"
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient([AssistantText(text="seen")])
    bus = EventBus(store)
    identity_handler = IdentityHandler(StaticIdentityResolver())
    content_handler = ContentIngestionHandler(runtime=runtime)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
    )
    bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)
    bus.subscribe(UserTextReceived, content_handler.handle_user_text)
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        TelegramTextReceived(
            telegram_user_id=123,
            telegram_chat_id=456,
            telegram_message_id=777,
            text="what is this?",
            attachments=[
                InboundAttachment(
                    kind="image",
                    file_name="photo.jpg",
                    mime_type="image/jpeg",
                    size_bytes=len(image_bytes),
                    workspace_path="/workspace/content/777/photo.jpg",
                    content_base64=image_base64,
                    source_id="telegram-file-id",
                )
            ],
        )
    )

    expected_attachment = ContentRef(
        kind="image",
        file_name="photo.jpg",
        mime_type="image/jpeg",
        size_bytes=len(image_bytes),
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        workspace_path="/workspace/content/777/photo.jpg",
        content_base64=image_base64,
    )
    assert runtime.content_write_calls == [
        ("/workspace/content/777/photo.jpg", image_bytes)
    ]
    assert llm.requests[0].messages == [
        UserMessage(text="what is this?", attachments=[expected_attachment])
    ]
    assert await projection.list_llm_messages("tg:456") == [
        UserMessage(text="what is this?", attachments=[expected_attachment])
    ]

    openai_message = message_to_openai(llm.requests[0].messages[0])
    assert openai_message["role"] == "user"
    assert openai_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
    }


@pytest.mark.asyncio
async def test_telegram_file_is_saved_and_referenced_without_file_bytes_in_llm(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    file_bytes = b"plain document"
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient([AssistantText(text="ok")])
    bus = EventBus(store)
    identity_handler = IdentityHandler(StaticIdentityResolver())
    content_handler = ContentIngestionHandler(runtime=runtime)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
    )
    bus.subscribe(TelegramTextReceived, identity_handler.handle_telegram_text)
    bus.subscribe(UserTextReceived, content_handler.handle_user_text)
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        TelegramTextReceived(
            telegram_user_id=123,
            telegram_chat_id=456,
            telegram_message_id=778,
            text="read this file",
            attachments=[
                InboundAttachment(
                    kind="file",
                    file_name="doc.txt",
                    mime_type="text/plain",
                    size_bytes=len(file_bytes),
                    workspace_path="/workspace/content/778/doc.txt",
                    content_base64=base64.b64encode(file_bytes).decode("ascii"),
                    source_id="telegram-doc-id",
                )
            ],
        )
    )

    assert runtime.content_write_calls == [
        ("/workspace/content/778/doc.txt", file_bytes)
    ]
    message = llm.requests[0].messages[0]
    assert message.attachments[0].content_base64 is None
    openai_payload = message_to_openai(message)
    assert json.dumps(openai_payload)
    assert "/workspace/content/778/doc.txt" in openai_payload["content"]


@pytest.mark.asyncio
async def test_file_read_on_image_injects_image_into_next_llm_context(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    image_bytes = b"\x89PNG\r\n\x1a\n" + (b"image-bytes" * 3_000)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
        files={"/workspace/content/picture.png": image_bytes.decode("latin1")},
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="open-image",
                name="file.read",
                input=FileReadInput(path="/workspace/content/picture.png"),
            ),
            AssistantText(text="image seen"),
        ]
    )
    bus = EventBus(store)
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:image-open",
            source="cli",
            text="open /workspace/content/picture.png",
        )
    )

    expected_attachment = ContentRef(
        kind="image",
        file_name="picture.png",
        mime_type="image/png",
        size_bytes=len(image_bytes),
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        workspace_path="/workspace/content/picture.png",
        content_base64=image_base64,
    )
    assert llm.requests[1].messages[-1] == UserMessage(
        text="Opened image file /workspace/content/picture.png",
        attachments=[expected_attachment],
    )
    completed_events = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ]
    assert completed_events[0].attachments == [expected_attachment]
    assert await projection.list_llm_messages("cli:image-open") == [
        UserMessage(text="open /workspace/content/picture.png"),
        llm.requests[1].messages[-3],
        llm.requests[1].messages[-2],
        UserMessage(
            text="Opened image file /workspace/content/picture.png",
            attachments=[expected_attachment],
        ),
    ]
    openai_message = message_to_openai(llm.requests[1].messages[-1])
    assert openai_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    }


@pytest.mark.asyncio
async def test_file_read_missing_file_returns_tool_result(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="missing",
                name="file.read",
                input=FileReadInput(path="/workspace/missing.png"),
            ),
            AssistantText(text="reported"),
        ]
    )
    bus = EventBus(store)
    tool_executor = ToolCallExecutor(runtime=runtime)
    conversation_projector = ConversationProjector(projection)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, conversation_projector.handle_user_text)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:missing-file",
            source="cli",
            text="open missing image",
        )
    )

    completed_events = [
        event for event in await store.list_events() if event.type == "tool.call.completed"
    ]
    assert completed_events[0].result.exit_code == 1
    assert completed_events[0].result.stderr == "No such file: /workspace/missing.png\n"
    assert completed_events[0].attachments == []
    assert llm.requests[1].messages[-1] == ToolResultMessage(
        call_id="missing",
        name="file.read",
        content=(
            "file.read stdout:\n"
            "stderr:\nNo such file: /workspace/missing.png\n\n"
            "exit_code: 1"
        ),
    )


@pytest.mark.asyncio
async def test_agent_can_create_delayed_and_cron_schedules_via_tools(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    schedule_store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now)
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    runtime = FakeUserRuntime(
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
    )
    llm = FakeLlmClient(
        [
            LlmToolCall(
                call_id="once",
                name="schedule.once",
                input=ScheduleOnceInput(
                    message="[scheduled] check it now",
                    delay_seconds=300,
                ),
            ),
            LlmToolCall(
                call_id="cron",
                name="schedule.cron",
                input=ScheduleCronInput(
                    message="[cron auto message] Pls get weather in Paris from api via curl and send summary",
                    cron="0 9 * * *",
                    timezone="Europe/Paris",
                ),
            ),
            AssistantText(text="scheduled"),
        ]
    )
    bus = EventBus(store)
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        task_store=task_store,
        schedule_store=schedule_store,
    )
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        tool_executor=tool_executor,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection).handle_user_text)
    bus.subscribe(UserTextReceived, agent_turn_handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, agent_turn_handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="tg:456",
            source="telegram",
            text="setup schedules",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    schedules = await schedule_store.list_for_conversation(
        user_id="u:1",
        conversation_id="tg:456",
    )
    assert [(schedule.kind, schedule.message) for schedule in schedules] == [
        ("once", "[scheduled] check it now"),
        ("cron", "[cron auto message] Pls get weather in Paris from api via curl and send summary"),
    ]
    assert schedules[0].next_run_at == now + timedelta(seconds=300)
    assert schedules[1].cron == "0 9 * * *"
    assert schedules[1].timezone == "Europe/Paris"


@pytest.mark.asyncio
async def test_scheduler_due_event_becomes_fake_user_message(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    schedule_store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now)
    schedule = await schedule_store.create_once(
        user_id="u:1",
        conversation_id="tg:456",
        message="[scheduled] check it now",
        reply_target=TelegramReplyTarget(chat_id=456),
        delay_seconds=0,
    )
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(store)
    captured: list[UserTextReceived] = []

    async def capture_user_text(event: UserTextReceived) -> tuple:
        captured.append(event)
        return ()

    due_handler = SchedulerDueHandler()
    bus.subscribe(ScheduledMessageDue, due_handler.handle_due)
    bus.subscribe(UserTextReceived, capture_user_text)

    pump = SchedulerPump(
        store=schedule_store,
        bus=bus,
        now=lambda: now + timedelta(seconds=1),
    )
    await pump.tick()

    assert [(event.source, event.text, event.reply_target) for event in captured] == [
        ("scheduler", "[scheduled] check it now", TelegramReplyTarget(chat_id=456))
    ]
    refreshed = await schedule_store.get(schedule.id)
    assert refreshed.status == "completed"
