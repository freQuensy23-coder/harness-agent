import asyncio
import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

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
from harness_agent.memory_service import MemoryService
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
from harness_agent.scheduler import (
    SchedulerDueHandler,
    SchedulerPump,
    SchedulerService,
    SQLiteScheduleStore,
)
from harness_agent.store import SQLiteEventStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.session_search_service import SessionSearchService
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    FileReadInput,
    ScheduleCronInput,
    ScheduleOnceInput,
    default_tool_registry,
)


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
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    identity_handler = IdentityHandler(StaticIdentityResolver())
    content_handler = ContentIngestionHandler(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
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
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    identity_handler = IdentityHandler(StaticIdentityResolver())
    content_handler = ContentIngestionHandler(runtime=runtime)
    conversation_projector = ConversationProjector(projection, turn_coordinator=coordinator)
    agent_turn_handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=default_tool_registry(),
        projection=projection,
        turn_coordinator=coordinator,
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
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=runtime)
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
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, agent_turn_handler.handle_tool_call_completed)
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
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(runtime=runtime)
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
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, conversation_projector.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, agent_turn_handler.handle_tool_call_completed)
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
    coordinator = ConversationTurnCoordinator()
    tool_results = ToolCallResultWaiter()
    tool_executor = tool_executor_for_test(
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
        turn_coordinator=coordinator,
    )
    bus.subscribe(UserTextReceived, ConversationProjector(projection, turn_coordinator=coordinator).handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection, turn_coordinator=coordinator).handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, agent_turn_handler.handle_tool_call_completed)
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


async def test_claim_due_cron_advances_next_run_from_stored_metadata(tmp_path: Path) -> None:
    now = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    schedule_store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now)
    schedule = await schedule_store.create_cron(
        user_id="u:1",
        conversation_id="tg:456",
        message="[cron] tick",
        reply_target=TelegramReplyTarget(chat_id=456),
        cron="* * * * *",
        timezone="UTC",
    )

    claimed = await schedule_store.claim_due(datetime(2026, 5, 19, 12, 1, tzinfo=UTC))

    assert len(claimed) == 1
    assert claimed[0].id == schedule.id
    assert claimed[0].kind == "cron"
    assert claimed[0].cron == "* * * * *"
    assert claimed[0].timezone == "UTC"
    refreshed = await schedule_store.get(schedule.id)
    assert refreshed.status == "active"
    assert refreshed.next_run_at == datetime(2026, 5, 19, 12, 2, tzinfo=UTC)


@pytest.mark.asyncio
async def test_schedule_cancel_rejects_wrong_user_and_conversation(tmp_path: Path) -> None:
    """Before the fix, cancel() scoped the UPDATE by user/conversation
    but then `get(schedule_id)` read by id only — a wrong-scope caller
    saw `status='active'` returned as if the cancel had succeeded, and
    the real schedule kept firing."""
    now = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now)
    schedule = await store.create_once(
        user_id="u:owner",
        conversation_id="tg:owner",
        message="ping",
        reply_target=TelegramReplyTarget(chat_id=1),
        delay_seconds=60,
    )

    # Wrong user.
    with pytest.raises(KeyError):
        await store.cancel(
            schedule_id=schedule.id,
            user_id="u:intruder",
            conversation_id="tg:owner",
        )

    # Wrong conversation under the right user.
    with pytest.raises(KeyError):
        await store.cancel(
            schedule_id=schedule.id,
            user_id="u:owner",
            conversation_id="tg:other",
        )

    # Schedule is still active after both rejected attempts.
    untouched = await store.get(schedule.id)
    assert untouched.status == "active"

    # The legitimate owner can still cancel under matching scope.
    cancelled = await store.cancel(
        schedule_id=schedule.id,
        user_id="u:owner",
        conversation_id="tg:owner",
    )
    assert cancelled.status == "cancelled"


class CountingPump:
    def __init__(self) -> None:
        self.ticks = 0
        self._tick_event = asyncio.Event()

    async def tick(self) -> None:
        self.ticks += 1
        self._tick_event.set()
        await asyncio.sleep(0)

    async def wait_for_tick_after(self, previous_ticks: int) -> None:
        while self.ticks <= previous_ticks:
            self._tick_event.clear()
            await asyncio.wait_for(self._tick_event.wait(), timeout=1)


@pytest.mark.asyncio
async def test_scheduler_due_survives_crash_between_state_advance_and_publish(
    tmp_path: Path,
) -> None:
    """Before the outbox fix, claim_due committed the state advance to
    SQLite and then SchedulerPump.tick() published ScheduledMessageDue
    in a separate step. A crash between those two steps marked the
    schedule fired in the database while no event ever reached the
    user. The fix moves the would-be due event into a transactional
    outbox so the next tick can re-publish it."""
    now_value = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now_value)
    await store.create_once(
        user_id="u:1",
        conversation_id="tg:1",
        message="take your pill",
        reply_target=TelegramReplyTarget(chat_id=1),
        delay_seconds=0,
    )

    # First "process": claim succeeds, publish raises (simulated crash).
    crashing_bus: list[ScheduledMessageDue] = []

    class _CrashingBus:
        async def publish(self, event: ScheduledMessageDue) -> None:
            crashing_bus.append(event)
            raise RuntimeError("simulated crash mid-publish")

    pump_crashing = SchedulerPump(
        store=store,
        bus=_CrashingBus(),  # type: ignore[arg-type]
        now=lambda: now_value + timedelta(seconds=1),
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        await pump_crashing.tick()

    # Schedule state was advanced (the "data loss" pre-fix would stop here).
    schedules = await store.list_for_conversation(user_id="u:1", conversation_id="tg:1")
    assert len(schedules) == 1
    assert schedules[0].status == "completed"
    # Outbox row survives the crash — recovery is possible.
    pending = await store.list_pending_due()
    assert len(pending) == 1
    assert pending[0].text == "take your pill"

    # Second "process": fresh bus replays the outbox.
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    real_bus = EventBus(event_store)
    captured: list[UserTextReceived] = []

    async def capture(event: UserTextReceived) -> tuple:
        captured.append(event)
        return ()

    bus_handler = SchedulerDueHandler()
    real_bus.subscribe(ScheduledMessageDue, bus_handler.handle_due)
    real_bus.subscribe(UserTextReceived, capture)

    pump_recovered = SchedulerPump(
        store=store,
        bus=real_bus,
        now=lambda: now_value + timedelta(seconds=10),
    )
    await pump_recovered.tick()

    assert [(e.source, e.text) for e in captured] == [("scheduler", "take your pill")]
    # Outbox drained — no more pending entries.
    assert await store.list_pending_due() == []


@pytest.mark.asyncio
async def test_scheduler_outbox_replay_is_idempotent_on_event_store(
    tmp_path: Path,
) -> None:
    """If a previous attempt published the event but crashed before
    deleting the outbox row, re-publishing must NOT produce a duplicate
    ScheduledMessageDue in the event store (the event id is deterministic
    from the outbox row, so the unique constraint in events.id catches
    it and tick() proceeds to delete the outbox row idempotently)."""
    now_value = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)
    store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3", now=lambda: now_value)
    await store.create_once(
        user_id="u:1",
        conversation_id="tg:1",
        message="ping",
        reply_target=TelegramReplyTarget(chat_id=1),
        delay_seconds=0,
    )
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    real_bus = EventBus(event_store)
    pump = SchedulerPump(
        store=store,
        bus=real_bus,
        now=lambda: now_value + timedelta(seconds=1),
    )

    await pump.tick()
    # Manually re-introduce the outbox entry to simulate "published but not
    # marked-published" — easiest way to force the duplicate path.
    pending_after_first = await store.list_pending_due()
    assert pending_after_first == []
    events_after_first = [e for e in await event_store.list_events() if e.type == "scheduled.message.due"]
    assert len(events_after_first) == 1
    replayed_event = events_after_first[0]

    # Re-insert the same outbox row using the same id; next tick will try
    # to publish a ScheduledMessageDue with the same id again.
    import aiosqlite
    async with aiosqlite.connect(tmp_path / "schedules.sqlite3") as db:
        await db.execute(
            """
            insert into scheduled_due_outbox(
                id, schedule_id, user_id, conversation_id, text, reply_target_json, created_at
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replayed_event.id,
                "(any)",
                "u:1",
                "tg:1",
                "ping",
                None,
                now_value.isoformat(),
            ),
        )
        await db.commit()

    await pump.tick()

    # Still exactly one ScheduledMessageDue in the event store.
    events_after_replay = [e for e in await event_store.list_events() if e.type == "scheduled.message.due"]
    assert len(events_after_replay) == 1
    # Outbox drained.
    assert await store.list_pending_due() == []


@pytest.mark.asyncio
async def test_scheduler_service_duplicate_start_stop_and_restart() -> None:
    pump = CountingPump()
    service = SchedulerService(pump=cast(SchedulerPump, pump), poll_seconds=0.01)

    await service.start()
    await pump.wait_for_tick_after(0)
    first_task = service._task
    await service.start()

    assert service._task is first_task

    await service.stop()
    stopped_at = pump.ticks
    assert service._task is None
    await asyncio.sleep(0.03)
    assert pump.ticks == stopped_at

    await service.start()
    await pump.wait_for_tick_after(stopped_at)
    assert service._task is not None
    assert service._task is not first_task
    await service.stop()
