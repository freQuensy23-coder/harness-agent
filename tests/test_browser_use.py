import asyncio
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest

from harness_agent.browser_use import (
    BrowserSessionPollHandler,
    BrowserSessionPump,
    BrowserSessionResultWaiter,
    BrowserUseService,
    CloudMessage,
    CloudMessagesPage,
    CloudProfile,
    CloudSessionState,
    SQLiteBrowserProfileStore,
    SQLiteBrowserSessionStore,
)
from harness_agent.bus import EventBus
from harness_agent.events import (
    BrowserProfileCreated,
    BrowserProfileEvicted,
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionMessageReceived,
    BrowserSessionPollDue,
    BrowserSessionStarted,
    BrowserSessionStopped,
)
from harness_agent.runtime import FakeUserRuntime
from harness_agent.store import SQLiteEventStore
from harness_agent.tool_executor import ToolCallExecutor
from harness_agent.tools import (
    BrowserGetInput,
    BrowserListInput,
    BrowserRunInput,
    BrowserSendInput,
    BrowserSpawnInput,
    BrowserStopInput,
)
from harness_agent.events import ToolCallRequested


class FakeCloudClient:
    def __init__(
        self,
        *,
        session_status_sequence: list[CloudSessionState] | None = None,
    ) -> None:
        self.profiles: dict[str, CloudProfile] = {}
        self.deleted_profiles: list[str] = []
        self.sessions: dict[str, CloudSessionState] = {}
        self.send_calls: list[tuple[str, str]] = []
        self.stop_calls: list[tuple[str, str]] = []
        self.create_session_calls: list[dict] = []
        self._status_sequence = session_status_sequence or []
        self._status_index: dict[str, int] = {}
        self.messages_by_session: dict[str, list[CloudMessage]] = {}

    async def create_profile(self, *, internal_user_id: str) -> CloudProfile:
        cloud_id = f"prof_{uuid4().hex[:8]}"
        profile = CloudProfile(id=cloud_id, user_id=internal_user_id, name=f"harness-{internal_user_id}")
        self.profiles[cloud_id] = profile
        return profile

    async def delete_profile(self, *, cloud_profile_id: str) -> None:
        self.deleted_profiles.append(cloud_profile_id)
        self.profiles.pop(cloud_profile_id, None)

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState:
        cloud_id = f"sess_{uuid4().hex[:8]}"
        state = CloudSessionState(
            id=cloud_id,
            status="created",
            step_count=0,
            output=None,
            live_url=f"https://live/{cloud_id}",
            profile_id=cloud_profile_id,
        )
        self.sessions[cloud_id] = state
        self.create_session_calls.append({
            "task": task,
            "cloud_profile_id": cloud_profile_id,
            "model": model,
            "keep_alive": keep_alive,
        })
        return state

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState:
        self.send_calls.append((cloud_session_id, task))
        existing = self.sessions[cloud_session_id]
        updated = existing.model_copy(update={"status": "running"})
        self.sessions[cloud_session_id] = updated
        return updated

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState:
        if self._status_sequence:
            idx = self._status_index.get(cloud_session_id, 0)
            if idx < len(self._status_sequence):
                step = self._status_sequence[idx].model_copy(update={"id": cloud_session_id})
                self._status_index[cloud_session_id] = idx + 1
                self.sessions[cloud_session_id] = step
                return step
        return self.sessions[cloud_session_id]

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState:
        self.stop_calls.append((cloud_session_id, strategy))
        existing = self.sessions[cloud_session_id]
        updated = existing.model_copy(update={"status": "stopped"})
        self.sessions[cloud_session_id] = updated
        return updated

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage:
        messages = self.messages_by_session.get(cloud_session_id, [])
        if after is None:
            return CloudMessagesPage(messages=list(messages), has_more=False)
        for idx, m in enumerate(messages):
            if m.id == after:
                return CloudMessagesPage(messages=list(messages[idx + 1:]), has_more=False)
        return CloudMessagesPage(messages=[], has_more=False)


def _build(
    tmp_path: Path,
    *,
    profile_cap: int = 5,
    status_sequence: list[CloudSessionState] | None = None,
):
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    profile_store = SQLiteBrowserProfileStore(tmp_path / "profiles.sqlite3")
    session_store = SQLiteBrowserSessionStore(tmp_path / "sessions.sqlite3")
    bus = EventBus(event_store)
    published: list = []

    async def record(event):
        published.append(event)
        return ()

    for event_type in (
        BrowserProfileCreated,
        BrowserProfileEvicted,
        BrowserSessionStarted,
        BrowserSessionPollDue,
        BrowserSessionMessageReceived,
        BrowserSessionCompleted,
        BrowserSessionFailed,
        BrowserSessionStopped,
    ):
        bus.subscribe(event_type, record)

    client = FakeCloudClient(session_status_sequence=status_sequence)
    waiter = BrowserSessionResultWaiter()
    bus.subscribe(BrowserSessionCompleted, waiter.handle_completed)
    bus.subscribe(BrowserSessionFailed, waiter.handle_failed)
    bus.subscribe(BrowserSessionStopped, waiter.handle_stopped)

    handler = BrowserSessionPollHandler(
        client=client,
        session_store=session_store,
    )
    bus.subscribe(BrowserSessionPollDue, handler.handle_poll_due)

    service = BrowserUseService(
        bus=bus,
        client=client,
        profile_store=profile_store,
        session_store=session_store,
        result_waiter=waiter,
        profile_cap=profile_cap,
        default_model="bu-mini",
        default_run_timeout_seconds=5.0,
    )
    pump = BrowserSessionPump(session_store=session_store, bus=bus)
    return service, client, waiter, handler, pump, session_store, published


@pytest.mark.asyncio
async def test_profile_created_on_first_use_and_reused(tmp_path: Path) -> None:
    service, client, _, _, _, _, published = _build(tmp_path)
    first = await service.ensure_profile(user_id="alice")
    second = await service.ensure_profile(user_id="alice")
    assert first.cloud_profile_id == second.cloud_profile_id
    assert len(client.profiles) == 1
    created = [e for e in published if isinstance(e, BrowserProfileCreated)]
    assert len(created) == 1
    assert created[0].user_id == "alice"


@pytest.mark.asyncio
async def test_lru_eviction_at_cap(tmp_path: Path) -> None:
    service, client, _, _, _, _, published = _build(tmp_path, profile_cap=2)
    await service.ensure_profile(user_id="alice")
    await asyncio.sleep(0.01)
    await service.ensure_profile(user_id="bob")
    await asyncio.sleep(0.01)
    await service.ensure_profile(user_id="alice")
    await asyncio.sleep(0.01)
    await service.ensure_profile(user_id="carol")
    evicted = [e for e in published if isinstance(e, BrowserProfileEvicted)]
    assert len(evicted) == 1
    assert evicted[0].evicted_user_id == "bob"
    assert evicted[0].requested_by_user_id == "carol"


@pytest.mark.asyncio
async def test_eviction_skips_users_with_live_sessions(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, _, _, published = _build(tmp_path, profile_cap=2, status_sequence=sequence)
    alice = await service.spawn(
        user_id="alice",
        conversation_id="ca",
        generation=1,
        parent_call_id="x",
        input=BrowserSpawnInput(task="t1"),
    )
    bob = await service.spawn(
        user_id="bob",
        conversation_id="cb",
        generation=1,
        parent_call_id="y",
        input=BrowserSpawnInput(task="t2"),
    )

    with pytest.raises(RuntimeError, match="cap reached"):
        await service.ensure_profile(user_id="carol")

    assert [e for e in published if isinstance(e, BrowserProfileEvicted)] == []

    await service.stop(user_id="alice", input=BrowserStopInput(session_id=alice.session_id))
    await service.stop(user_id="bob", input=BrowserStopInput(session_id=bob.session_id))


@pytest.mark.asyncio
async def test_user_isolation(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, _, _, _ = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="ca",
        generation=1,
        parent_call_id="x",
        input=BrowserSpawnInput(task="t"),
    )
    with pytest.raises(KeyError):
        await service.get(user_id="bob", input=BrowserGetInput(session_id=record.session_id))
    with pytest.raises(KeyError):
        await service.send(
            user_id="bob",
            input=BrowserSendInput(session_id=record.session_id, task="hi"),
        )
    with pytest.raises(KeyError):
        await service.stop(user_id="bob", input=BrowserStopInput(session_id=record.session_id))

    mine, _ = await service.get(user_id="alice", input=BrowserGetInput(session_id=record.session_id))
    assert mine.user_id == "alice"
    await service.stop(user_id="alice", input=BrowserStopInput(session_id=record.session_id))


async def _run_with_ticker(pump, coro, *, timeout: float = 3.0, tick_interval: float = 0.005):
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            await pump.tick()
            await asyncio.sleep(tick_interval)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        stop.set()
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_run_completes_via_pump_tick(tmp_path: Path) -> None:
    sequence = [
        CloudSessionState(id="x", status="running", step_count=1),
        CloudSessionState(id="x", status="idle", step_count=2, output="navigated to example.com"),
    ]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await _run_with_ticker(
        pump,
        service.run(
            user_id="alice",
            conversation_id="c1",
            generation=1,
            parent_call_id="p1",
            input=BrowserRunInput(task="navigate", timeout_seconds=10.0),
        ),
    )
    assert record.status == "completed"
    assert record.output == "navigated to example.com"
    completions = [e for e in published if isinstance(e, BrowserSessionCompleted)]
    assert len(completions) == 1
    assert completions[0].output == "navigated to example.com"


@pytest.mark.asyncio
async def test_run_fails_via_pump_tick(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="error", step_count=0)]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await _run_with_ticker(
        pump,
        service.run(
            user_id="alice",
            conversation_id="c1",
            generation=1,
            parent_call_id="p1",
            input=BrowserRunInput(task="navigate", timeout_seconds=10.0),
        ),
    )
    assert record.status == "error"
    failures = [e for e in published if isinstance(e, BrowserSessionFailed)]
    assert len(failures) == 1
    assert failures[0].status == "error"


@pytest.mark.asyncio
async def test_run_times_out_publishes_failed(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, client, _, _, _, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.run(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserRunInput(task="navigate", timeout_seconds=0.1),
    )
    assert record.status == "timed_out"
    assert any(c[1] == "session" for c in client.stop_calls)
    failures = [e for e in published if isinstance(e, BrowserSessionFailed) and e.status == "timed_out"]
    assert len(failures) == 1


@pytest.mark.asyncio
async def test_send_then_complete_second_task_via_pump(tmp_path: Path) -> None:
    sequence = [
        CloudSessionState(id="x", status="running", step_count=1),
        CloudSessionState(id="x", status="idle", step_count=2, output="first task done"),
        CloudSessionState(id="x", status="running", step_count=3),
        CloudSessionState(id="x", status="idle", step_count=4, output="second task done"),
    ]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserSpawnInput(task="first"),
    )
    await pump.tick()
    await pump.tick()
    first = [e for e in published if isinstance(e, BrowserSessionCompleted)]
    assert len(first) == 1
    assert first[0].output == "first task done"

    await service.send(
        user_id="alice",
        input=BrowserSendInput(session_id=record.session_id, task="second"),
    )
    await pump.tick()
    await pump.tick()
    completions = [e for e in published if isinstance(e, BrowserSessionCompleted)]
    assert len(completions) == 2
    assert completions[1].output == "second task done"


@pytest.mark.asyncio
async def test_pump_publishes_poll_due_per_active_session(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    a = await service.spawn(
        user_id="alice",
        conversation_id="ca",
        generation=1,
        parent_call_id="x",
        input=BrowserSpawnInput(task="t1"),
    )
    b = await service.spawn(
        user_id="bob",
        conversation_id="cb",
        generation=1,
        parent_call_id="y",
        input=BrowserSpawnInput(task="t2"),
    )
    published.clear()
    await pump.tick()
    polls = [e for e in published if isinstance(e, BrowserSessionPollDue)]
    poll_ids = {p.session_id for p in polls}
    assert poll_ids == {a.session_id, b.session_id}
    await service.stop(user_id="alice", input=BrowserStopInput(session_id=a.session_id))
    await service.stop(user_id="bob", input=BrowserStopInput(session_id=b.session_id))


@pytest.mark.asyncio
async def test_pump_skips_terminal_sessions(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="error", step_count=0)]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await _run_with_ticker(
        pump,
        service.run(
            user_id="alice",
            conversation_id="c1",
            generation=1,
            parent_call_id="p1",
            input=BrowserRunInput(task="t", timeout_seconds=10.0),
        ),
    )
    assert record.status == "error"
    published.clear()
    await pump.tick()
    polls = [e for e in published if isinstance(e, BrowserSessionPollDue)]
    assert polls == []


@pytest.mark.asyncio
async def test_send_requires_keep_alive_session(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="idle", step_count=1, output="done")]
    service, _, _, _, pump, _, _ = _build(tmp_path, status_sequence=sequence)
    record = await _run_with_ticker(
        pump,
        service.run(
            user_id="alice",
            conversation_id="c1",
            generation=1,
            parent_call_id="p1",
            input=BrowserRunInput(task="t", timeout_seconds=10.0),
        ),
    )
    with pytest.raises(ValueError, match="keep-alive"):
        await service.send(
            user_id="alice",
            input=BrowserSendInput(session_id=record.session_id, task="hi"),
        )


@pytest.mark.asyncio
async def test_messages_drained_and_emitted_through_bus(tmp_path: Path) -> None:
    sequence = [
        CloudSessionState(id="x", status="running", step_count=1),
        CloudSessionState(id="x", status="idle", step_count=2, output="done"),
    ]
    service, client, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserSpawnInput(task="t"),
    )
    client.messages_by_session[record.cloud_session_id] = [
        CloudMessage.model_validate({
            "id": "m1",
            "sessionId": record.cloud_session_id,
            "role": "ai",
            "data": "step 1",
            "summary": "s1",
        }),
        CloudMessage.model_validate({
            "id": "m2",
            "sessionId": record.cloud_session_id,
            "role": "ai",
            "data": "step 2",
            "summary": "s2",
        }),
    ]
    await pump.tick()
    messages = [e for e in published if isinstance(e, BrowserSessionMessageReceived)]
    assert [m.cloud_message_id for m in messages] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_list_for_user_only_returns_owned_sessions(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, _, _, _ = _build(tmp_path, status_sequence=sequence)
    alice = await service.spawn(
        user_id="alice",
        conversation_id="ca",
        generation=1,
        parent_call_id="x",
        input=BrowserSpawnInput(task="t1"),
    )
    bob = await service.spawn(
        user_id="bob",
        conversation_id="cb",
        generation=1,
        parent_call_id="y",
        input=BrowserSpawnInput(task="t2"),
    )
    alice_sessions = await service.list_for_user(user_id="alice", input=BrowserListInput())
    assert {s.session_id for s in alice_sessions} == {alice.session_id}
    bob_sessions = await service.list_for_user(user_id="bob", input=BrowserListInput())
    assert {s.session_id for s in bob_sessions} == {bob.session_id}
    await service.stop(user_id="alice", input=BrowserStopInput(session_id=alice.session_id))
    await service.stop(user_id="bob", input=BrowserStopInput(session_id=bob.session_id))


@pytest.mark.asyncio
async def test_executor_routes_browser_run(tmp_path: Path) -> None:
    sequence = [
        CloudSessionState(id="x", status="idle", step_count=1, output="result"),
    ]
    service, _, _, _, pump, _, _ = _build(tmp_path, status_sequence=sequence)
    executor = ToolCallExecutor(
        runtime=FakeUserRuntime(),
        browser_use_service=service,
    )

    event = ToolCallRequested(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        call_id="call-1",
        tool_name="browser.run",
        input=BrowserRunInput(task="navigate", timeout_seconds=10.0),
    )

    completed_events = await _run_with_ticker(
        pump,
        executor.handle_tool_call_requested(event),
    )
    completed = completed_events[0]
    assert completed.tool_name == "browser.run"
    assert "result" in completed.result.stdout
    assert completed.result.exit_code == 0


@pytest.mark.asyncio
async def test_executor_browser_get_enforces_user_ownership(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, _, _, _ = _build(tmp_path, status_sequence=sequence)
    executor = ToolCallExecutor(
        runtime=FakeUserRuntime(),
        browser_use_service=service,
    )
    record = await service.spawn(
        user_id="alice",
        conversation_id="ca",
        generation=1,
        parent_call_id="x",
        input=BrowserSpawnInput(task="t"),
    )
    event = ToolCallRequested(
        user_id="bob",
        conversation_id="cb",
        generation=1,
        call_id="probe",
        tool_name="browser.get",
        input=BrowserGetInput(session_id=record.session_id),
    )
    with pytest.raises(KeyError):
        await executor.handle_tool_call_requested(event)
    await service.stop(user_id="alice", input=BrowserStopInput(session_id=record.session_id))


@pytest.mark.asyncio
async def test_executor_raises_when_service_not_configured() -> None:
    executor = ToolCallExecutor(runtime=FakeUserRuntime())
    event = ToolCallRequested(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        call_id="x",
        tool_name="browser.run",
        input=BrowserRunInput(task="t"),
    )
    with pytest.raises(RuntimeError, match="browser-use service"):
        await executor.handle_tool_call_requested(event)


@pytest.mark.asyncio
async def test_different_users_cannot_see_each_others_sessions_or_runs(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 50
    service, _, _, _, _, store, _ = _build(tmp_path, status_sequence=sequence)

    alice1 = await service.spawn(
        user_id="alice",
        conversation_id="alice-conv",
        generation=1,
        parent_call_id="a1",
        input=BrowserSpawnInput(task="alice-task-1"),
    )
    alice2 = await service.spawn(
        user_id="alice",
        conversation_id="alice-conv",
        generation=1,
        parent_call_id="a2",
        input=BrowserSpawnInput(task="alice-task-2"),
    )
    bob1 = await service.spawn(
        user_id="bob",
        conversation_id="bob-conv",
        generation=1,
        parent_call_id="b1",
        input=BrowserSpawnInput(task="bob-task-1"),
    )

    alice_list = await service.list_for_user(user_id="alice", input=BrowserListInput())
    bob_list = await service.list_for_user(user_id="bob", input=BrowserListInput())
    assert {s.session_id for s in alice_list} == {alice1.session_id, alice2.session_id}
    assert {s.session_id for s in bob_list} == {bob1.session_id}

    for victim in (alice1.session_id, alice2.session_id):
        with pytest.raises(KeyError):
            await service.get(user_id="bob", input=BrowserGetInput(session_id=victim))
        with pytest.raises(KeyError):
            await service.send(
                user_id="bob",
                input=BrowserSendInput(session_id=victim, task="hijack"),
            )
        with pytest.raises(KeyError):
            await service.stop(user_id="bob", input=BrowserStopInput(session_id=victim))

    with pytest.raises(KeyError):
        await service.get(user_id="alice", input=BrowserGetInput(session_id=bob1.session_id))

    leak_attempt = await store.get_for_user(session_id=alice1.session_id, user_id="bob")
    assert leak_attempt is None

    for sid in (alice1.session_id, alice2.session_id):
        await service.stop(user_id="alice", input=BrowserStopInput(session_id=sid))
    await service.stop(user_id="bob", input=BrowserStopInput(session_id=bob1.session_id))


@pytest.mark.asyncio
async def test_cloud_stop_emits_stopped_not_failed(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="stopped", step_count=2)]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserSpawnInput(task="t"),
    )
    await pump.tick()
    failures = [e for e in published if isinstance(e, BrowserSessionFailed)]
    stops = [e for e in published if isinstance(e, BrowserSessionStopped)]
    assert failures == []
    assert len(stops) == 1
    assert stops[0].session_id == record.session_id


@pytest.mark.asyncio
async def test_handler_drains_paginated_messages(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=1)]
    service, client, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserSpawnInput(task="t"),
    )
    cloud_id = record.cloud_session_id
    client.messages_by_session[cloud_id] = [
        CloudMessage.model_validate({"id": f"m{i}", "sessionId": cloud_id, "role": "ai", "data": f"step {i}"})
        for i in range(120)
    ]
    await pump.tick()
    messages = [e for e in published if isinstance(e, BrowserSessionMessageReceived)]
    assert len(messages) == 120
    assert messages[0].cloud_message_id == "m0"
    assert messages[-1].cloud_message_id == "m119"


@pytest.mark.asyncio
async def test_idle_completion_triggers_on_status_transition(tmp_path: Path) -> None:
    sequence = [
        CloudSessionState(id="x", status="running", step_count=0),
        CloudSessionState(id="x", status="idle", step_count=0, output="immediate"),
    ]
    service, _, _, _, pump, _, published = _build(tmp_path, status_sequence=sequence)
    record = await service.spawn(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserSpawnInput(task="t"),
    )
    await pump.tick()
    await pump.tick()
    completions = [e for e in published if isinstance(e, BrowserSessionCompleted)]
    assert len(completions) == 1
    assert completions[0].output == "immediate"
    assert completions[0].session_id == record.session_id


@pytest.mark.asyncio
async def test_run_uses_config_default_timeout_when_input_omits_it(tmp_path: Path) -> None:
    sequence = [CloudSessionState(id="x", status="running", step_count=0)] * 200
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    profile_store = SQLiteBrowserProfileStore(tmp_path / "profiles.sqlite3")
    session_store = SQLiteBrowserSessionStore(tmp_path / "sessions.sqlite3")
    bus = EventBus(event_store)
    client = FakeCloudClient(session_status_sequence=sequence)
    waiter = BrowserSessionResultWaiter()
    bus.subscribe(BrowserSessionCompleted, waiter.handle_completed)
    bus.subscribe(BrowserSessionFailed, waiter.handle_failed)
    handler = BrowserSessionPollHandler(client=client, session_store=session_store)
    bus.subscribe(BrowserSessionPollDue, handler.handle_poll_due)
    service = BrowserUseService(
        bus=bus,
        client=client,
        profile_store=profile_store,
        session_store=session_store,
        result_waiter=waiter,
        profile_cap=5,
        default_model="bu-mini",
        default_run_timeout_seconds=0.1,
    )
    record = await service.run(
        user_id="alice",
        conversation_id="c1",
        generation=1,
        parent_call_id="p1",
        input=BrowserRunInput(task="navigate"),
    )
    assert record.status == "timed_out"


@pytest.mark.asyncio
async def test_harness_app_background_services_are_ref_counted(tmp_path: Path) -> None:
    from harness_agent.app import HarnessApp
    from harness_agent.config import (
        BrowserUseConfig,
        DatabaseConfig,
        HarnessConfig,
        LlmConfig,
    )

    config = HarnessConfig(
        database=DatabaseConfig(path=tmp_path / "harness.sqlite3"),
        llm=LlmConfig(api_key="test"),
        browser_use=BrowserUseConfig(api_key="test"),
    )
    app = HarnessApp(config=config)

    pump_starts: list[None] = []
    pump_stops: list[None] = []
    sched_starts: list[None] = []
    sched_stops: list[None] = []

    async def pump_start():
        pump_starts.append(None)

    async def pump_stop():
        pump_stops.append(None)

    async def sched_start():
        sched_starts.append(None)

    async def sched_stop():
        sched_stops.append(None)

    app.browser_use_pump_service.start = pump_start  # type: ignore[assignment]
    app.browser_use_pump_service.stop = pump_stop  # type: ignore[assignment]
    app.scheduler_service.start = sched_start  # type: ignore[assignment]
    app.scheduler_service.stop = sched_stop  # type: ignore[assignment]

    await app.start_background_services()
    await app.start_background_services()
    assert len(pump_starts) == 1
    assert len(sched_starts) == 1

    await app.stop_background_services()
    assert pump_stops == []
    assert sched_stops == []

    await app.stop_background_services()
    assert len(pump_stops) == 1
    assert len(sched_stops) == 1


@pytest.mark.asyncio
async def test_profile_store_lru_ordering(tmp_path: Path) -> None:
    store = SQLiteBrowserProfileStore(tmp_path / "profiles.sqlite3")
    await store.upsert_touch(user_id="alice", cloud_profile_id="A")
    await asyncio.sleep(0.01)
    await store.upsert_touch(user_id="bob", cloud_profile_id="B")
    await asyncio.sleep(0.01)
    await store.upsert_touch(user_id="carol", cloud_profile_id="C")
    await asyncio.sleep(0.01)
    await store.touch(user_id="alice")
    ordered = await store.list_by_lru()
    assert [r.user_id for r in ordered] == ["bob", "carol", "alice"]
