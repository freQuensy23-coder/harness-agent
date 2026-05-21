import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import aiosqlite
import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from harness_agent.bus import EventBus
from harness_agent.events import (
    AgentEvent,
    BrowserProfileCreated,
    BrowserProfileEvicted,
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionMessageReceived,
    BrowserSessionPollDue,
    BrowserSessionStarted,
    BrowserSessionStopped,
    EventBase,
)
from harness_agent.tools import (
    BrowserGetInput,
    BrowserListInput,
    BrowserRunInput,
    BrowserSendInput,
    BrowserSpawnInput,
    BrowserStopInput,
)


CloudSessionStatus = Literal["created", "idle", "running", "stopped", "timed_out", "error"]
LOCAL_STATUS_COMPLETED = "completed"
ACTIVE_LOCAL_STATUSES: frozenset[str] = frozenset({"created", "running", "idle"})

EventTuple = tuple[EventBase, ...]


# --- Cloud DTOs -------------------------------------------------------------


class CloudSessionState(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    status: CloudSessionStatus
    step_count: int = Field(default=0, validation_alias="stepCount", serialization_alias="stepCount")
    output: Any = None
    last_step_summary: str | None = Field(
        default=None, validation_alias="lastStepSummary", serialization_alias="lastStepSummary"
    )
    live_url: str | None = Field(
        default=None, validation_alias="liveUrl", serialization_alias="liveUrl"
    )
    profile_id: str | None = Field(
        default=None, validation_alias="profileId", serialization_alias="profileId"
    )
    total_cost_usd: str | None = Field(
        default=None, validation_alias="totalCostUsd", serialization_alias="totalCostUsd"
    )
    is_task_successful: bool | None = Field(
        default=None, validation_alias="isTaskSuccessful", serialization_alias="isTaskSuccessful"
    )


class CloudProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    name: str | None = None
    user_id: str | None = Field(
        default=None, validation_alias="userId", serialization_alias="userId"
    )


class CloudMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    session_id: str = Field(validation_alias="sessionId", serialization_alias="sessionId")
    role: str
    data: str
    summary: str | None = None
    type: str | None = None


class CloudMessagesPage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    messages: list[CloudMessage] = Field(default_factory=list)
    has_more: bool = Field(
        default=False, validation_alias="hasMore", serialization_alias="hasMore"
    )


# --- Cloud client protocol --------------------------------------------------


class BrowserUseCloudClient(Protocol):
    async def create_profile(self, *, internal_user_id: str) -> CloudProfile: ...

    async def delete_profile(self, *, cloud_profile_id: str) -> None: ...

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState: ...

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState: ...

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState: ...

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState: ...

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage: ...


class HttpxBrowserUseClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.browser-use.com/api/v3",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _http(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers={"X-Browser-Use-API-Key": self._api_key},
                    timeout=self._timeout_seconds,
                )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def create_profile(self, *, internal_user_id: str) -> CloudProfile:
        http = await self._http()
        response = await http.post(
            "/profiles",
            json={"name": f"harness-{internal_user_id}", "userId": internal_user_id},
        )
        response.raise_for_status()
        return CloudProfile.model_validate(response.json())

    async def delete_profile(self, *, cloud_profile_id: str) -> None:
        http = await self._http()
        response = await http.delete(f"/profiles/{cloud_profile_id}")
        response.raise_for_status()

    async def create_session(
        self,
        *,
        task: str,
        cloud_profile_id: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None = None,
    ) -> CloudSessionState:
        http = await self._http()
        body: dict[str, Any] = {
            "task": task,
            "model": model,
            "profileId": cloud_profile_id,
            "keepAlive": keep_alive,
        }
        if proxy_country_code is not None:
            body["proxyCountryCode"] = proxy_country_code
        response = await http.post("/sessions", json=body)
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def send_task(
        self,
        *,
        cloud_session_id: str,
        task: str,
        model: str,
    ) -> CloudSessionState:
        http = await self._http()
        body = {
            "task": task,
            "model": model,
            "sessionId": cloud_session_id,
            "keepAlive": True,
        }
        response = await http.post("/sessions", json=body)
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def get_session(self, *, cloud_session_id: str) -> CloudSessionState:
        http = await self._http()
        response = await http.get(f"/sessions/{cloud_session_id}")
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def stop_session(
        self,
        *,
        cloud_session_id: str,
        strategy: Literal["task", "session"],
    ) -> CloudSessionState:
        http = await self._http()
        response = await http.post(
            f"/sessions/{cloud_session_id}/stop",
            json={"strategy": strategy},
        )
        response.raise_for_status()
        return CloudSessionState.model_validate(response.json())

    async def list_messages(
        self,
        *,
        cloud_session_id: str,
        after: str | None = None,
        limit: int = 50,
    ) -> CloudMessagesPage:
        http = await self._http()
        params: dict[str, str | int] = {"limit": limit}
        if after is not None:
            params["after"] = after
        response = await http.get(
            f"/sessions/{cloud_session_id}/messages",
            params=params,
        )
        response.raise_for_status()
        return CloudMessagesPage.model_validate(response.json())


# --- Persistent records -----------------------------------------------------


class BrowserProfileRecord(BaseModel):
    user_id: str
    cloud_profile_id: str
    created_at: datetime
    last_used_at: datetime


class BrowserSessionRecord(BaseModel):
    session_id: str
    user_id: str
    conversation_id: str
    generation: int
    parent_call_id: str
    cloud_session_id: str
    cloud_profile_id: str
    status: str
    keep_alive: bool
    task: str
    model: str
    live_url: str | None = None
    output: str | None = None
    error: str | None = None
    last_cloud_message_id: str | None = None
    created_at: datetime
    updated_at: datetime


# --- Stores -----------------------------------------------------------------


class SQLiteBrowserProfileStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def upsert_touch(
        self,
        *,
        user_id: str,
        cloud_profile_id: str,
    ) -> BrowserProfileRecord:
        await self._ensure_schema()
        now = datetime.now(UTC)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into browser_profiles (user_id, cloud_profile_id, created_at, last_used_at)
                values (?, ?, ?, ?)
                on conflict(user_id) do update set
                    cloud_profile_id = excluded.cloud_profile_id,
                    last_used_at = excluded.last_used_at
                """,
                (user_id, cloud_profile_id, now.isoformat(), now.isoformat()),
            )
            await db.commit()
        existing = await self.get(user_id=user_id)
        if existing is None:
            raise RuntimeError("profile upsert lost record")
        return existing

    async def touch(self, *, user_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "update browser_profiles set last_used_at = ? where user_id = ?",
                (datetime.now(UTC).isoformat(), user_id),
            )
            await db.commit()

    async def get(self, *, user_id: str) -> BrowserProfileRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select user_id, cloud_profile_id, created_at, last_used_at
                from browser_profiles where user_id = ?
                """,
                (user_id,),
            )
        if not rows:
            return None
        return _profile_from_row(rows[0])

    async def list_by_lru(self) -> list[BrowserProfileRecord]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select user_id, cloud_profile_id, created_at, last_used_at
                from browser_profiles order by last_used_at asc
                """
            )
        return [_profile_from_row(row) for row in rows]

    async def delete(self, *, user_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "delete from browser_profiles where user_id = ?",
                (user_id,),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists browser_profiles (
                    user_id text primary key,
                    cloud_profile_id text not null,
                    created_at text not null,
                    last_used_at text not null
                )
                """
            )
            await db.commit()


_SESSION_COLUMNS = (
    "session_id, user_id, conversation_id, generation, parent_call_id, "
    "cloud_session_id, cloud_profile_id, status, keep_alive, task, model, "
    "live_url, output, error, last_cloud_message_id, created_at, updated_at"
)


class SQLiteBrowserSessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def create(self, record: BrowserSessionRecord) -> BrowserSessionRecord:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                f"""
                insert into browser_sessions ({_SESSION_COLUMNS})
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _session_row(record),
            )
            await db.commit()
        return record

    async def update_status(
        self,
        *,
        session_id: str,
        status: str,
        live_url: str | None = None,
        output: str | None = None,
        error: str | None = None,
    ) -> BrowserSessionRecord | None:
        await self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update browser_sessions
                set status = ?,
                    live_url = coalesce(?, live_url),
                    output = coalesce(?, output),
                    error = coalesce(?, error),
                    updated_at = ?
                where session_id = ?
                """,
                (status, live_url, output, error, now, session_id),
            )
            await db.commit()
        return await self.get_internal(session_id=session_id)

    async def set_last_cloud_message_id(
        self,
        *,
        session_id: str,
        cloud_message_id: str,
    ) -> None:
        await self._ensure_schema()
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update browser_sessions
                set last_cloud_message_id = ?,
                    updated_at = ?
                where session_id = ?
                """,
                (cloud_message_id, now, session_id),
            )
            await db.commit()

    async def get_internal(self, *, session_id: str) -> BrowserSessionRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                f"select {_SESSION_COLUMNS} from browser_sessions where session_id = ?",
                (session_id,),
            )
        if not rows:
            return None
        return _session_from_row(rows[0])

    async def get_for_user(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> BrowserSessionRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                f"""
                select {_SESSION_COLUMNS} from browser_sessions
                where session_id = ? and user_id = ?
                """,
                (session_id, user_id),
            )
        if not rows:
            return None
        return _session_from_row(rows[0])

    async def list_for_user(
        self,
        *,
        user_id: str,
        include_terminal: bool,
        limit: int,
    ) -> list[BrowserSessionRecord]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            if include_terminal:
                rows = await db.execute_fetchall(
                    f"""
                    select {_SESSION_COLUMNS} from browser_sessions
                    where user_id = ?
                    order by created_at desc
                    limit ?
                    """,
                    (user_id, limit),
                )
            else:
                placeholders = ",".join("?" for _ in ACTIVE_LOCAL_STATUSES)
                rows = await db.execute_fetchall(
                    f"""
                    select {_SESSION_COLUMNS} from browser_sessions
                    where user_id = ? and status in ({placeholders})
                    order by created_at desc
                    limit ?
                    """,
                    (user_id, *sorted(ACTIVE_LOCAL_STATUSES), limit),
                )
        return [_session_from_row(row) for row in rows]

    async def list_active(self) -> list[BrowserSessionRecord]:
        await self._ensure_schema()
        placeholders = ",".join("?" for _ in ACTIVE_LOCAL_STATUSES)
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                f"""
                select {_SESSION_COLUMNS} from browser_sessions
                where status in ({placeholders})
                order by updated_at asc
                """,
                tuple(sorted(ACTIVE_LOCAL_STATUSES)),
            )
        return [_session_from_row(row) for row in rows]

    async def count_active_for_user(self, *, user_id: str) -> int:
        await self._ensure_schema()
        placeholders = ",".join("?" for _ in ACTIVE_LOCAL_STATUSES)
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                f"""
                select count(*) from browser_sessions
                where user_id = ? and status in ({placeholders})
                """,
                (user_id, *sorted(ACTIVE_LOCAL_STATUSES)),
            )
        return int(rows[0][0])

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists browser_sessions (
                    session_id text primary key,
                    user_id text not null,
                    conversation_id text not null,
                    generation integer not null,
                    parent_call_id text not null,
                    cloud_session_id text not null,
                    cloud_profile_id text not null,
                    status text not null,
                    keep_alive integer not null,
                    task text not null,
                    model text not null,
                    live_url text,
                    output text,
                    error text,
                    last_cloud_message_id text,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            await db.execute(
                "create index if not exists browser_sessions_user on browser_sessions(user_id)"
            )
            await db.execute(
                "create index if not exists browser_sessions_status on browser_sessions(status)"
            )
            await db.commit()


# --- Result waiter (event-mediated completion for browser.run) --------------


TerminalEvent = (
    BrowserSessionCompleted
    | BrowserSessionFailed
    | BrowserSessionStopped
)


class BrowserSessionResultWaiter:
    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[TerminalEvent]] = {}

    def expect(self, *, session_id: str) -> None:
        self._futures[session_id] = asyncio.get_running_loop().create_future()

    async def wait(self, *, session_id: str) -> TerminalEvent:
        return await self._futures[session_id]

    def forget(self, *, session_id: str) -> None:
        self._futures.pop(session_id, None)

    async def handle_completed(self, event: BrowserSessionCompleted) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    async def handle_failed(self, event: BrowserSessionFailed) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    async def handle_stopped(self, event: BrowserSessionStopped) -> EventTuple:
        self._resolve(event.session_id, event)
        return ()

    def _resolve(self, session_id: str, event: TerminalEvent) -> None:
        future = self._futures.get(session_id)
        if future is not None and not future.done():
            future.set_result(event)


# --- Poll handler (subscribed to BrowserSessionPollDue) ---------------------


class BrowserSessionPollHandler:
    """Drives one polling cycle for a session in response to a PollDue event.

    Returns the lifecycle events the bus should publish next. All state is
    read from and written to the session store; no in-memory coordination.
    """

    def __init__(
        self,
        *,
        client: BrowserUseCloudClient,
        session_store: SQLiteBrowserSessionStore,
    ) -> None:
        self._client = client
        self._sessions = session_store

    async def handle_poll_due(self, event: BrowserSessionPollDue) -> EventTuple:
        record = await self._sessions.get_internal(session_id=event.session_id)
        if record is None or record.status not in ACTIVE_LOCAL_STATUSES:
            return ()
        try:
            state = await self._client.get_session(cloud_session_id=record.cloud_session_id)
            messages = await self._drain_all_messages(record)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.warning(
                "browser poll failed for session {id}: {err}",
                id=record.session_id,
                err=error,
            )
            return ()

        emitted: list[EventBase] = []
        for message in messages:
            emitted.append(
                BrowserSessionMessageReceived(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    cloud_message_id=message.id,
                    role=message.role,
                    summary=message.summary,
                    data=message.data,
                )
            )

        if _is_success_terminal(state, record):
            output_text = _render_output(state.output)
            local_status = (
                "idle"
                if record.keep_alive and state.status == "idle"
                else LOCAL_STATUS_COMPLETED
            )
            await self._sessions.update_status(
                session_id=record.session_id,
                status=local_status,
                output=output_text,
            )
            emitted.append(
                BrowserSessionCompleted(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    output=output_text,
                    step_count=state.step_count,
                )
            )
            return tuple(emitted)

        if state.status == "stopped" and state.is_task_successful is False:
            error_text = "cloud reported is_task_successful=false"
            await self._sessions.update_status(
                session_id=record.session_id,
                status="error",
                error=error_text,
            )
            emitted.append(
                BrowserSessionFailed(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    status="error",
                    error=error_text,
                )
            )
            return tuple(emitted)

        if state.status == "stopped":
            await self._sessions.update_status(
                session_id=record.session_id,
                status="stopped",
            )
            emitted.append(
                BrowserSessionStopped(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    requested_by_user_id=record.user_id,
                )
            )
            return tuple(emitted)

        if state.status in ("timed_out", "error"):
            await self._sessions.update_status(
                session_id=record.session_id,
                status=state.status,
            )
            emitted.append(
                BrowserSessionFailed(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    status=state.status,
                    error=None,
                )
            )
            return tuple(emitted)

        await self._sessions.update_status(
            session_id=record.session_id,
            status=state.status,
            live_url=state.live_url,
        )
        return tuple(emitted)

    async def _drain_all_messages(
        self,
        record: BrowserSessionRecord,
    ) -> list[CloudMessage]:
        drained: list[CloudMessage] = []
        cursor = record.last_cloud_message_id
        max_pages = 20
        while max_pages > 0:
            page = await self._client.list_messages(
                cloud_session_id=record.cloud_session_id,
                after=cursor,
                limit=50,
            )
            for message in page.messages:
                drained.append(message)
                cursor = message.id
                await self._sessions.set_last_cloud_message_id(
                    session_id=record.session_id,
                    cloud_message_id=message.id,
                )
            if not page.has_more or not page.messages:
                break
            max_pages -= 1
        return drained


# --- Pump (publishes BrowserSessionPollDue for every active session) --------


class BrowserSessionPump:
    def __init__(
        self,
        *,
        session_store: SQLiteBrowserSessionStore,
        bus: EventBus,
    ) -> None:
        self._sessions = session_store
        self._bus = bus

    async def tick(self) -> None:
        active = await self._sessions.list_active()
        for record in active:
            await self._bus.publish(
                BrowserSessionPollDue(
                    session_id=record.session_id,
                    user_id=record.user_id,
                )
            )


class BrowserSessionPumpService:
    def __init__(self, *, pump: BrowserSessionPump, poll_seconds: float) -> None:
        self._pump = pump
        self._poll_seconds = poll_seconds
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._pump.tick()
            except Exception as exc:
                logger.warning(
                    "browser pump tick failed: {err}",
                    err=str(exc) or exc.__class__.__name__,
                )
            await asyncio.sleep(self._poll_seconds)


# --- Service (orchestrates profile lifecycle + session lifecycle commands) --


class BrowserUseService:
    def __init__(
        self,
        *,
        bus: EventBus,
        client: BrowserUseCloudClient,
        profile_store: SQLiteBrowserProfileStore,
        session_store: SQLiteBrowserSessionStore,
        result_waiter: BrowserSessionResultWaiter,
        profile_cap: int,
        default_model: str,
        default_run_timeout_seconds: float = 600.0,
    ) -> None:
        self._bus = bus
        self._client = client
        self._profiles = profile_store
        self._sessions = session_store
        self._result_waiter = result_waiter
        self._profile_cap = profile_cap
        self._default_model = default_model
        self._default_run_timeout_seconds = default_run_timeout_seconds
        self._profile_lock = asyncio.Lock()

    async def run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        parent_call_id: str,
        input: BrowserRunInput,
    ) -> BrowserSessionRecord:
        record = await self._start_session(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            parent_call_id=parent_call_id,
            task=input.task,
            model=input.model or self._default_model,
            keep_alive=False,
            proxy_country_code=input.proxy_country_code,
        )
        self._result_waiter.expect(session_id=record.session_id)
        timeout = (
            input.timeout_seconds
            if input.timeout_seconds is not None
            else self._default_run_timeout_seconds
        )
        try:
            await asyncio.wait_for(
                self._result_waiter.wait(session_id=record.session_id),
                timeout=timeout,
            )
        except TimeoutError:
            await self._handle_timeout(record, timeout=timeout)
        finally:
            self._result_waiter.forget(session_id=record.session_id)
        latest = await self._sessions.get_for_user(
            session_id=record.session_id,
            user_id=user_id,
        )
        if latest is None:
            raise RuntimeError("session disappeared after run")
        return latest

    async def spawn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        parent_call_id: str,
        input: BrowserSpawnInput,
    ) -> BrowserSessionRecord:
        return await self._start_session(
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            parent_call_id=parent_call_id,
            task=input.task,
            model=input.model or self._default_model,
            keep_alive=True,
            proxy_country_code=input.proxy_country_code,
        )

    async def get(
        self,
        *,
        user_id: str,
        input: BrowserGetInput,
    ) -> tuple[BrowserSessionRecord, list[CloudMessage]]:
        record = await self._sessions.get_for_user(
            session_id=input.session_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(input.session_id)
        messages: list[CloudMessage] = []
        if input.include_messages:
            page = await self._client.list_messages(
                cloud_session_id=record.cloud_session_id,
                limit=input.messages_limit,
            )
            messages = page.messages
        return record, messages

    async def send(
        self,
        *,
        user_id: str,
        input: BrowserSendInput,
    ) -> BrowserSessionRecord:
        record = await self._sessions.get_for_user(
            session_id=input.session_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(input.session_id)
        if not record.keep_alive:
            raise ValueError(
                "browser.send only works on keep-alive sessions started via browser.spawn"
            )
        state = await self._client.send_task(
            cloud_session_id=record.cloud_session_id,
            task=input.task,
            model=record.model,
        )
        await self._profiles.touch(user_id=user_id)
        updated = await self._sessions.update_status(
            session_id=record.session_id,
            status=state.status,
            live_url=state.live_url,
        )
        return updated or record

    async def stop(
        self,
        *,
        user_id: str,
        input: BrowserStopInput,
    ) -> BrowserSessionRecord:
        record = await self._sessions.get_for_user(
            session_id=input.session_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(input.session_id)
        state = await self._client.stop_session(
            cloud_session_id=record.cloud_session_id,
            strategy=input.strategy,
        )
        updated = await self._sessions.update_status(
            session_id=record.session_id,
            status=state.status,
        )
        await self._bus.publish(
            BrowserSessionStopped(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                cloud_session_id=record.cloud_session_id,
                requested_by_user_id=user_id,
            )
        )
        return updated or record

    async def list_for_user(
        self,
        *,
        user_id: str,
        input: BrowserListInput,
    ) -> list[BrowserSessionRecord]:
        return await self._sessions.list_for_user(
            user_id=user_id,
            include_terminal=input.include_terminal,
            limit=input.limit,
        )

    async def ensure_profile(self, *, user_id: str) -> BrowserProfileRecord:
        async with self._profile_lock:
            existing = await self._profiles.get(user_id=user_id)
            if existing is not None:
                await self._profiles.touch(user_id=user_id)
                touched = await self._profiles.get(user_id=user_id)
                if touched is None:
                    raise RuntimeError("profile lost after touch")
                return touched
            await self._evict_for_capacity(requesting_user_id=user_id)
            cloud_profile = await self._client.create_profile(internal_user_id=user_id)
            record = await self._profiles.upsert_touch(
                user_id=user_id,
                cloud_profile_id=cloud_profile.id,
            )
            await self._bus.publish(
                BrowserProfileCreated(
                    user_id=user_id,
                    cloud_profile_id=cloud_profile.id,
                )
            )
            return record

    async def _start_session(
        self,
        *,
        user_id: str,
        conversation_id: str,
        generation: int,
        parent_call_id: str,
        task: str,
        model: str,
        keep_alive: bool,
        proxy_country_code: str | None,
    ) -> BrowserSessionRecord:
        profile = await self.ensure_profile(user_id=user_id)
        state = await self._client.create_session(
            task=task,
            cloud_profile_id=profile.cloud_profile_id,
            model=model,
            keep_alive=keep_alive,
            proxy_country_code=proxy_country_code,
        )
        now = datetime.now(UTC)
        record = BrowserSessionRecord(
            session_id=uuid4().hex,
            user_id=user_id,
            conversation_id=conversation_id,
            generation=generation,
            parent_call_id=parent_call_id,
            cloud_session_id=state.id,
            cloud_profile_id=profile.cloud_profile_id,
            status=state.status,
            keep_alive=keep_alive,
            task=task,
            model=model,
            live_url=state.live_url,
            created_at=now,
            updated_at=now,
        )
        await self._sessions.create(record)
        await self._bus.publish(
            BrowserSessionStarted(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                generation=record.generation,
                parent_call_id=record.parent_call_id,
                cloud_session_id=record.cloud_session_id,
                cloud_profile_id=record.cloud_profile_id,
                live_url=record.live_url,
            )
        )
        return record

    async def _evict_for_capacity(self, *, requesting_user_id: str) -> None:
        existing = await self._profiles.list_by_lru()
        if len(existing) < self._profile_cap:
            return
        for candidate in existing:
            if candidate.user_id == requesting_user_id:
                continue
            active = await self._sessions.count_active_for_user(user_id=candidate.user_id)
            if active > 0:
                logger.info(
                    "Skipping eviction of profile for {user_id}: {n} active sessions",
                    user_id=candidate.user_id,
                    n=active,
                )
                continue
            await self._evict(candidate, requesting_user_id=requesting_user_id)
            return
        raise RuntimeError(
            "Cannot create browser profile: cap reached and every other profile "
            "has live sessions. Try again after sessions complete."
        )

    async def _evict(
        self,
        candidate: BrowserProfileRecord,
        *,
        requesting_user_id: str,
    ) -> None:
        try:
            await self._client.delete_profile(cloud_profile_id=candidate.cloud_profile_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
        await self._profiles.delete(user_id=candidate.user_id)
        await self._bus.publish(
            BrowserProfileEvicted(
                evicted_user_id=candidate.user_id,
                cloud_profile_id=candidate.cloud_profile_id,
                requested_by_user_id=requesting_user_id,
            )
        )

    async def _handle_timeout(
        self,
        record: BrowserSessionRecord,
        *,
        timeout: float,
    ) -> None:
        error = f"browser.run timed out after {timeout}s"
        try:
            await self._client.stop_session(
                cloud_session_id=record.cloud_session_id,
                strategy="session",
            )
        except Exception as exc:
            logger.warning(
                "Failed to stop timed-out cloud session {id}: {err}",
                id=record.cloud_session_id,
                err=str(exc) or exc.__class__.__name__,
            )
        await self._sessions.update_status(
            session_id=record.session_id,
            status="timed_out",
            error=error,
        )
        await self._bus.publish(
            BrowserSessionFailed(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                cloud_session_id=record.cloud_session_id,
                status="timed_out",
                error=error,
            )
        )


# --- Rendering helpers ------------------------------------------------------


def render_browser_session(
    record: BrowserSessionRecord,
    *,
    messages: list[CloudMessage] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "session_id": record.session_id,
        "status": record.status,
        "task": record.task,
        "model": record.model,
        "keep_alive": record.keep_alive,
        "live_url": record.live_url,
        "output": record.output,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
    if messages is not None:
        payload["messages"] = [
            {
                "role": m.role,
                "summary": m.summary,
                "data": m.data,
            }
            for m in messages
        ]
    return _json_dumps(payload)


def render_browser_sessions(records: list[BrowserSessionRecord]) -> str:
    return _json_dumps(
        [
            {
                "session_id": r.session_id,
                "status": r.status,
                "task": r.task,
                "keep_alive": r.keep_alive,
                "live_url": r.live_url,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in records
        ]
    )


def _is_success_terminal(
    state: CloudSessionState,
    record: BrowserSessionRecord,
) -> bool:
    """Decide whether a cloud terminal state is success vs. abort.

    The cloud transitions non-keep-alive sessions running -> stopped on
    completion, but `is_task_successful` is sometimes left null even on
    success. When `is_task_successful` is unset, presence of `output` is
    the reliable success signal; an explicit `is_task_successful=false`
    overrides both and is treated as failure upstream.
    """
    if state.status == "idle" and record.status != "idle":
        return True
    if state.status == "stopped":
        if state.is_task_successful is True:
            return True
        if state.is_task_successful is None and state.output is not None:
            return True
    return False


def _render_output(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        return output
    return _json_dumps(output)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _profile_from_row(row: tuple) -> BrowserProfileRecord:
    return BrowserProfileRecord(
        user_id=row[0],
        cloud_profile_id=row[1],
        created_at=datetime.fromisoformat(row[2]),
        last_used_at=datetime.fromisoformat(row[3]),
    )


def _session_row(record: BrowserSessionRecord) -> tuple:
    return (
        record.session_id,
        record.user_id,
        record.conversation_id,
        record.generation,
        record.parent_call_id,
        record.cloud_session_id,
        record.cloud_profile_id,
        record.status,
        1 if record.keep_alive else 0,
        record.task,
        record.model,
        record.live_url,
        record.output,
        record.error,
        record.last_cloud_message_id,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
    )


def _session_from_row(row: tuple) -> BrowserSessionRecord:
    return BrowserSessionRecord(
        session_id=row[0],
        user_id=row[1],
        conversation_id=row[2],
        generation=int(row[3]),
        parent_call_id=row[4],
        cloud_session_id=row[5],
        cloud_profile_id=row[6],
        status=row[7],
        keep_alive=bool(row[8]),
        task=row[9],
        model=row[10],
        live_url=row[11],
        output=row[12],
        error=row[13],
        last_cloud_message_id=row[14],
        created_at=datetime.fromisoformat(row[15]),
        updated_at=datetime.fromisoformat(row[16]),
    )


__all__ = [
    "AgentEvent",
    "BrowserProfileRecord",
    "BrowserSessionPollHandler",
    "BrowserSessionPump",
    "BrowserSessionPumpService",
    "BrowserSessionRecord",
    "BrowserSessionResultWaiter",
    "BrowserUseCloudClient",
    "BrowserUseService",
    "CloudMessage",
    "CloudMessagesPage",
    "CloudProfile",
    "CloudSessionState",
    "HttpxBrowserUseClient",
    "SQLiteBrowserProfileStore",
    "SQLiteBrowserSessionStore",
    "render_browser_session",
    "render_browser_sessions",
]
