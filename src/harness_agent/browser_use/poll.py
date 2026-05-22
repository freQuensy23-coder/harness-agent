"""Polling pipeline.

BrowserSessionPump publishes BrowserSessionPollDue for every active
session on each tick. BrowserSessionPollHandler subscribes to PollDue,
calls the cloud, and emits the corresponding lifecycle events.
BrowserSessionPumpService runs the pump on a background asyncio loop."""

import asyncio
from typing import Any

from loguru import logger

from harness_agent.browser_use.cloud_client import BrowserUseCloudClient
from harness_agent.browser_use.cloud_dtos import CloudMessage, CloudSessionState
from harness_agent.browser_use.records import (
    ACTIVE_LOCAL_STATUSES,
    BrowserSessionRecord,
)
from harness_agent.browser_use.stores import SQLiteBrowserSessionStore
from harness_agent.bus import EventBus
from harness_agent.events import (
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionMessageReceived,
    BrowserSessionPollDue,
    BrowserSessionStatusChanged,
    BrowserSessionStopped,
    EventBase,
)


EventTuple = tuple[EventBase, ...]


class BrowserSessionPollHandler:
    """One polling cycle for a session. Reads/writes state via the
    session store; no in-memory coordination."""

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
            # Terminal projection writes (completed/failed/stopped) are
            # done by BrowserUseService handlers subscribed to these
            # terminal events; the event log is the source of truth.
            keep_alive_idle = record.keep_alive and state.status == "idle"
            emitted.append(
                BrowserSessionCompleted(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    output=_render_output(state.output),
                    step_count=state.step_count,
                    keep_alive_idle=keep_alive_idle,
                )
            )
            return tuple(emitted)

        if state.status == "stopped" and state.is_task_successful is False:
            emitted.append(
                BrowserSessionFailed(
                    session_id=record.session_id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    cloud_session_id=record.cloud_session_id,
                    status="error",
                    error="cloud reported is_task_successful=false",
                )
            )
            return tuple(emitted)

        if state.status == "stopped":
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

        # Non-terminal status transition (e.g. created -> running):
        # emit StatusChanged; the projection write happens in the
        # subscribed handler so mid-flight state changes are also event-
        # derived.
        emitted.append(
            BrowserSessionStatusChanged(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                cloud_session_id=record.cloud_session_id,
                status=state.status,
                live_url=state.live_url,
            )
        )
        return tuple(emitted)

    async def _drain_all_messages(
        self,
        record: BrowserSessionRecord,
    ) -> list[CloudMessage]:
        """Page through cloud messages newer than the stored cursor.
        Does NOT advance the SQLite cursor here -- the cursor is moved
        forward by handle_message_received() AFTER EventBus has
        persisted the corresponding BrowserSessionMessageReceived event.
        A crash between the cloud fetch and event persistence leaves the
        cursor where it was, so the next tick re-fetches the same
        messages instead of silently skipping them."""
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
            if not page.has_more or not page.messages:
                break
            max_pages -= 1
        return drained

    async def handle_message_received(
        self,
        event: BrowserSessionMessageReceived,
    ) -> EventTuple:
        """Advance the stored cursor AFTER EventBus has persisted the
        message event. This makes the SQLite cursor an event-derived
        projection: a missing event in the log means the corresponding
        cursor advance also never happened."""
        await self._sessions.set_last_cloud_message_id(
            session_id=event.session_id,
            cloud_message_id=event.cloud_message_id,
        )
        return ()


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
        if self._task is not None and not self._task.done():
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
    import json

    return json.dumps(output, indent=2, default=str)
