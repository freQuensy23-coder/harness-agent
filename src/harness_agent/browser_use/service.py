"""Top-level orchestration for browser.* tool calls.

BrowserUseService owns profile lifecycle (LRU eviction with live-session
skip) and session command verbs (run/spawn/get/send/stop/list). It
publishes a typed lifecycle event for every persistent state change
and writes the SQLite projection inside the same call, so the event
log stays the canonical timeline even though the projection write is
not itself event-driven."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from loguru import logger

from harness_agent.browser_use.cloud_client import BrowserUseCloudClient
from harness_agent.browser_use.cloud_dtos import CloudMessage
from harness_agent.browser_use.records import (
    LOCAL_STATUS_COMPLETED,
    BrowserProfileRecord,
    BrowserSessionRecord,
)
from harness_agent.browser_use.stores import (
    SQLiteBrowserProfileStore,
    SQLiteBrowserSessionStore,
)
from harness_agent.browser_use.waiter import BrowserSessionResultWaiter
from harness_agent.bus import EventBus
from harness_agent.events import (
    BrowserProfileCreated,
    BrowserProfileEvicted,
    BrowserProfileTouched,
    BrowserSessionCompleted,
    BrowserSessionFailed,
    BrowserSessionStarted,
    BrowserSessionStatusChanged,
    BrowserSessionStopped,
    BrowserSessionTaskSent,
)
from harness_agent.tools import (
    BrowserGetInput,
    BrowserListInput,
    BrowserRunInput,
    BrowserSendInput,
    BrowserSpawnInput,
    BrowserStopInput,
)


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
        await self._bus.publish(
            BrowserSessionTaskSent(
                session_id=record.session_id,
                user_id=record.user_id,
                cloud_session_id=record.cloud_session_id,
                task=input.task,
                status=state.status,
            )
        )
        await self._bus.publish(
            BrowserSessionStatusChanged(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                cloud_session_id=record.cloud_session_id,
                status=state.status,
                live_url=state.live_url,
            )
        )
        await self._bus.publish(
            BrowserProfileTouched(
                user_id=user_id,
                cloud_profile_id=record.cloud_profile_id,
            )
        )
        updated = await self._sessions.get_for_user(
            session_id=record.session_id,
            user_id=user_id,
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
        await self._client.stop_session(
            cloud_session_id=record.cloud_session_id,
            strategy=input.strategy,
        )
        # The projection write happens in handle_session_stopped, which is
        # subscribed before BrowserSessionResultWaiter.handle_stopped so a
        # browser.run waiter that resumes on this event sees the updated
        # row, not the previous running state.
        await self._bus.publish(
            BrowserSessionStopped(
                session_id=record.session_id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                cloud_session_id=record.cloud_session_id,
                requested_by_user_id=user_id,
            )
        )
        updated = await self._sessions.get_for_user(
            session_id=record.session_id,
            user_id=user_id,
        )
        return updated or record

    # ------------------------- event handlers -------------------------

    async def handle_profile_touched(self, event: BrowserProfileTouched) -> tuple[()]:
        await self._profiles.touch(user_id=event.user_id)
        return ()

    async def handle_session_status_changed(
        self,
        event: BrowserSessionStatusChanged,
    ) -> tuple[()]:
        """Apply a non-terminal status/live_url change to the projection.
        Terminal transitions (completed/failed/stopped) have their own
        handlers."""
        await self._sessions.update_status(
            session_id=event.session_id,
            status=event.status,
            live_url=event.live_url,
        )
        return ()

    async def handle_session_stopped(self, event: BrowserSessionStopped) -> tuple[()]:
        """Write the terminal 'stopped' status to the projection. Wire
        this BEFORE BrowserSessionResultWaiter.handle_stopped so a
        browser.run that resumes on this event already sees the
        updated row."""
        await self._sessions.update_status(
            session_id=event.session_id,
            status="stopped",
        )
        return ()

    async def handle_session_completed(self, event: BrowserSessionCompleted) -> tuple[()]:
        """Write the terminal status. A keep-alive spawn that the cloud
        flipped to idle stays 'idle' (still reusable via browser.send);
        all other completions move to 'completed'."""
        status = "idle" if event.keep_alive_idle else LOCAL_STATUS_COMPLETED
        await self._sessions.update_status(
            session_id=event.session_id,
            status=status,
            output=event.output,
        )
        return ()

    async def handle_session_failed(self, event: BrowserSessionFailed) -> tuple[()]:
        await self._sessions.update_status(
            session_id=event.session_id,
            status=event.status,
            error=event.error,
        )
        return ()

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
                # Touch via the event path so LRU recency is derived
                # from the audit log, not an out-of-band SQLite update.
                await self._bus.publish(
                    BrowserProfileTouched(
                        user_id=user_id,
                        cloud_profile_id=existing.cloud_profile_id,
                    )
                )
                touched = await self._profiles.get(user_id=user_id)
                if touched is None:
                    raise RuntimeError("profile lost after touch")
                return touched
            await self._evict_for_capacity(requesting_user_id=user_id)
            cloud_profile = await self._client.create_profile(internal_user_id=user_id)
            await self._bus.publish(
                BrowserProfileCreated(
                    user_id=user_id,
                    cloud_profile_id=cloud_profile.id,
                )
            )
            record = await self._profiles.upsert_touch(
                user_id=user_id,
                cloud_profile_id=cloud_profile.id,
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
        await self._sessions.create(record)
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
        await self._bus.publish(
            BrowserProfileEvicted(
                evicted_user_id=candidate.user_id,
                cloud_profile_id=candidate.cloud_profile_id,
                requested_by_user_id=requesting_user_id,
            )
        )
        await self._profiles.delete(user_id=candidate.user_id)

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
        await self._sessions.update_status(
            session_id=record.session_id,
            status="timed_out",
            error=error,
        )
