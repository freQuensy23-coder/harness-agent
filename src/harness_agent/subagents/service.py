"""Event-driven sub-agent lifecycle coordinator.

run() publishes SubAgentRequested and awaits a terminal event.
handle_requested promotes the request to SubAgentStarted; handle_started
spawns the child-turn task + timeout watchdog. Terminal-event handlers
(completed/failed/cancelled) are the only writers of the row's final
status."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from harness_agent.bus import EventBus
from harness_agent.events import (
    AssistantTextProduced,
    EventBase,
    SubAgentCancelled,
    SubAgentCompleted,
    SubAgentFailed,
    SubAgentRequested,
    SubAgentStarted,
    SubAgentTimedOut,
    UserTextReceived,
)
from harness_agent.subagents.models import SubAgentRecord
from harness_agent.subagents.store import SQLiteSubAgentStore
from harness_agent.tools import AgentRunInput, AgentSpawnInput


EventBatch = tuple[EventBase, ...]
_RUN_WATCHDOG_GRACE_SECONDS = 10.0


class SubAgentService:
    """Event-driven sub-agent lifecycle coordinator."""

    def __init__(
        self,
        *,
        bus: EventBus,
        store: SQLiteSubAgentStore,
    ) -> None:
        self._bus = bus
        self._store = store
        self._child_tasks: dict[str, asyncio.Task[Any]] = {}
        self._timeout_tasks: dict[str, asyncio.Task[Any]] = {}
        self._pending_records: dict[str, asyncio.Future[SubAgentRecord]] = {}

    # ------------------------- tool API -------------------------

    async def run(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentRunInput,
    ) -> SubAgentRecord:
        agent_id, child_conversation_id = self._mint_ids(parent_conversation_id)
        future: asyncio.Future[SubAgentRecord] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_records[agent_id] = future
        # Hard upper bound so run() always settles even if a terminal event
        # never reaches the bus (handler crash, lost publish): once we go
        # past timeout_seconds + watchdog grace, fall back to the store.
        deadline = input.timeout_seconds + _RUN_WATCHDOG_GRACE_SECONDS
        try:
            await self._bus.publish(
                SubAgentRequested(
                    agent_id=agent_id,
                    user_id=user_id,
                    parent_conversation_id=parent_conversation_id,
                    child_conversation_id=child_conversation_id,
                    parent_call_id=parent_call_id,
                    name=input.name,
                    prompt=input.prompt,
                    timeout_seconds=input.timeout_seconds,
                )
            )
            try:
                return await asyncio.wait_for(future, timeout=deadline)
            except TimeoutError:
                return await self._settle_from_store(
                    agent_id=agent_id,
                    fallback_error=(
                        f"sub-agent {agent_id} did not publish a terminal event "
                        f"within {deadline:.0f}s"
                    ),
                )
        finally:
            self._pending_records.pop(agent_id, None)

    async def spawn(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentSpawnInput,
    ) -> SubAgentRecord:
        agent_id, child_conversation_id = self._mint_ids(parent_conversation_id)
        now = datetime.now(UTC)
        snapshot = SubAgentRecord(
            id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            child_conversation_id=child_conversation_id,
            parent_call_id=parent_call_id,
            name=input.name,
            prompt=input.prompt,
            status="running",
            created_at=now,
            updated_at=now,
        )
        await self._bus.publish(
            SubAgentRequested(
                agent_id=agent_id,
                user_id=user_id,
                parent_conversation_id=parent_conversation_id,
                child_conversation_id=child_conversation_id,
                parent_call_id=parent_call_id,
                name=input.name,
                prompt=input.prompt,
                timeout_seconds=input.timeout_seconds,
            )
        )
        return snapshot

    async def result(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord | None:
        return await self._store.get_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )

    async def list_for_parent(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        include_completed: bool,
    ) -> list[SubAgentRecord]:
        return await self._store.list_for_parent(
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            include_completed=include_completed,
        )

    async def cancel(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord | None:
        record = await self._store.get_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )
        if record is None:
            return None
        if record.status != "running":
            return record
        # Publish first so the audit event is durably persisted before any
        # store mutation. handle_cancelled does the SQLite transition.
        await self._bus.publish(
            SubAgentCancelled(
                agent_id=record.id,
                user_id=record.user_id,
                parent_conversation_id=record.parent_conversation_id,
                child_conversation_id=record.child_conversation_id,
            )
        )
        return await self._store.get_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )

    # ------------------------- event handlers -------------------------

    async def handle_requested(self, event: SubAgentRequested) -> EventBatch:
        record = await self._store.insert(
            agent_id=event.agent_id,
            user_id=event.user_id,
            parent_conversation_id=event.parent_conversation_id,
            child_conversation_id=event.child_conversation_id,
            parent_call_id=event.parent_call_id,
            name=event.name,
            prompt=event.prompt,
        )
        return (
            SubAgentStarted(
                agent_id=record.id,
                user_id=record.user_id,
                parent_conversation_id=record.parent_conversation_id,
                child_conversation_id=record.child_conversation_id,
                parent_call_id=record.parent_call_id,
                name=record.name,
                prompt=record.prompt,
                timeout_seconds=event.timeout_seconds,
            ),
        )

    async def handle_started(self, event: SubAgentStarted) -> EventBatch:
        self._child_tasks[event.agent_id] = asyncio.create_task(
            self._run_child_turn(event)
        )
        self._timeout_tasks[event.agent_id] = asyncio.create_task(
            self._timeout_watchdog(event)
        )
        return ()

    async def handle_assistant_text(self, event: AssistantTextProduced) -> EventBatch:
        record = await self._store.get_by_child_conversation_id(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
        )
        if record is None or record.status != "running":
            return ()
        # Emit the terminal event; handle_completed will mutate the store
        # only after EventBus has persisted this event.
        return (
            SubAgentCompleted(
                agent_id=record.id,
                user_id=record.user_id,
                parent_conversation_id=record.parent_conversation_id,
                child_conversation_id=record.child_conversation_id,
                result=event.text,
            ),
        )

    async def handle_timed_out(self, event: SubAgentTimedOut) -> EventBatch:
        record = await self._store.get(event.agent_id)
        if record is None or record.status != "running":
            return ()
        return (
            SubAgentFailed(
                agent_id=record.id,
                user_id=record.user_id,
                parent_conversation_id=record.parent_conversation_id,
                child_conversation_id=record.child_conversation_id,
                error=f"sub-agent {record.id} timed out",
            ),
        )

    async def handle_completed(self, event: SubAgentCompleted) -> EventBatch:
        completed = await self._store.complete(event.agent_id, event.result)
        if completed.status == "completed":
            self._cancel_timeout(event.agent_id)
            self._forget_child(event.agent_id)
        await self._resolve_pending(event.agent_id)
        return ()

    async def handle_failed(self, event: SubAgentFailed) -> EventBatch:
        failed = await self._store.fail(event.agent_id, event.error)
        if failed.status == "failed":
            # Cancel the watchdog and the child turn task, but never
            # cancel the task that is currently dispatching us: the
            # watchdog itself, or the child-turn error path, are both
            # legitimate sources of SubAgentFailed and they will exit
            # naturally once this handler returns.
            current = asyncio.current_task()
            timeout_task = self._timeout_tasks.pop(event.agent_id, None)
            if timeout_task is not None and timeout_task is not current and not timeout_task.done():
                timeout_task.cancel()
            child_task = self._child_tasks.pop(event.agent_id, None)
            if child_task is not None and child_task is not current and not child_task.done():
                child_task.cancel()
                try:
                    await child_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        await self._resolve_pending(event.agent_id)
        return ()

    async def handle_cancelled(self, event: SubAgentCancelled) -> EventBatch:
        await self._store.cancel_for_parent(
            agent_id=event.agent_id,
            user_id=event.user_id,
            parent_conversation_id=event.parent_conversation_id,
        )
        self._cancel_timeout(event.agent_id)
        await self._kill_child(event.agent_id)
        await self._resolve_pending(event.agent_id)
        return ()

    # ------------------------- lookup -------------------------

    async def get_by_child_conversation_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None:
        return await self._store.get_by_child_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    # ------------------------- internals -------------------------

    def _mint_ids(self, parent_conversation_id: str) -> tuple[str, str]:
        agent_id = uuid4().hex
        child_conversation_id = f"{parent_conversation_id}:subagent:{agent_id}"
        return agent_id, child_conversation_id

    async def _settle_from_store(
        self,
        *,
        agent_id: str,
        fallback_error: str,
    ) -> SubAgentRecord:
        record = await self._store.get(agent_id)
        if record is None:
            raise RuntimeError(f"sub-agent {agent_id} disappeared")
        if record.status != "running":
            return record
        # Settle via the event path so handle_failed remains the only
        # writer for the failed transition. If the event chain is
        # miswired and the row stays running, that is a bus configuration
        # bug, not a recoverable state -- raise loudly.
        await self._bus.publish(
            SubAgentFailed(
                agent_id=record.id,
                user_id=record.user_id,
                parent_conversation_id=record.parent_conversation_id,
                child_conversation_id=record.child_conversation_id,
                error=fallback_error,
            )
        )
        settled = await self._store.get(agent_id)
        if settled is None:
            raise RuntimeError(f"sub-agent {agent_id} disappeared after watchdog publish")
        if settled.status == "running":
            raise RuntimeError(
                f"sub-agent {agent_id} did not transition after SubAgentFailed; "
                "check that SubAgentService.handle_failed is subscribed."
            )
        return settled

    async def _resolve_pending(self, agent_id: str) -> None:
        future = self._pending_records.get(agent_id)
        if future is None or future.done():
            return
        record = await self._store.get(agent_id)
        if record is None:
            future.set_exception(RuntimeError(f"sub-agent {agent_id} disappeared"))
            return
        future.set_result(record)

    async def _run_child_turn(self, event: SubAgentStarted) -> None:
        try:
            await self._bus.publish(
                UserTextReceived(
                    user_id=event.user_id,
                    conversation_id=event.child_conversation_id,
                    source="subagent",
                    text=event.prompt,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            record = await self._store.get(event.agent_id)
            if record is None or record.status != "running":
                self._forget_child(event.agent_id)
                return
            # Publish first; handle_failed performs the store mutation only
            # after the event has been persisted by EventBus.
            await self._bus.publish(
                SubAgentFailed(
                    agent_id=record.id,
                    user_id=record.user_id,
                    parent_conversation_id=record.parent_conversation_id,
                    child_conversation_id=record.child_conversation_id,
                    error=error,
                )
            )

    async def _timeout_watchdog(self, event: SubAgentStarted) -> None:
        try:
            await asyncio.sleep(event.timeout_seconds)
        except asyncio.CancelledError:
            return
        try:
            await self._bus.publish(
                SubAgentTimedOut(
                    agent_id=event.agent_id,
                    user_id=event.user_id,
                    parent_conversation_id=event.parent_conversation_id,
                    child_conversation_id=event.child_conversation_id,
                )
            )
        except asyncio.CancelledError:
            return

    def _cancel_timeout(self, agent_id: str) -> None:
        task = self._timeout_tasks.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()

    def _forget_child(self, agent_id: str) -> None:
        self._child_tasks.pop(agent_id, None)

    async def _kill_child(self, agent_id: str) -> None:
        task = self._child_tasks.pop(agent_id, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except Exception:
            return
