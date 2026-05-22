import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel

from harness_agent.bus import EventBus
from harness_agent.db import fetchall_rows
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
from harness_agent.tools import AgentRunInput, AgentSpawnInput


SubAgentStatus = Literal["running", "completed", "failed", "cancelled"]


EventBatch = tuple[EventBase, ...]


class SubAgentRecord(BaseModel):
    id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str
    status: SubAgentStatus
    result: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SubAgentLookup(Protocol):
    async def get_by_child_conversation_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None: ...


class SQLiteSubAgentStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def insert(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
        child_conversation_id: str,
        parent_call_id: str,
        name: str,
        prompt: str,
    ) -> SubAgentRecord:
        now = datetime.now(UTC)
        record = SubAgentRecord(
            id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            child_conversation_id=child_conversation_id,
            parent_call_id=parent_call_id,
            name=name,
            prompt=prompt,
            status="running",
            created_at=now,
            updated_at=now,
        )
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into sub_agents (
                    id,
                    user_id,
                    parent_conversation_id,
                    child_conversation_id,
                    parent_call_id,
                    name,
                    prompt,
                    status,
                    result,
                    error,
                    created_at,
                    updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _record_row(record),
            )
            await db.commit()
        return record

    async def complete(self, agent_id: str, result: str) -> SubAgentRecord:
        return await self._transition(agent_id, status="completed", result=result, error=None)

    async def fail(self, agent_id: str, error: str) -> SubAgentRecord:
        return await self._transition(agent_id, status="failed", result=None, error=error)

    async def cancel_for_parent(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord | None:
        return await self._transition_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            status="cancelled",
            result=None,
            error=None,
        )

    async def get(self, agent_id: str) -> SubAgentRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where id = ?",
                (agent_id,),
            )
        if not rows:
            return None
        return _record_from_row(rows[0])

    async def get_for_parent(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where id = ? and user_id = ? and parent_conversation_id = ?",
                (agent_id, user_id, parent_conversation_id),
            )
        if not rows:
            return None
        return _record_from_row(rows[0])

    async def get_by_child_conversation_id(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> SubAgentRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where user_id = ? and child_conversation_id = ?",
                (user_id, conversation_id),
            )
        if not rows:
            return None
        return _record_from_row(rows[0])

    async def list_for_parent(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        include_completed: bool,
    ) -> list[SubAgentRecord]:
        await self._ensure_schema()
        query = _SELECT_BASE + " where user_id = ? and parent_conversation_id = ?"
        params: tuple[Any, ...] = (user_id, parent_conversation_id)
        if not include_completed:
            query += " and status = ?"
            params = (*params, "running")
        query += " order by created_at asc"
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(db, query, params)
        return [_record_from_row(row) for row in rows]

    async def _transition(
        self,
        agent_id: str,
        *,
        status: SubAgentStatus,
        result: str | None,
        error: str | None,
    ) -> SubAgentRecord:
        record = await self._transition_running(
            agent_id=agent_id,
            status=status,
            result=result,
            error=error,
        )
        if record is None:
            raise KeyError(agent_id)
        return record

    async def _transition_for_parent(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
        status: SubAgentStatus,
        result: str | None,
        error: str | None,
    ) -> SubAgentRecord | None:
        now = datetime.now(UTC)
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update sub_agents
                set status = ?, result = ?, error = ?, updated_at = ?
                where id = ?
                  and user_id = ?
                  and parent_conversation_id = ?
                  and status = ?
                """,
                (
                    status,
                    result,
                    error,
                    now.isoformat(),
                    agent_id,
                    user_id,
                    parent_conversation_id,
                    "running",
                ),
            )
            await db.commit()
        return await self.get_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )

    async def _transition_running(
        self,
        *,
        agent_id: str,
        status: SubAgentStatus,
        result: str | None,
        error: str | None,
    ) -> SubAgentRecord | None:
        now = datetime.now(UTC)
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update sub_agents
                set status = ?, result = ?, error = ?, updated_at = ?
                where id = ? and status = ?
                """,
                (status, result, error, now.isoformat(), agent_id, "running"),
            )
            await db.commit()
        return await self.get(agent_id)

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists sub_agents (
                    id text primary key,
                    user_id text not null,
                    parent_conversation_id text not null,
                    child_conversation_id text not null unique,
                    parent_call_id text not null,
                    name text not null,
                    prompt text not null,
                    status text not null,
                    result text,
                    error text,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            await db.commit()


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
            return await future
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
        cancelled = await self._store.cancel_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )
        if cancelled is None or cancelled.status != "cancelled":
            return cancelled
        self._cancel_timeout(agent_id)
        await self._kill_child(agent_id)
        await self._bus.publish(
            SubAgentCancelled(
                agent_id=cancelled.id,
                user_id=cancelled.user_id,
                parent_conversation_id=cancelled.parent_conversation_id,
                child_conversation_id=cancelled.child_conversation_id,
            )
        )
        return cancelled

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
        completed = await self._store.complete(record.id, event.text)
        if completed.status != "completed":
            return ()
        self._cancel_timeout(record.id)
        self._forget_child(record.id)
        return (
            SubAgentCompleted(
                agent_id=completed.id,
                user_id=completed.user_id,
                parent_conversation_id=completed.parent_conversation_id,
                child_conversation_id=completed.child_conversation_id,
                result=event.text,
            ),
        )

    async def handle_timed_out(self, event: SubAgentTimedOut) -> EventBatch:
        record = await self._store.get(event.agent_id)
        if record is None or record.status != "running":
            return ()
        error = f"sub-agent {record.id} timed out"
        failed = await self._store.fail(record.id, error)
        if failed.status != "failed":
            return ()
        self._forget_timeout(record.id)
        await self._kill_child(record.id)
        return (
            SubAgentFailed(
                agent_id=failed.id,
                user_id=failed.user_id,
                parent_conversation_id=failed.parent_conversation_id,
                child_conversation_id=failed.child_conversation_id,
                error=error,
            ),
        )

    async def handle_completed(self, event: SubAgentCompleted) -> EventBatch:
        await self._resolve_pending(event.agent_id)
        return ()

    async def handle_failed(self, event: SubAgentFailed) -> EventBatch:
        await self._resolve_pending(event.agent_id)
        return ()

    async def handle_cancelled(self, event: SubAgentCancelled) -> EventBatch:
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
            failed = await self._store.fail(record.id, error)
            if failed.status != "failed":
                self._forget_child(event.agent_id)
                return
            self._cancel_timeout(record.id)
            self._forget_child(record.id)
            await self._bus.publish(
                SubAgentFailed(
                    agent_id=failed.id,
                    user_id=failed.user_id,
                    parent_conversation_id=failed.parent_conversation_id,
                    child_conversation_id=failed.child_conversation_id,
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

    def _forget_timeout(self, agent_id: str) -> None:
        self._timeout_tasks.pop(agent_id, None)

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


def render_sub_agent_record(record: SubAgentRecord) -> str:
    return record.model_dump_json(indent=2)


def render_sub_agent_records(records: list[SubAgentRecord]) -> str:
    return "[\n" + ",\n".join(record.model_dump_json(indent=2) for record in records) + "\n]"


_SELECT_BASE = (
    "select "
    "id, user_id, parent_conversation_id, child_conversation_id, parent_call_id, "
    "name, prompt, status, result, error, created_at, updated_at "
    "from sub_agents"
)


def _record_row(record: SubAgentRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.user_id,
        record.parent_conversation_id,
        record.child_conversation_id,
        record.parent_call_id,
        record.name,
        record.prompt,
        record.status,
        record.result,
        record.error,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
    )


def _record_from_row(row: tuple[Any, ...]) -> SubAgentRecord:
    return SubAgentRecord(
        id=row[0],
        user_id=row[1],
        parent_conversation_id=row[2],
        child_conversation_id=row[3],
        parent_call_id=row[4],
        name=row[5],
        prompt=row[6],
        status=row[7],
        result=row[8],
        error=row[9],
        created_at=datetime.fromisoformat(row[10]),
        updated_at=datetime.fromisoformat(row[11]),
    )
