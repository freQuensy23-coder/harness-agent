import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
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
    SubAgentStarted,
    UserTextReceived,
)
from harness_agent.tools import AgentRunInput, AgentSpawnInput


SubAgentStatus = Literal["running", "completed", "failed", "cancelled"]


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


class SQLiteSubAgentStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def create(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        name: str,
        prompt: str,
    ) -> SubAgentRecord:
        now = datetime.now(UTC)
        agent_id = uuid4().hex
        record = SubAgentRecord(
            id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            child_conversation_id=f"{parent_conversation_id}:subagent:{agent_id}",
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
                """
                select
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
                from sub_agents
                where id = ?
                """,
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
                """
                select
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
                from sub_agents
                where id = ?
                  and user_id = ?
                  and parent_conversation_id = ?
                """,
                (agent_id, user_id, parent_conversation_id),
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
        if include_completed:
            return await self._list_for_parent(
                user_id=user_id,
                parent_conversation_id=parent_conversation_id,
            )
        return await self._list_running_for_parent(
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )

    async def _list_for_parent(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
    ) -> list[SubAgentRecord]:
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select
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
                from sub_agents
                where user_id = ?
                  and parent_conversation_id = ?
                order by created_at asc
                """,
                (user_id, parent_conversation_id),
            )
        return [_record_from_row(row) for row in rows]

    async def _list_running_for_parent(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
    ) -> list[SubAgentRecord]:
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select
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
                from sub_agents
                where user_id = ?
                  and parent_conversation_id = ?
                  and status = ?
                order by created_at asc
                """,
                (user_id, parent_conversation_id, "running"),
            )
        return [_record_from_row(row) for row in rows]

    async def _transition(
        self,
        agent_id: str,
        *,
        status: SubAgentStatus,
        result: str | None,
        error: str | None,
    ) -> SubAgentRecord:
        record = await self._transition_row(
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
        return await self._transition_row(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            status=status,
            result=result,
            error=error,
        )

    async def _transition_row(
        self,
        *,
        agent_id: str,
        status: SubAgentStatus,
        result: str | None,
        error: str | None,
        user_id: str | None = None,
        parent_conversation_id: str | None = None,
    ) -> SubAgentRecord | None:
        if user_id is not None and parent_conversation_id is None:
            raise ValueError("parent_conversation_id is required")
        now = datetime.now(UTC)
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            if user_id is None:
                await db.execute(
                    """
                    update sub_agents
                    set status = ?,
                        result = ?,
                        error = ?,
                        updated_at = ?
                    where id = ?
                      and status = ?
                    """,
                    (status, result, error, now.isoformat(), agent_id, "running"),
                )
            else:
                await db.execute(
                    """
                    update sub_agents
                    set status = ?,
                        result = ?,
                        error = ?,
                        updated_at = ?
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
        if user_id is None:
            return await self.get(agent_id)
        assert parent_conversation_id is not None
        return await self.get_for_parent(
            agent_id=agent_id,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
        )

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


class SubAgentResultWaiter:
    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[str]] = {}

    def expect(self, conversation_id: str) -> None:
        self._pending[conversation_id] = asyncio.get_running_loop().create_future()

    async def wait(self, conversation_id: str) -> str:
        return await self._pending[conversation_id]

    def forget(self, conversation_id: str) -> None:
        self._pending.pop(conversation_id, None)

    async def handle_assistant_text(self, event: AssistantTextProduced) -> tuple[EventBase, ...]:
        future = self._pending.get(event.conversation_id)
        if future is not None and not future.done():
            future.set_result(event.text)
        return ()


class SubAgentService:
    def __init__(
        self,
        *,
        bus: EventBus,
        store: SQLiteSubAgentStore,
        result_waiter: SubAgentResultWaiter,
    ) -> None:
        self._bus = bus
        self._store = store
        self._result_waiter = result_waiter
        self._tasks: dict[str, asyncio.Task[SubAgentRecord]] = {}

    def _forget_task(self, agent_id: str) -> None:
        self._tasks.pop(agent_id, None)

    async def run(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentRunInput,
    ) -> SubAgentRecord:
        record = await self._store.create(
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            parent_call_id=parent_call_id,
            name=input.name,
            prompt=input.prompt,
        )
        return await self._execute(record, timeout_seconds=input.timeout_seconds)

    async def spawn(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentSpawnInput,
    ) -> SubAgentRecord:
        record = await self._store.create(
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
            parent_call_id=parent_call_id,
            name=input.name,
            prompt=input.prompt,
        )
        task = asyncio.create_task(
            self._execute(record, timeout_seconds=input.timeout_seconds)
        )
        self._tasks[record.id] = task
        task.add_done_callback(
            lambda completed, agent_id=record.id: self._forget_task(agent_id)
        )
        return record

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
        record = await self.result(
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
        if cancelled is None:
            return None
        if cancelled.status != "cancelled":
            return cancelled
        task = self._tasks.pop(agent_id, None)
        if task is not None:
            task.cancel()
            await self._await_cancelled_task(task)
        await self._bus.publish(
            SubAgentCancelled(
                agent_id=cancelled.id,
                user_id=cancelled.user_id,
                parent_conversation_id=cancelled.parent_conversation_id,
                child_conversation_id=cancelled.child_conversation_id,
            )
        )
        return cancelled

    async def _await_cancelled_task(self, task: asyncio.Task[Any]) -> None:
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _execute(
        self,
        record: SubAgentRecord,
        *,
        timeout_seconds: float,
    ) -> SubAgentRecord:
        self._result_waiter.expect(record.child_conversation_id)
        try:
            await self._bus.publish(
                SubAgentStarted(
                    agent_id=record.id,
                    user_id=record.user_id,
                    parent_conversation_id=record.parent_conversation_id,
                    child_conversation_id=record.child_conversation_id,
                    parent_call_id=record.parent_call_id,
                    name=record.name,
                )
            )
            result = await asyncio.wait_for(
                self._publish_child_turn_and_wait_result(record),
                timeout=timeout_seconds,
            )
            completed = await self._store.complete(record.id, result)
            if completed.status != "completed":
                return completed
            await self._bus.publish(
                SubAgentCompleted(
                    agent_id=completed.id,
                    user_id=completed.user_id,
                    parent_conversation_id=completed.parent_conversation_id,
                    child_conversation_id=completed.child_conversation_id,
                    result=result,
                )
            )
            return completed
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            failed = await self._store.fail(record.id, error)
            if failed.status != "failed":
                return failed
            await self._bus.publish(
                SubAgentFailed(
                    agent_id=failed.id,
                    user_id=failed.user_id,
                    parent_conversation_id=failed.parent_conversation_id,
                    child_conversation_id=failed.child_conversation_id,
                    error=error,
                )
            )
            return failed
        finally:
            self._result_waiter.forget(record.child_conversation_id)

    async def _publish_child_turn_and_wait_result(self, record: SubAgentRecord) -> str:
        publish_task = asyncio.create_task(
            self._bus.publish(
                UserTextReceived(
                    user_id=record.user_id,
                    conversation_id=record.child_conversation_id,
                    source="subagent",
                    text=record.prompt,
                )
            )
        )
        wait_task = asyncio.create_task(
            self._result_waiter.wait(record.child_conversation_id)
        )
        try:
            done, _ = await asyncio.wait(
                {publish_task, wait_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if publish_task in done:
                await publish_task
                return await wait_task
            result = await wait_task
            await publish_task
            return result
        finally:
            for task in (publish_task, wait_task):
                if not task.done():
                    task.cancel()
                    await self._await_cancelled_task(task)


def render_sub_agent_record(record: SubAgentRecord) -> str:
    return record.model_dump_json(indent=2)


def render_sub_agent_records(records: list[SubAgentRecord]) -> str:
    return "[\n" + ",\n".join(record.model_dump_json(indent=2) for record in records) + "\n]"


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
