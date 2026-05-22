"""SQLite-backed projection store for sub-agent records. Read/write
methods are scoped to (user_id, parent_conversation_id) so a parent
conversation cannot read or mutate another conversation's sub-agents."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from harness_agent.db import fetchall_rows
from harness_agent.subagents.models import SubAgentRecord, SubAgentStatus


_SELECT_BASE = (
    "select "
    "id, user_id, parent_conversation_id, child_conversation_id, parent_call_id, "
    "name, prompt, status, result, error, created_at, updated_at "
    "from sub_agents"
)


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
            status=status,
            result=result,
            error=error,
            user_id=user_id,
            parent_conversation_id=parent_conversation_id,
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
                    set status = ?, result = ?, error = ?, updated_at = ?
                    where id = ? and status = ?
                    """,
                    (status, result, error, now.isoformat(), agent_id, "running"),
                )
            else:
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
