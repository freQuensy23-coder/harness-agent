"""SQLite-backed projection stores for browser-use profiles and
sessions. Both stores are event-derived projections: the
BrowserUseService writes to them after publishing a typed event."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from harness_agent.browser_use.records import (
    ACTIVE_LOCAL_STATUSES,
    BrowserProfileRecord,
    BrowserSessionRecord,
)
from harness_agent.db import fetchall_rows


_SESSION_COLUMNS = (
    "session_id, user_id, conversation_id, generation, parent_call_id, "
    "cloud_session_id, cloud_profile_id, status, keep_alive, task, model, "
    "live_url, output, error, last_cloud_message_id, created_at, updated_at"
)


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
            rows = await fetchall_rows(
                db,
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
            rows = await fetchall_rows(
                db,
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
            rows = await fetchall_rows(
                db,
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
            rows = await fetchall_rows(
                db,
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
                rows = await fetchall_rows(
                    db,
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
                rows = await fetchall_rows(
                    db,
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
            rows = await fetchall_rows(
                db,
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
            rows = await fetchall_rows(
                db,
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


def _profile_from_row(row: tuple[Any, ...]) -> BrowserProfileRecord:
    return BrowserProfileRecord(
        user_id=row[0],
        cloud_profile_id=row[1],
        created_at=datetime.fromisoformat(row[2]),
        last_used_at=datetime.fromisoformat(row[3]),
    )


def _session_row(record: BrowserSessionRecord) -> tuple[Any, ...]:
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


def _session_from_row(row: tuple[Any, ...]) -> BrowserSessionRecord:
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
