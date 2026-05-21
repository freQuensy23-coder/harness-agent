from pathlib import Path
from typing import Any

import aiosqlite

from harness_agent.db import fetchall_rows
from harness_agent.runtime.models import SpawnedProcessRecord


class SQLiteSpawnedProcessStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def create(self, record: SpawnedProcessRecord) -> SpawnedProcessRecord:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into spawned_processes (
                    process_id,
                    user_id,
                    container_name,
                    command,
                    cwd,
                    base_path,
                    stdout_path,
                    stderr_path,
                    pid_path,
                    exit_code_path,
                    stdout_offset,
                    stderr_offset
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _spawned_process_row(record),
            )
            await db.commit()
        return record

    async def get(self, *, process_id: str, user_id: str) -> SpawnedProcessRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select
                    process_id,
                    user_id,
                    container_name,
                    command,
                    cwd,
                    base_path,
                    stdout_path,
                    stderr_path,
                    pid_path,
                    exit_code_path,
                    stdout_offset,
                    stderr_offset
                from spawned_processes
                where process_id = ? and user_id = ?
                """,
                (process_id, user_id),
            )
        if not rows:
            return None
        return _spawned_process_from_row(rows[0])

    async def update_offsets(
        self,
        *,
        process_id: str,
        user_id: str,
        stdout_offset: int,
        stderr_offset: int,
    ) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update spawned_processes
                set stdout_offset = ?, stderr_offset = ?
                where process_id = ? and user_id = ?
                """,
                (stdout_offset, stderr_offset, process_id, user_id),
            )
            await db.commit()

    async def delete(self, *, process_id: str, user_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                delete from spawned_processes
                where process_id = ? and user_id = ?
                """,
                (process_id, user_id),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists spawned_processes (
                    process_id text not null,
                    user_id text not null,
                    container_name text not null,
                    command text not null,
                    cwd text not null,
                    base_path text not null,
                    stdout_path text not null,
                    stderr_path text not null,
                    pid_path text not null,
                    exit_code_path text not null,
                    stdout_offset integer not null,
                    stderr_offset integer not null,
                    primary key (process_id, user_id)
                )
                """
            )
            await db.commit()


def _spawned_process_row(record: SpawnedProcessRecord) -> tuple[Any, ...]:
    return (
        record.process_id,
        record.user_id,
        record.container_name,
        record.command,
        record.cwd,
        record.base_path,
        record.stdout_path,
        record.stderr_path,
        record.pid_path,
        record.exit_code_path,
        record.stdout_offset,
        record.stderr_offset,
    )


def _spawned_process_from_row(row: tuple[Any, ...]) -> SpawnedProcessRecord:
    return SpawnedProcessRecord(
        process_id=row[0],
        user_id=row[1],
        container_name=row[2],
        command=row[3],
        cwd=row[4],
        base_path=row[5],
        stdout_path=row[6],
        stderr_path=row[7],
        pid_path=row[8],
        exit_code_path=row[9],
        stdout_offset=row[10],
        stderr_offset=row[11],
    )
