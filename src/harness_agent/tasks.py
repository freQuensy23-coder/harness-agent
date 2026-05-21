import json
from pathlib import Path
from uuid import uuid4

import aiosqlite
from pydantic import BaseModel

from harness_agent.db import fetchall_rows


class Task(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    title: str
    status: str


class SQLiteTaskStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def create(
        self,
        *,
        user_id: str,
        conversation_id: str,
        title: str,
        status: str,
    ) -> Task:
        await self._ensure_schema()
        task = Task(
            id=uuid4().hex,
            user_id=user_id,
            conversation_id=conversation_id,
            title=title,
            status=status,
        )
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into tasks (id, user_id, conversation_id, title, status)
                values (?, ?, ?, ?, ?)
                """,
                (task.id, user_id, conversation_id, title, status),
            )
            await db.commit()
        return task

    async def get(self, *, task_id: str, user_id: str, conversation_id: str) -> Task | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select id, user_id, conversation_id, title, status
                from tasks
                where id = ? and user_id = ? and conversation_id = ?
                """,
                (task_id, user_id, conversation_id),
            )
        if not rows:
            return None
        row = rows[0]
        return Task(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2],
            title=row[3],
            status=row[4],
        )

    async def list(
        self,
        *,
        user_id: str,
        conversation_id: str,
        include_stopped: bool,
    ) -> list[Task]:
        await self._ensure_schema()
        where = "user_id = ? and conversation_id = ?"
        params: list[str] = [user_id, conversation_id]
        if not include_stopped:
            where += " and status != ?"
            params.append("stopped")
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                f"""
                select id, user_id, conversation_id, title, status
                from tasks
                where {where}
                order by sequence asc
                """,
                params,
            )
        return [
            Task(
                id=row[0],
                user_id=row[1],
                conversation_id=row[2],
                title=row[3],
                status=row[4],
            )
            for row in rows
        ]

    async def update(
        self,
        *,
        task_id: str,
        user_id: str,
        conversation_id: str,
        title: str | None,
        status: str | None,
    ) -> Task | None:
        existing = await self.get(
            task_id=task_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if existing is None:
            return None
        next_title = existing.title if title is None else title
        next_status = existing.status if status is None else status
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update tasks
                set title = ?, status = ?
                where id = ? and user_id = ? and conversation_id = ?
                """,
                (next_title, next_status, task_id, user_id, conversation_id),
            )
            await db.commit()
        return await self.get(task_id=task_id, user_id=user_id, conversation_id=conversation_id)

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists tasks (
                    sequence integer primary key autoincrement,
                    id text not null unique,
                    user_id text not null,
                    conversation_id text not null,
                    title text not null,
                    status text not null
                )
                """
            )
            await db.commit()


def tasks_to_json(tasks: list[Task]) -> str:
    return json.dumps([task.model_dump() for task in tasks], indent=2)
