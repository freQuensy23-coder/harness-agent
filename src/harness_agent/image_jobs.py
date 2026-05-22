import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import aiosqlite
from loguru import logger
from pydantic import BaseModel

from harness_agent.db import fetchall_rows
from harness_agent.image_generate import (
    ImageGenerationError,
    ImageGenerator,
)
from harness_agent.runtime.protocols import UserRuntime
from harness_agent.tools import ImageGenerateInput


ImageJobStatus = Literal["running", "completed", "failed"]


class ImageJobRecord(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    parent_call_id: str
    prompt: str
    output_path: str
    aspect_ratio: str
    status: ImageJobStatus
    mime_type: str | None = None
    size_bytes: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SQLiteImageJobStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def insert(
        self,
        *,
        job_id: str,
        user_id: str,
        conversation_id: str,
        parent_call_id: str,
        prompt: str,
        output_path: str,
        aspect_ratio: str,
    ) -> ImageJobRecord:
        now = datetime.now(UTC)
        record = ImageJobRecord(
            id=job_id,
            user_id=user_id,
            conversation_id=conversation_id,
            parent_call_id=parent_call_id,
            prompt=prompt,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            status="running",
            created_at=now,
            updated_at=now,
        )
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into image_jobs (
                    id, user_id, conversation_id, parent_call_id, prompt,
                    output_path, aspect_ratio, status, mime_type, size_bytes,
                    error, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _record_row(record),
            )
            await db.commit()
        return record

    async def complete(
        self,
        *,
        job_id: str,
        mime_type: str,
        size_bytes: int,
    ) -> ImageJobRecord | None:
        return await self._transition(
            job_id=job_id,
            status="completed",
            mime_type=mime_type,
            size_bytes=size_bytes,
            error=None,
        )

    async def fail(self, *, job_id: str, error: str) -> ImageJobRecord | None:
        return await self._transition(
            job_id=job_id,
            status="failed",
            mime_type=None,
            size_bytes=None,
            error=error,
        )

    async def get(
        self,
        *,
        job_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ImageJobRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where id = ? and user_id = ? and conversation_id = ?",
                (job_id, user_id, conversation_id),
            )
        if not rows:
            return None
        return _record_from_row(rows[0])

    async def list_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[ImageJobRecord]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE
                + " where user_id = ? and conversation_id = ? order by created_at asc",
                (user_id, conversation_id),
            )
        return [_record_from_row(row) for row in rows]

    async def _transition(
        self,
        *,
        job_id: str,
        status: ImageJobStatus,
        mime_type: str | None,
        size_bytes: int | None,
        error: str | None,
    ) -> ImageJobRecord | None:
        now = datetime.now(UTC)
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update image_jobs
                set status = ?, mime_type = ?, size_bytes = ?, error = ?, updated_at = ?
                where id = ? and status = ?
                """,
                (
                    status,
                    mime_type,
                    size_bytes,
                    error,
                    now.isoformat(),
                    job_id,
                    "running",
                ),
            )
            await db.commit()
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where id = ?",
                (job_id,),
            )
        if not rows:
            return None
        return _record_from_row(rows[0])

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists image_jobs (
                    id text primary key,
                    user_id text not null,
                    conversation_id text not null,
                    parent_call_id text not null,
                    prompt text not null,
                    output_path text not null,
                    aspect_ratio text not null,
                    status text not null,
                    mime_type text,
                    size_bytes integer,
                    error text,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            await db.commit()


class ImageJobService:
    def __init__(
        self,
        *,
        store: SQLiteImageJobStore,
        generator: ImageGenerator,
        runtime: UserRuntime,
    ) -> None:
        self._store = store
        self._generator = generator
        self._runtime = runtime
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        *,
        user_id: str,
        conversation_id: str,
        parent_call_id: str,
        input: ImageGenerateInput,
    ) -> ImageJobRecord:
        job_id = uuid4().hex
        record = await self._store.insert(
            job_id=job_id,
            user_id=user_id,
            conversation_id=conversation_id,
            parent_call_id=parent_call_id,
            prompt=input.prompt,
            output_path=input.output_path,
            aspect_ratio=input.aspect_ratio,
        )
        self._tasks[job_id] = asyncio.create_task(self._run(record, input))
        return record

    async def get(
        self,
        *,
        job_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ImageJobRecord | None:
        return await self._store.get(
            job_id=job_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def list_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> list[ImageJobRecord]:
        return await self._store.list_for_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def _run(self, record: ImageJobRecord, input: ImageGenerateInput) -> None:
        try:
            await self._run_to_completion(record, input)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("image job {} failed", record.id)
            await self._store.fail(
                job_id=record.id,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        finally:
            self._tasks.pop(record.id, None)

    async def _run_to_completion(
        self,
        record: ImageJobRecord,
        input: ImageGenerateInput,
    ) -> None:
        try:
            image = await self._generator.generate(input)
        except ImageGenerationError as exc:
            await self._store.fail(job_id=record.id, error=str(exc))
            return
        write = await self._runtime.write_content_file(
            record.user_id,
            record.output_path,
            image.data,
        )
        if write.exit_code != 0:
            await self._store.fail(
                job_id=record.id,
                error=write.stderr or f"failed to write {record.output_path}",
            )
            return
        await self._store.complete(
            job_id=record.id,
            mime_type=image.mime_type,
            size_bytes=len(image.data),
        )


def render_image_job_record(record: ImageJobRecord) -> str:
    return record.model_dump_json(indent=2)


_SELECT_BASE = (
    "select "
    "id, user_id, conversation_id, parent_call_id, prompt, output_path, "
    "aspect_ratio, status, mime_type, size_bytes, error, created_at, updated_at "
    "from image_jobs"
)


def _record_row(record: ImageJobRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.user_id,
        record.conversation_id,
        record.parent_call_id,
        record.prompt,
        record.output_path,
        record.aspect_ratio,
        record.status,
        record.mime_type,
        record.size_bytes,
        record.error,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
    )


def _record_from_row(row: tuple[Any, ...]) -> ImageJobRecord:
    return ImageJobRecord(
        id=row[0],
        user_id=row[1],
        conversation_id=row[2],
        parent_call_id=row[3],
        prompt=row[4],
        output_path=row[5],
        aspect_ratio=row[6],
        status=row[7],
        mime_type=row[8],
        size_bytes=row[9],
        error=row[10],
        created_at=datetime.fromisoformat(row[11]),
        updated_at=datetime.fromisoformat(row[12]),
    )
