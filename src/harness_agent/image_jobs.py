import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import aiosqlite
from loguru import logger
from pydantic import BaseModel

from harness_agent.bus import EventBus
from harness_agent.content import content_ref_from_bytes
from harness_agent.db import fetchall_rows
from harness_agent.events import (
    EventBase,
    ImageJobCompleted,
    ImageJobFailed,
    ImageJobRequested,
    ImageJobStarted,
)
from harness_agent.image_generate import (
    ImageGenerationError,
    ImageGenerator,
)
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime.protocols import UserRuntime
from harness_agent.tools import ImageGenerateInput


ImageJobStatus = Literal["running", "completed", "failed"]


EventBatch = tuple[EventBase, ...]


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

    async def list_running(self) -> list[ImageJobRecord]:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                _SELECT_BASE + " where status = 'running'",
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
    """Event-driven image generation lifecycle.

    `start()` publishes ImageJobRequested and awaits ImageJobStarted, returning
    the running snapshot. The actual generator HTTP call runs in a background
    task spawned from handle_started; it emits ImageJobCompleted on success or
    ImageJobFailed on error, both of which are persisted to the store by their
    respective handlers.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        store: SQLiteImageJobStore,
        generator: ImageGenerator,
        runtime: UserRuntime,
    ) -> None:
        self._bus = bus
        self._store = store
        self._generator = generator
        self._runtime = runtime
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._pending_started: dict[str, asyncio.Future[ImageJobRecord]] = {}

    # ------------------------- tool API -------------------------

    async def start(
        self,
        *,
        user_id: str,
        conversation_id: str,
        parent_call_id: str,
        input: ImageGenerateInput,
    ) -> ImageJobRecord:
        job_id = _mint_job_id()
        future: asyncio.Future[ImageJobRecord] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_started[job_id] = future
        try:
            await self._bus.publish(
                ImageJobRequested(
                    job_id=job_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    parent_call_id=parent_call_id,
                    prompt=input.prompt,
                    output_path=input.output_path,
                    aspect_ratio=input.aspect_ratio,
                )
            )
            return await future
        finally:
            self._pending_started.pop(job_id, None)

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

    async def recover_interrupted_jobs(self) -> list[ImageJobRecord]:
        """Mark any 'running' jobs left over from a crashed process as failed.

        Background tasks live only in memory, so a record stuck on 'running'
        after restart means no one will ever complete it. Emit a failure event
        so subscribers see the lifecycle close.
        """
        stuck = await self._store.list_running()
        recovered: list[ImageJobRecord] = []
        for record in stuck:
            await self._bus.publish(
                ImageJobFailed(
                    job_id=record.id,
                    user_id=record.user_id,
                    conversation_id=record.conversation_id,
                    error="interrupted by process restart",
                )
            )
            recovered.append(record)
        return recovered

    # ------------------------- event handlers -------------------------

    async def handle_requested(self, event: ImageJobRequested) -> EventBatch:
        record = await self._store.insert(
            job_id=event.job_id,
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            parent_call_id=event.parent_call_id,
            prompt=event.prompt,
            output_path=event.output_path,
            aspect_ratio=event.aspect_ratio,
        )
        return (
            ImageJobStarted(
                job_id=record.id,
                user_id=record.user_id,
                conversation_id=record.conversation_id,
                parent_call_id=record.parent_call_id,
                prompt=record.prompt,
                output_path=record.output_path,
                aspect_ratio=record.aspect_ratio,
            ),
        )

    async def handle_started(self, event: ImageJobStarted) -> EventBatch:
        record = await self._store.get(
            job_id=event.job_id,
            user_id=event.user_id,
            conversation_id=event.conversation_id,
        )
        future = self._pending_started.get(event.job_id)
        if future is not None and not future.done() and record is not None:
            future.set_result(record)
        self._tasks[event.job_id] = asyncio.create_task(self._run(event))
        return ()

    async def handle_completed(self, event: ImageJobCompleted) -> EventBatch:
        await self._store.complete(
            job_id=event.job_id,
            mime_type=event.mime_type,
            size_bytes=event.size_bytes,
        )
        return ()

    async def handle_failed(self, event: ImageJobFailed) -> EventBatch:
        await self._store.fail(job_id=event.job_id, error=event.error)
        return ()

    # ------------------------- background work -------------------------

    async def _run(self, started: ImageJobStarted) -> None:
        try:
            input = ImageGenerateInput(
                prompt=started.prompt,
                output_path=started.output_path,
                aspect_ratio=_coerce_aspect_ratio(started.aspect_ratio),
            )
            await self._run_to_terminal(started, input)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("image job {} failed", started.job_id)
            await self._bus.publish(
                ImageJobFailed(
                    job_id=started.job_id,
                    user_id=started.user_id,
                    conversation_id=started.conversation_id,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            )
        finally:
            self._tasks.pop(started.job_id, None)

    async def _run_to_terminal(
        self,
        started: ImageJobStarted,
        input: ImageGenerateInput,
    ) -> None:
        try:
            image = await self._generator.generate(input)
        except ImageGenerationError as exc:
            await self._bus.publish(
                ImageJobFailed(
                    job_id=started.job_id,
                    user_id=started.user_id,
                    conversation_id=started.conversation_id,
                    error=str(exc),
                )
            )
            return
        write = await self._runtime.write_content_file(
            started.user_id,
            started.output_path,
            image.data,
        )
        if write.exit_code != 0:
            await self._bus.publish(
                ImageJobFailed(
                    job_id=started.job_id,
                    user_id=started.user_id,
                    conversation_id=started.conversation_id,
                    error=write.stderr or f"failed to write {started.output_path}",
                )
            )
            return
        await self._bus.publish(
            ImageJobCompleted(
                job_id=started.job_id,
                user_id=started.user_id,
                conversation_id=started.conversation_id,
                output_path=started.output_path,
                mime_type=image.mime_type,
                size_bytes=len(image.data),
            )
        )


class ImageDeliveryHandler:
    """Subscribes to ImageJobCompleted and surfaces the rendered image into the
    conversation context as a multimodal user message with a ContentRef.

    The image is already on disk under output_path (written by the service's
    background task). We re-read those bytes once here to build the ContentRef
    and append a synthetic user-style message to the projection. The next agent
    turn for this conversation will see the image in its history without the
    model having to call image.status.
    """

    def __init__(
        self,
        *,
        runtime: UserRuntime,
        projection: SQLiteConversationProjection,
    ) -> None:
        self._runtime = runtime
        self._projection = projection

    async def handle_completed(self, event: ImageJobCompleted) -> EventBatch:
        read = await self._runtime.read_file_bytes(
            event.user_id, event.output_path, None
        )
        if read.result.exit_code != 0 or read.file is None:
            logger.warning(
                "image job {} completed but file {} is unreadable: {}",
                event.job_id,
                event.output_path,
                read.result.stderr,
            )
            return ()
        content_ref = content_ref_from_bytes(
            kind="image",
            file_name=_basename(event.output_path),
            mime_type=event.mime_type,
            workspace_path=event.output_path,
            content=read.file.content,
        )
        await self._projection.append_user_message(
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            text=(
                f"[image.generate job {event.job_id} completed: "
                f"{event.mime_type} saved to {event.output_path}]"
            ),
            attachments=[content_ref],
        )
        return ()


def render_image_job_record(record: ImageJobRecord) -> str:
    return record.model_dump_json(indent=2)


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1] or path


_VALID_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
}


def _coerce_aspect_ratio(value: str) -> Any:
    # ImageJobStarted carries aspect_ratio as a plain string; ImageGenerateInput
    # has a Literal. Validate here rather than re-deriving the union elsewhere.
    if value not in _VALID_ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect_ratio: {value}")
    return value


def _mint_job_id() -> str:
    from uuid import uuid4

    return uuid4().hex


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
