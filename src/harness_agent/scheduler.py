import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import aiosqlite
from croniter import croniter
from pydantic import BaseModel, TypeAdapter

from harness_agent.bus import EventBus
from harness_agent.db import fetchall_rows
from harness_agent.events import EventBase, ReplyTarget, ScheduledMessageDue, UserTextReceived


ScheduleKind = Literal["once", "cron"]
ScheduleStatus = Literal["active", "completed", "cancelled"]


class ScheduledMessageBase(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    status: ScheduleStatus
    message: str
    next_run_at: datetime
    reply_target: ReplyTarget | None = None


class OnceScheduledMessage(ScheduledMessageBase):
    kind: Literal["once"] = "once"


class CronScheduledMessage(ScheduledMessageBase):
    kind: Literal["cron"] = "cron"
    cron: str
    timezone: str


ScheduledMessage = OnceScheduledMessage | CronScheduledMessage


class SQLiteScheduleStore:
    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = path
        self._now = utc_now if now is None else now
        self._reply_target_adapter: TypeAdapter[ReplyTarget] = TypeAdapter(ReplyTarget)

    async def create_once(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        reply_target: ReplyTarget | None,
        delay_seconds: int | None = None,
        run_at_utc: datetime | None = None,
    ) -> ScheduledMessage:
        if run_at_utc is None:
            delay = 0 if delay_seconds is None else delay_seconds
            run_at_utc = self._now() + timedelta(seconds=delay)
        return await self._insert(
            user_id=user_id,
            conversation_id=conversation_id,
            kind="once",
            message=message,
            next_run_at=run_at_utc.astimezone(UTC),
            reply_target=reply_target,
            cron=None,
            timezone=None,
        )

    async def create_cron(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
        reply_target: ReplyTarget | None,
        cron: str,
        timezone: str,
    ) -> ScheduledMessage:
        next_run_at = next_cron_run(
            cron=cron,
            timezone=timezone,
            after=self._now(),
        )
        return await self._insert(
            user_id=user_id,
            conversation_id=conversation_id,
            kind="cron",
            message=message,
            next_run_at=next_run_at,
            reply_target=reply_target,
            cron=cron,
            timezone=timezone,
        )

    async def list_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        include_stopped: bool = True,
    ) -> list[ScheduledMessage]:
        await self._ensure_schema()
        status_clause = "" if include_stopped else "and status = 'active'"
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                f"""
                select
                    id,
                    user_id,
                    conversation_id,
                    kind,
                    status,
                    message,
                    next_run_at,
                    reply_target_json,
                    cron,
                    timezone
                from scheduled_messages
                where user_id = ?
                  and conversation_id = ?
                  {status_clause}
                order by created_at asc
                """,
                (user_id, conversation_id),
            )
        return [self._row_to_schedule(row) for row in rows]

    async def get(self, schedule_id: str) -> ScheduledMessage:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select
                    id,
                    user_id,
                    conversation_id,
                    kind,
                    status,
                    message,
                    next_run_at,
                    reply_target_json,
                    cron,
                    timezone
                from scheduled_messages
                where id = ?
                """,
                (schedule_id,),
            )
        if not rows:
            raise KeyError(schedule_id)
        return self._row_to_schedule(rows[0])

    async def cancel(
        self,
        *,
        schedule_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ScheduledMessage:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update scheduled_messages
                set status = 'cancelled'
                where id = ?
                  and user_id = ?
                  and conversation_id = ?
                """,
                (schedule_id, user_id, conversation_id),
            )
            await db.commit()
        return await self.get(schedule_id)

    async def claim_due(self, now: datetime) -> list[ScheduledMessage]:
        await self._ensure_schema()
        now = now.astimezone(UTC)
        async with aiosqlite.connect(self._path) as db:
            await db.execute("begin immediate")
            rows = await fetchall_rows(
                db,
                """
                select
                    id,
                    user_id,
                    conversation_id,
                    kind,
                    status,
                    message,
                    next_run_at,
                    reply_target_json,
                    cron,
                    timezone
                from scheduled_messages
                where status = 'active'
                  and next_run_at <= ?
                order by next_run_at asc
                """,
                (now.isoformat(),),
            )
            schedules = [self._row_to_schedule(row) for row in rows]
            for schedule in schedules:
                if schedule.kind == "once":
                    await db.execute(
                        """
                        update scheduled_messages
                        set status = 'completed'
                        where id = ?
                        """,
                        (schedule.id,),
                    )
                else:
                    await db.execute(
                        """
                        update scheduled_messages
                        set next_run_at = ?
                        where id = ?
                        """,
                        (
                            next_cron_run(
                                cron=schedule.cron,
                                timezone=schedule.timezone,
                                after=now,
                            ).isoformat(),
                            schedule.id,
                        ),
                    )
            await db.commit()
        return schedules

    async def _insert(
        self,
        *,
        user_id: str,
        conversation_id: str,
        kind: ScheduleKind,
        message: str,
        next_run_at: datetime,
        reply_target: ReplyTarget | None,
        cron: str | None,
        timezone: str | None,
    ) -> ScheduledMessage:
        await self._ensure_schema()
        schedule_id = uuid4().hex
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into scheduled_messages (
                    id,
                    user_id,
                    conversation_id,
                    kind,
                    status,
                    message,
                    next_run_at,
                    reply_target_json,
                    cron,
                    timezone,
                    created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule_id,
                    user_id,
                    conversation_id,
                    kind,
                    "active",
                    message,
                    next_run_at.astimezone(UTC).isoformat(),
                    reply_target.model_dump_json() if reply_target is not None else None,
                    cron,
                    timezone,
                    self._now().astimezone(UTC).isoformat(),
                ),
            )
            await db.commit()
        return await self.get(schedule_id)

    def _row_to_schedule(self, row: tuple[Any, ...]) -> ScheduledMessage:
        reply_target: ReplyTarget | None = None
        if row[7] is not None:
            reply_target = self._reply_target_adapter.validate_python(json.loads(row[7]))
        kind = cast(ScheduleKind, row[3])
        status = cast(ScheduleStatus, row[4])
        next_run_at = datetime.fromisoformat(str(row[6])).astimezone(UTC)
        if kind == "once":
            return OnceScheduledMessage(
                id=str(row[0]),
                user_id=str(row[1]),
                conversation_id=str(row[2]),
                status=status,
                message=str(row[5]),
                next_run_at=next_run_at,
                reply_target=reply_target,
            )
        if kind == "cron":
            if row[8] is None or row[9] is None:
                raise RuntimeError(f"cron schedule {row[0]} is missing cron metadata")
            return CronScheduledMessage(
                id=str(row[0]),
                user_id=str(row[1]),
                conversation_id=str(row[2]),
                status=status,
                message=str(row[5]),
                next_run_at=next_run_at,
                reply_target=reply_target,
                cron=str(row[8]),
                timezone=str(row[9]),
            )
        raise ValueError(f"unknown schedule kind: {kind}")

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists scheduled_messages (
                    id text primary key,
                    user_id text not null,
                    conversation_id text not null,
                    kind text not null,
                    status text not null,
                    message text not null,
                    next_run_at text not null,
                    reply_target_json text,
                    cron text,
                    timezone text,
                    created_at text not null
                )
                """
            )
            await db.commit()


class SchedulerDueHandler:
    async def handle_due(self, event: ScheduledMessageDue) -> tuple[EventBase, ...]:
        return (
            UserTextReceived(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                source="scheduler",
                text=event.text,
                reply_target=event.reply_target,
            ),
        )


class SchedulerPump:
    def __init__(
        self,
        *,
        store: SQLiteScheduleStore,
        bus: EventBus,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._now = utc_now if now is None else now

    async def tick(self) -> None:
        schedules = await self._store.claim_due(self._now())
        for schedule in schedules:
            await self._bus.publish(
                ScheduledMessageDue(
                    schedule_id=schedule.id,
                    user_id=schedule.user_id,
                    conversation_id=schedule.conversation_id,
                    text=schedule.message,
                    reply_target=schedule.reply_target,
                )
            )


class SchedulerService:
    def __init__(self, *, pump: SchedulerPump, poll_seconds: float) -> None:
        self._pump = pump
        self._poll_seconds = poll_seconds
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        task = self._task
        self._task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            await self._pump.tick()
            await asyncio.sleep(self._poll_seconds)


def next_cron_run(*, cron: str, timezone: str, after: datetime) -> datetime:
    zone = ZoneInfo(timezone)
    local_after = after.astimezone(zone)
    return croniter(cron, local_after).get_next(datetime).astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)
