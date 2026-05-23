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


class PendingDue(BaseModel):
    """One entry in the durable outbox: state has already been advanced
    in `scheduled_messages` (one-shot → completed, cron → next_run_at
    shifted), but the corresponding `ScheduledMessageDue` event has
    not yet been published. The pump drains these on each tick."""
    due_id: str
    schedule_id: str
    user_id: str
    conversation_id: str
    text: str
    reply_target: ReplyTarget | None = None


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
            cursor = await db.execute(
                """
                update scheduled_messages
                set status = 'cancelled'
                where id = ?
                  and user_id = ?
                  and conversation_id = ?
                """,
                (schedule_id, user_id, conversation_id),
            )
            if cursor.rowcount == 0:
                await db.rollback()
                raise KeyError(schedule_id)
            await db.commit()
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
                  and user_id = ?
                  and conversation_id = ?
                """,
                (schedule_id, user_id, conversation_id),
            )
        if not rows:
            raise KeyError(schedule_id)
        return self._row_to_schedule(rows[0])

    async def claim_due(self, now: datetime) -> list[ScheduledMessage]:
        """Advance the state of every due schedule AND write a pending due
        entry to the outbox — atomically in one SQLite transaction. The
        pump publishes from the outbox afterwards; if the process crashes
        between this commit and the publish, the outbox row survives and
        the next tick will re-publish it."""
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
                await db.execute(
                    """
                    insert into scheduled_due_outbox (
                        id,
                        schedule_id,
                        user_id,
                        conversation_id,
                        text,
                        reply_target_json,
                        created_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        schedule.id,
                        schedule.user_id,
                        schedule.conversation_id,
                        schedule.message,
                        schedule.reply_target.model_dump_json()
                        if schedule.reply_target is not None
                        else None,
                        now.isoformat(),
                    ),
                )
            await db.commit()
        return schedules

    async def list_pending_due(self) -> list[PendingDue]:
        """Read all outbox rows in creation order. Used by the pump to
        re-publish anything that the previous tick committed but did not
        get to publish (process crash, restart, etc.)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await fetchall_rows(
                db,
                """
                select id, schedule_id, user_id, conversation_id, text, reply_target_json
                from scheduled_due_outbox
                order by created_at asc, id asc
                """,
                (),
            )
        out: list[PendingDue] = []
        for row in rows:
            reply_target: ReplyTarget | None = None
            if row[5] is not None:
                reply_target = self._reply_target_adapter.validate_python(json.loads(row[5]))
            out.append(
                PendingDue(
                    due_id=row[0],
                    schedule_id=row[1],
                    user_id=row[2],
                    conversation_id=row[3],
                    text=row[4],
                    reply_target=reply_target,
                )
            )
        return out

    async def mark_due_published(self, due_id: str) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "delete from scheduled_due_outbox where id = ?",
                (due_id,),
            )
            await db.commit()

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
            await db.execute(
                """
                create table if not exists scheduled_due_outbox (
                    id text primary key,
                    schedule_id text not null,
                    user_id text not null,
                    conversation_id text not null,
                    text text not null,
                    reply_target_json text,
                    created_at text not null
                )
                """
            )
            await db.commit()


class SchedulerDueHandler:
    async def handle_due(self, event: ScheduledMessageDue) -> tuple[EventBase, ...]:
        # Deterministic UserTextReceived id derived from the outbox-owned
        # ScheduledMessageDue id so that an idempotent_replay re-dispatch
        # produces the same downstream event — the event store rejects the
        # duplicate, the bus dispatches its handlers anyway, and the user
        # ends up with exactly one synthetic message even if the previous
        # attempt crashed between store.append and handler dispatch.
        return (
            UserTextReceived(
                id=f"scheduler-due:{event.id}",
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
        await self._store.claim_due(self._now())
        for pending in await self._store.list_pending_due():
            event = ScheduledMessageDue(
                id=pending.due_id,
                schedule_id=pending.schedule_id,
                user_id=pending.user_id,
                conversation_id=pending.conversation_id,
                text=pending.text,
                reply_target=pending.reply_target,
            )
            # idempotent_replay re-dispatches handlers even when the
            # event id is already in the store, so a previous attempt
            # that crashed between store.append and handler dispatch
            # still delivers UserTextReceived on retry. The downstream
            # SchedulerDueHandler keys UserTextReceived's id on this
            # event's id so the cascade is also append-idempotent.
            await self._bus.publish(event, idempotent_replay=True)
            await self._store.mark_due_published(pending.due_id)


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
        self._task = None

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
