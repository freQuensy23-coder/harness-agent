"""Schedule tool handlers for `schedule.*` calls."""

from datetime import UTC, datetime

from harness_agent.events import ToolCallRequested
from harness_agent.runtime import RuntimeToolResult
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.tools import (
    ScheduleCancelInput,
    ScheduleCronInput,
    ScheduleListInput,
    ScheduleOnceInput,
)


class ScheduleToolHandlers:
    """Handlers backed by `SQLiteScheduleStore` for `schedule.*` tools."""

    def __init__(self, *, schedule_store: SQLiteScheduleStore) -> None:
        self._schedule_store = schedule_store

    async def once(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleOnceInput.model_validate(event.input)
        run_at_utc = None
        if input.run_at_utc is not None:
            run_at_utc = datetime.fromisoformat(input.run_at_utc).astimezone(UTC)
        schedule = await self._schedule_store.create_once(
            user_id=user_id,
            conversation_id=event.conversation_id,
            message=input.message,
            reply_target=event.reply_target,
            delay_seconds=input.delay_seconds,
            run_at_utc=run_at_utc,
        )
        return RuntimeToolResult(stdout=schedule.model_dump_json(indent=2))

    async def cron(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleCronInput.model_validate(event.input)
        schedule = await self._schedule_store.create_cron(
            user_id=user_id,
            conversation_id=event.conversation_id,
            message=input.message,
            reply_target=event.reply_target,
            cron=input.cron,
            timezone=input.timezone,
        )
        return RuntimeToolResult(stdout=schedule.model_dump_json(indent=2))

    async def list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleListInput.model_validate(event.input)
        schedules = await self._schedule_store.list_for_conversation(
            user_id=user_id,
            conversation_id=event.conversation_id,
            include_stopped=input.include_stopped,
        )
        return RuntimeToolResult(
            stdout="[\n"
            + ",\n".join(schedule.model_dump_json(indent=2) for schedule in schedules)
            + "\n]"
        )

    async def cancel(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleCancelInput.model_validate(event.input)
        schedule = await self._schedule_store.cancel(
            schedule_id=input.schedule_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
        )
        return RuntimeToolResult(stdout=schedule.model_dump_json(indent=2))
