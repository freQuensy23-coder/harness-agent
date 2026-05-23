"""Task tool handlers for `task.*` calls."""

from harness_agent.events import ToolCallRequested
from harness_agent.runtime import RuntimeToolResult
from harness_agent.tasks import SQLiteTaskStore, tasks_to_json
from harness_agent.tools import (
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskStopInput,
    TaskUpdateInput,
)


class TaskToolHandlers:
    """Handlers backed by `SQLiteTaskStore` for the `task.*` tool family."""

    def __init__(self, *, task_store: SQLiteTaskStore) -> None:
        self._task_store = task_store

    async def create(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskCreateInput.model_validate(event.input)
        task = await self._task_store.create(
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=input.title,
            status=input.status,
        )
        return RuntimeToolResult(stdout=task.model_dump_json(indent=2))

    async def get(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskGetInput.model_validate(event.input)
        task = await self._task_store.get(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
        )
        if task is None:
            raise KeyError(input.task_id)
        return RuntimeToolResult(stdout=task.model_dump_json(indent=2))

    async def list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskListInput.model_validate(event.input)
        tasks = await self._task_store.list(
            user_id=user_id,
            conversation_id=event.conversation_id,
            include_stopped=input.include_stopped,
        )
        return RuntimeToolResult(stdout=tasks_to_json(tasks))

    async def update(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskUpdateInput.model_validate(event.input)
        task = await self._task_store.update(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=input.title,
            status=input.status,
        )
        if task is None:
            raise KeyError(input.task_id)
        return RuntimeToolResult(stdout=task.model_dump_json(indent=2))

    async def stop(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskStopInput.model_validate(event.input)
        task = await self._task_store.update(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=None,
            status="stopped",
        )
        if task is None:
            raise KeyError(input.task_id)
        return RuntimeToolResult(stdout=task.model_dump_json(indent=2))
