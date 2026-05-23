"""Router that dispatches `ToolCallRequested` events to per-domain handlers."""

from collections.abc import Awaitable, Callable

from harness_agent.content import content_ref_from_workspace_file
from harness_agent.events import ToolCallCompleted, ToolCallError, ToolCallRequested
from harness_agent.image_jobs import ImageJobService
from harness_agent.mcp import McpManager
from harness_agent.memory_service import MemoryService
from harness_agent.runtime import RuntimeToolResult, UserRuntime
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.session_search_service import SessionSearchService
from harness_agent.subagents import SubAgentService
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_call_results import ToolExecutionResult
from harness_agent.tool_call_waiter import (
    EventBatch,
    ToolCallResultWaiter,
    tool_result_spill_path,
)
from harness_agent.tool_image_handlers import ImageToolHandlers
from harness_agent.tool_schedule_handlers import ScheduleToolHandlers
from harness_agent.tool_sub_agent_handlers import SubAgentToolHandlers
from harness_agent.tool_task_handlers import TaskToolHandlers
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    McpToolInput,
    MemoryToolInput,
    SessionSearchInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
    SkillListInput,
    SkillReadInput,
    ToolRegistry,
    WebFetchInput,
    default_tool_registry,
)
from harness_agent.web_fetch import WebFetcher


# Re-exports preserved so existing call sites keep working.
__all__ = ["ToolCallExecutor", "ToolCallResultWaiter", "ToolExecutionResult"]


ToolExecutor = Callable[[str, ToolCallRequested], Awaitable[RuntimeToolResult]]


class ToolCallExecutor:
    """Routes tool calls to runtime, shell/file, web, image, MCP, and domain handlers."""

    def __init__(
        self,
        *,
        runtime: UserRuntime,
        memory_service: MemoryService,
        session_search: SessionSearchService,
        task_store: SQLiteTaskStore | None = None,
        schedule_store: SQLiteScheduleStore | None = None,
        web_fetcher: WebFetcher | None = None,
        image_jobs: ImageJobService | None = None,
        mcp_manager: McpManager | None = None,
        sub_agents: SubAgentService | None = None,
        max_model_output_chars: int = 20_000,
    ) -> None:
        self._runtime = runtime
        self._web_fetcher = web_fetcher
        self._mcp_manager = mcp_manager
        self._memory_service = memory_service
        self._session_search_service = session_search
        self._max_model_output_chars = max_model_output_chars
        self._image_handlers: ImageToolHandlers | None = (
            ImageToolHandlers(runtime=runtime, image_jobs=image_jobs)
            if image_jobs is not None
            else None
        )
        self._tool_executors: dict[str, ToolExecutor] = {
            "shell.exec": self._shell_exec,
            "shell.spawn": self._shell_spawn,
            "shell.read": self._shell_read,
            "shell.kill": self._shell_kill,
            "file.write": self._file_write,
            "file.edit": self._file_edit,
            "file.multi_edit": self._file_multi_edit,
            "file.glob": self._file_glob,
            "file.grep": self._file_grep,
            "file.list": self._file_list,
            "skill.list": self._skill_list,
            "skill.read": self._skill_read,
            "memory": self._memory,
            "session.search": self._session_search,
        }
        if self._web_fetcher is not None:
            self._tool_executors["web.fetch"] = self._web_fetch
        if self._image_handlers is not None:
            self._tool_executors["image.generate"] = self._image_handlers.generate
        if task_store is not None:
            tasks = TaskToolHandlers(task_store=task_store)
            self._tool_executors.update(
                {
                    "task.create": tasks.create,
                    "task.get": tasks.get,
                    "task.list": tasks.list,
                    "task.update": tasks.update,
                    "task.stop": tasks.stop,
                }
            )
        if schedule_store is not None:
            schedules = ScheduleToolHandlers(schedule_store=schedule_store)
            self._tool_executors.update(
                {
                    "schedule.once": schedules.once,
                    "schedule.cron": schedules.cron,
                    "schedule.list": schedules.list,
                    "schedule.cancel": schedules.cancel,
                }
            )
        if sub_agents is not None:
            agents = SubAgentToolHandlers(sub_agents=sub_agents)
            self._tool_executors.update(
                {
                    "agent.run": agents.run,
                    "agent.spawn": agents.spawn,
                    "agent.result": agents.result,
                    "agent.list": agents.list,
                    "agent.cancel": agents.cancel,
                }
            )
        # `tool_registry()` advertises tools only when their dep is wired.
        self._task_store = task_store
        self._schedule_store = schedule_store
        self._image_jobs = image_jobs
        self._sub_agents = sub_agents

    def tool_registry(self) -> ToolRegistry:
        return default_tool_registry(
            include_web_fetch=self._web_fetcher is not None,
            include_image_generation=self._image_jobs is not None,
            include_tasks=self._task_store is not None,
            include_schedules=self._schedule_store is not None,
            include_agents=self._sub_agents is not None,
        )

    async def handle_tool_call_requested(self, event: ToolCallRequested) -> EventBatch:
        try:
            execution = await self._execute_tool(event)
            execution = await self._spill_oversized_result(event, execution)
        except Exception as exc:
            error = str(exc)
            execution = ToolExecutionResult(
                result=RuntimeToolResult(stderr=error, exit_code=1)
            )
            return (
                ToolCallError(
                    user_id=event.user_id,
                    conversation_id=event.conversation_id,
                    generation=event.generation,
                    call_id=event.call_id,
                    tool_name=event.tool_name,
                    input=event.input,
                    error=error,
                    reply_target=event.reply_target,
                ),
                ToolCallCompleted(
                    user_id=event.user_id,
                    conversation_id=event.conversation_id,
                    generation=event.generation,
                    call_id=event.call_id,
                    tool_name=event.tool_name,
                    input=event.input,
                    result=execution.result,
                    attachments=execution.attachments,
                    reply_target=event.reply_target,
                ),
            )
        return (
            ToolCallCompleted(
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                generation=event.generation,
                call_id=event.call_id,
                tool_name=event.tool_name,
                input=event.input,
                result=execution.result,
                attachments=execution.attachments,
                reply_target=event.reply_target,
            ),
        )

    async def _execute_tool(self, event: ToolCallRequested) -> ToolExecutionResult:
        if event.tool_name == "file.read":
            return await self._file_read_for_model(event.user_id, event)
        if event.tool_name == "image.status":
            if self._image_handlers is None:
                raise RuntimeError("image job service is not configured")
            return await self._image_handlers.status(event.user_id, event)
        if event.tool_name.startswith("mcp."):
            return ToolExecutionResult(result=await self._mcp_call(event))
        if event.tool_name not in self._tool_executors:
            raise ValueError(f"Unsupported tool call: {event.tool_name}")
        return ToolExecutionResult(
            result=await self._tool_executors[event.tool_name](event.user_id, event)
        )

    async def _spill_oversized_result(
        self,
        event: ToolCallRequested,
        execution: ToolExecutionResult,
    ) -> ToolExecutionResult:
        rendered = execution.result.render_for_llm(event.tool_name)
        if len(rendered) <= self._max_model_output_chars:
            return execution
        path = tool_result_spill_path(event)
        write = await self._runtime.write_content_file(
            event.user_id,
            path,
            rendered.encode("utf-8"),
        )
        if write.exit_code != 0:
            raise RuntimeError(write.stderr or f"failed to save oversized tool result to {path}")
        return ToolExecutionResult(
            result=RuntimeToolResult(
                stdout=(
                    "truncated, because it is too long. "
                    f"Full result saved to {path}. "
                    "You can read it in chunks with file.read."
                ),
                exit_code=execution.result.exit_code,
            ),
            attachments=execution.attachments,
        )

    async def _mcp_call(self, event: ToolCallRequested) -> RuntimeToolResult:
        if self._mcp_manager is None:
            raise RuntimeError("mcp manager is not configured")
        input = McpToolInput.model_validate(event.input)
        return await self._mcp_manager.call_tool(
            user_id=event.user_id,
            tool_name=event.tool_name,
            arguments=input.arguments,
        )

    async def _shell_exec(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.shell_exec(user_id, ShellExecInput.model_validate(event.input))

    async def _shell_spawn(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.shell_spawn(user_id, ShellSpawnInput.model_validate(event.input))

    async def _shell_read(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.shell_read(user_id, ShellReadInput.model_validate(event.input))

    async def _shell_kill(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.shell_kill(user_id, ShellKillInput.model_validate(event.input))

    async def _file_read_for_model(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> ToolExecutionResult:
        input = FileReadInput.model_validate(event.input)
        read = await self._runtime.read_file_bytes(user_id, input.path, input.max_bytes)
        if read.result.exit_code != 0:
            return ToolExecutionResult(result=read.result)
        if read.file is None:
            raise RuntimeError("file read succeeded without file content")
        workspace_file = read.file
        content_ref = content_ref_from_workspace_file(workspace_file)
        if content_ref.kind == "image":
            full_read = await self._runtime.read_file_bytes(user_id, input.path, None)
            if full_read.result.exit_code != 0:
                return ToolExecutionResult(result=full_read.result)
            if full_read.file is None:
                raise RuntimeError("image file read succeeded without file content")
            full_content_ref = content_ref_from_workspace_file(full_read.file)
            return ToolExecutionResult(
                result=RuntimeToolResult(
                    stdout=f"Opened image file {full_content_ref.workspace_path}\n"
                ),
                attachments=[full_content_ref],
            )
        return ToolExecutionResult(
            result=RuntimeToolResult(
                stdout=workspace_file.content.decode("utf-8", errors="replace")
            )
        )

    async def _file_write(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_write(user_id, FileWriteInput.model_validate(event.input))

    async def _file_edit(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_edit(user_id, FileEditInput.model_validate(event.input))

    async def _file_multi_edit(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_multi_edit(
            user_id, FileMultiEditInput.model_validate(event.input)
        )

    async def _file_glob(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_glob(user_id, FileGlobInput.model_validate(event.input))

    async def _file_grep(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_grep(user_id, FileGrepInput.model_validate(event.input))

    async def _file_list(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._runtime.file_list(user_id, FileListInput.model_validate(event.input))

    async def _web_fetch(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        if self._web_fetcher is None:
            raise RuntimeError("web fetcher is not configured")
        return await self._web_fetcher.fetch(WebFetchInput.model_validate(event.input))

    async def _skill_list(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        input = SkillListInput.model_validate(event.input)
        skills = await self._runtime.list_skills(user_id)
        if input.include_descriptions:
            text = "\n".join(f"{skill.name}: {skill.description}" for skill in skills)
        else:
            text = "\n".join(skill.name for skill in skills)
        return RuntimeToolResult(stdout=text)

    async def _skill_read(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        input = SkillReadInput.model_validate(event.input)
        skills = await self._runtime.list_skills(user_id)
        for skill in skills:
            if skill.name == input.name:
                return RuntimeToolResult(stdout=skill.render_for_prompt())
        raise KeyError(input.name)

    async def _memory(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._memory_service.execute(
            user_id, MemoryToolInput.model_validate(event.input)
        )

    async def _session_search(self, user_id: str, event: ToolCallRequested) -> RuntimeToolResult:
        return await self._session_search_service.execute(
            user_id=user_id,
            current_conversation_id=event.conversation_id,
            input=SessionSearchInput.model_validate(event.input),
        )
