import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from harness_agent.browser_use import (
    BrowserUseService,
    render_browser_session,
    render_browser_sessions,
)
from harness_agent.content import ContentRef, content_ref_from_workspace_file
from harness_agent.events import EventBase, ToolCallCompleted, ToolCallRequested
from harness_agent.mcp import McpManager
from harness_agent.runtime import RuntimeToolResult, UserRuntime
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.subagents import (
    SubAgentService,
    render_sub_agent_record,
    render_sub_agent_records,
)
from harness_agent.tasks import SQLiteTaskStore, tasks_to_json
from harness_agent.tools import (
    AgentCancelInput,
    AgentListInput,
    AgentResultInput,
    AgentRunInput,
    AgentSpawnInput,
    BrowserGetInput,
    BrowserListInput,
    BrowserRunInput,
    BrowserSendInput,
    BrowserSpawnInput,
    BrowserStopInput,
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    McpToolInput,
    ScheduleCancelInput,
    ScheduleCronInput,
    ScheduleListInput,
    ScheduleOnceInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
    SkillListInput,
    SkillReadInput,
    TaskCreateInput,
    TaskGetInput,
    TaskListInput,
    TaskStopInput,
    TaskUpdateInput,
    WebFetchInput,
)
from harness_agent.web_fetch import WebFetcher


EventBatch = tuple[EventBase, ...]
ToolExecutor = Callable[[str, ToolCallRequested], Awaitable[RuntimeToolResult]]
ToolCallKey = tuple[str, int, str]


class ToolExecutionResult(BaseModel):
    result: RuntimeToolResult
    attachments: list[ContentRef] = Field(default_factory=list)


class ToolCallResultWaiter:
    def __init__(self) -> None:
        self._pending: dict[ToolCallKey, asyncio.Future[ToolCallCompleted]] = {}

    def expect(self, event: ToolCallRequested) -> None:
        self._pending[_tool_call_key(event)] = asyncio.get_running_loop().create_future()

    async def wait(self, event: ToolCallRequested) -> ToolCallCompleted:
        key = _tool_call_key(event)
        future = self._pending[key]
        try:
            return await future
        finally:
            del self._pending[key]

    async def handle_tool_call_completed(self, event: ToolCallCompleted) -> EventBatch:
        key = _tool_call_key(event)
        future = self._pending.get(key)
        if future is not None and not future.done():
            future.set_result(event)
        return ()


class ToolCallExecutor:
    def __init__(
        self,
        *,
        runtime: UserRuntime,
        task_store: SQLiteTaskStore | None = None,
        schedule_store: SQLiteScheduleStore | None = None,
        web_fetcher: WebFetcher | None = None,
        mcp_manager: McpManager | None = None,
        sub_agents: SubAgentService | None = None,
        browser_use_service: BrowserUseService | None = None,
    ) -> None:
        self._runtime = runtime
        self._task_store = task_store
        self._schedule_store = schedule_store
        self._web_fetcher = web_fetcher
        self._mcp_manager = mcp_manager
        self._sub_agents = sub_agents
        self._browser_use = browser_use_service
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
            "web.fetch": self._web_fetch,
            "task.create": self._task_create,
            "task.get": self._task_get,
            "task.list": self._task_list,
            "task.update": self._task_update,
            "task.stop": self._task_stop,
            "schedule.once": self._schedule_once,
            "schedule.cron": self._schedule_cron,
            "schedule.list": self._schedule_list,
            "schedule.cancel": self._schedule_cancel,
            "skill.list": self._skill_list,
            "skill.read": self._skill_read,
            "agent.run": self._agent_run,
            "agent.spawn": self._agent_spawn,
            "agent.result": self._agent_result,
            "agent.list": self._agent_list,
            "agent.cancel": self._agent_cancel,
            "browser.run": self._browser_run,
            "browser.spawn": self._browser_spawn,
            "browser.get": self._browser_get,
            "browser.send": self._browser_send,
            "browser.stop": self._browser_stop,
            "browser.list": self._browser_list,
        }

    async def handle_tool_call_requested(self, event: ToolCallRequested) -> EventBatch:
        execution = await self._execute_tool(event)
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
            ),
        )

    async def _execute_tool(self, event: ToolCallRequested) -> ToolExecutionResult:
        if event.tool_name == "file.read":
            return await self._file_read_for_model(event.user_id, event)
        if event.tool_name not in self._tool_executors:
            if event.tool_name.startswith("mcp."):
                return ToolExecutionResult(result=await self._mcp_call(event))
            raise ValueError(f"Unsupported tool call: {event.tool_name}")
        return ToolExecutionResult(
            result=await self._tool_executors[event.tool_name](event.user_id, event)
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

    async def _shell_exec(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.shell_exec(user_id, ShellExecInput.model_validate(event.input))

    async def _shell_spawn(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.shell_spawn(user_id, ShellSpawnInput.model_validate(event.input))

    async def _shell_read(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.shell_read(user_id, ShellReadInput.model_validate(event.input))

    async def _shell_kill(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.shell_kill(user_id, ShellKillInput.model_validate(event.input))

    async def _file_read_for_model(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> ToolExecutionResult:
        input = FileReadInput.model_validate(event.input)
        read = await self._runtime.read_file_bytes(
            user_id,
            input.path,
            input.max_bytes,
        )
        if read.result.exit_code != 0:
            return ToolExecutionResult(result=read.result)
        if read.file is None:
            raise RuntimeError("file read succeeded without file content")
        workspace_file = read.file
        content_ref = content_ref_from_workspace_file(workspace_file)
        if content_ref.kind == "image":
            full_read = await self._runtime.read_file_bytes(
                user_id,
                input.path,
                None,
            )
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

    async def _file_write(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_write(user_id, FileWriteInput.model_validate(event.input))

    async def _file_edit(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_edit(user_id, FileEditInput.model_validate(event.input))

    async def _file_multi_edit(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_multi_edit(
            user_id,
            FileMultiEditInput.model_validate(event.input),
        )

    async def _file_glob(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_glob(user_id, FileGlobInput.model_validate(event.input))

    async def _file_grep(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_grep(user_id, FileGrepInput.model_validate(event.input))

    async def _file_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        return await self._runtime.file_list(user_id, FileListInput.model_validate(event.input))

    async def _web_fetch(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        if self._web_fetcher is None:
            raise RuntimeError("web fetcher is not configured")
        return await self._web_fetcher.fetch(WebFetchInput.model_validate(event.input))

    async def _task_create(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskCreateInput.model_validate(event.input)
        task = await self._require_task_store().create(
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=input.title,
            status=input.status,
        )
        return self._text_result(task.model_dump_json(indent=2))

    async def _task_get(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskGetInput.model_validate(event.input)
        task = await self._require_task_store().get(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
        )
        if task is None:
            raise KeyError(input.task_id)
        return self._text_result(task.model_dump_json(indent=2))

    async def _task_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskListInput.model_validate(event.input)
        tasks = await self._require_task_store().list(
            user_id=user_id,
            conversation_id=event.conversation_id,
            include_stopped=input.include_stopped,
        )
        return self._text_result(tasks_to_json(tasks))

    async def _task_update(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskUpdateInput.model_validate(event.input)
        task = await self._require_task_store().update(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=input.title,
            status=input.status,
        )
        if task is None:
            raise KeyError(input.task_id)
        return self._text_result(task.model_dump_json(indent=2))

    async def _task_stop(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = TaskStopInput.model_validate(event.input)
        task = await self._require_task_store().update(
            task_id=input.task_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
            title=None,
            status="stopped",
        )
        if task is None:
            raise KeyError(input.task_id)
        return self._text_result(task.model_dump_json(indent=2))

    async def _skill_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = SkillListInput.model_validate(event.input)
        skills = await self._runtime.list_skills(user_id)
        if input.include_descriptions:
            text = "\n".join(f"{skill.name}: {skill.description}" for skill in skills)
        else:
            text = "\n".join(skill.name for skill in skills)
        return self._text_result(text)

    async def _skill_read(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = SkillReadInput.model_validate(event.input)
        skills = await self._runtime.list_skills(user_id)
        for skill in skills:
            if skill.name == input.name:
                return self._text_result(skill.render_for_prompt())
        raise KeyError(input.name)

    async def _schedule_once(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleOnceInput.model_validate(event.input)
        run_at_utc = None
        if input.run_at_utc is not None:
            run_at_utc = datetime.fromisoformat(input.run_at_utc).astimezone(UTC)
        schedule = await self._require_schedule_store().create_once(
            user_id=user_id,
            conversation_id=event.conversation_id,
            message=input.message,
            reply_target=event.reply_target,
            delay_seconds=input.delay_seconds,
            run_at_utc=run_at_utc,
        )
        return self._text_result(schedule.model_dump_json(indent=2))

    async def _schedule_cron(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleCronInput.model_validate(event.input)
        schedule = await self._require_schedule_store().create_cron(
            user_id=user_id,
            conversation_id=event.conversation_id,
            message=input.message,
            reply_target=event.reply_target,
            cron=input.cron,
            timezone=input.timezone,
        )
        return self._text_result(schedule.model_dump_json(indent=2))

    async def _schedule_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleListInput.model_validate(event.input)
        schedules = await self._require_schedule_store().list_for_conversation(
            user_id=user_id,
            conversation_id=event.conversation_id,
            include_stopped=input.include_stopped,
        )
        return self._text_result(
            "[\n"
            + ",\n".join(schedule.model_dump_json(indent=2) for schedule in schedules)
            + "\n]"
        )

    async def _schedule_cancel(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = ScheduleCancelInput.model_validate(event.input)
        schedule = await self._require_schedule_store().cancel(
            schedule_id=input.schedule_id,
            user_id=user_id,
            conversation_id=event.conversation_id,
        )
        return self._text_result(schedule.model_dump_json(indent=2))

    async def _agent_run(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentRunInput.model_validate(event.input)
        record = await self._require_sub_agents().run(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            parent_call_id=event.call_id,
            input=input,
        )
        exit_code = 0 if record.status == "completed" else 1
        return RuntimeToolResult(
            stdout=render_sub_agent_record(record),
            stderr="" if record.error is None else record.error,
            exit_code=exit_code,
        )

    async def _agent_spawn(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentSpawnInput.model_validate(event.input)
        record = await self._require_sub_agents().spawn(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            parent_call_id=event.call_id,
            input=input,
        )
        return self._text_result(render_sub_agent_record(record))

    async def _agent_result(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentResultInput.model_validate(event.input)
        record = await self._require_sub_agents().result(
            agent_id=input.agent_id,
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
        )
        if record is None:
            return RuntimeToolResult(stderr=f"Unknown sub-agent: {input.agent_id}\n", exit_code=1)
        return self._text_result(render_sub_agent_record(record))

    async def _agent_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentListInput.model_validate(event.input)
        records = await self._require_sub_agents().list_for_parent(
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
            include_completed=input.include_completed,
        )
        return self._text_result(render_sub_agent_records(records))

    async def _agent_cancel(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = AgentCancelInput.model_validate(event.input)
        record = await self._require_sub_agents().cancel(
            agent_id=input.agent_id,
            user_id=user_id,
            parent_conversation_id=event.conversation_id,
        )
        if record is None:
            return RuntimeToolResult(stderr=f"Unknown sub-agent: {input.agent_id}\n", exit_code=1)
        return self._text_result(render_sub_agent_record(record))

    async def _browser_run(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserRunInput.model_validate(event.input)
        record = await self._require_browser_use().run(
            user_id=user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            parent_call_id=event.call_id,
            input=input,
        )
        exit_code = 0 if record.status == "completed" and record.error is None else 1
        return RuntimeToolResult(
            stdout=render_browser_session(record),
            stderr="" if record.error is None else record.error,
            exit_code=exit_code,
        )

    async def _browser_spawn(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserSpawnInput.model_validate(event.input)
        record = await self._require_browser_use().spawn(
            user_id=user_id,
            conversation_id=event.conversation_id,
            generation=event.generation,
            parent_call_id=event.call_id,
            input=input,
        )
        return self._text_result(render_browser_session(record))

    async def _browser_get(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserGetInput.model_validate(event.input)
        record, messages = await self._require_browser_use().get(
            user_id=user_id,
            input=input,
        )
        return self._text_result(
            render_browser_session(record, messages=messages if input.include_messages else None)
        )

    async def _browser_send(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserSendInput.model_validate(event.input)
        record = await self._require_browser_use().send(user_id=user_id, input=input)
        return self._text_result(render_browser_session(record))

    async def _browser_stop(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserStopInput.model_validate(event.input)
        record = await self._require_browser_use().stop(user_id=user_id, input=input)
        return self._text_result(render_browser_session(record))

    async def _browser_list(
        self,
        user_id: str,
        event: ToolCallRequested,
    ) -> RuntimeToolResult:
        input = BrowserListInput.model_validate(event.input)
        records = await self._require_browser_use().list_for_user(
            user_id=user_id,
            input=input,
        )
        return self._text_result(render_browser_sessions(records))

    def _require_task_store(self) -> SQLiteTaskStore:
        if self._task_store is None:
            raise RuntimeError("task store is not configured")
        return self._task_store

    def _require_schedule_store(self) -> SQLiteScheduleStore:
        if self._schedule_store is None:
            raise RuntimeError("schedule store is not configured")
        return self._schedule_store

    def _require_sub_agents(self) -> SubAgentService:
        if self._sub_agents is None:
            raise RuntimeError("sub-agent service is not configured")
        return self._sub_agents

    def _require_browser_use(self) -> BrowserUseService:
        if self._browser_use is None:
            raise RuntimeError("browser-use service is not configured")
        return self._browser_use

    def _text_result(self, text: str) -> RuntimeToolResult:
        return RuntimeToolResult(stdout=text)


def _tool_call_key(event: ToolCallRequested | ToolCallCompleted) -> ToolCallKey:
    return (event.conversation_id, event.generation, event.call_id)
