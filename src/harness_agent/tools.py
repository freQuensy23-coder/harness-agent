from typing import Any, Literal, cast

from pydantic import BaseModel, Field


class ShellExecInput(BaseModel):
    command: str
    cwd: str = "/workspace"
    timeout_seconds: int = 60


class ShellSpawnInput(BaseModel):
    command: str
    cwd: str = "/workspace"


class ShellReadInput(BaseModel):
    process_id: str
    max_bytes: int = 20000


class ShellKillInput(BaseModel):
    process_id: str


class FileReadInput(BaseModel):
    path: str
    max_bytes: int = 20000


class FileWriteInput(BaseModel):
    path: str
    content: str


class FileEditInput(BaseModel):
    path: str
    old: str
    new: str
    replace_all: bool = False


class FileEditOperation(BaseModel):
    old: str
    new: str
    replace_all: bool = False


class FileMultiEditInput(BaseModel):
    path: str
    edits: list[FileEditOperation] = Field(min_length=1)


class FileGlobInput(BaseModel):
    pattern: str
    cwd: str = "/workspace"


class FileGrepInput(BaseModel):
    pattern: str
    path: str = "/workspace"


class FileListInput(BaseModel):
    path: str = "/workspace"


class WebFetchInput(BaseModel):
    url: str
    prompt: str = Field(min_length=1)
    max_bytes: int = 20000


class TaskCreateInput(BaseModel):
    title: str
    status: Literal["pending", "in_progress", "completed", "stopped"] = "pending"


class TaskGetInput(BaseModel):
    task_id: str


class TaskListInput(BaseModel):
    include_stopped: bool = False


class TaskUpdateInput(BaseModel):
    task_id: str
    title: str | None = None
    status: Literal["pending", "in_progress", "completed", "stopped"] | None = None


class TaskStopInput(BaseModel):
    task_id: str


class ScheduleOnceInput(BaseModel):
    message: str
    run_at_utc: str | None = None
    delay_seconds: int | None = Field(default=None, ge=0)


class ScheduleCronInput(BaseModel):
    message: str
    cron: str
    timezone: str = "UTC"


class ScheduleListInput(BaseModel):
    include_stopped: bool = False


class ScheduleCancelInput(BaseModel):
    schedule_id: str


class SkillListInput(BaseModel):
    include_descriptions: bool = True


class SkillReadInput(BaseModel):
    name: str


class McpToolInput(BaseModel):
    arguments: dict[str, Any]


class MemoryToolInput(BaseModel):
    action: Literal["add", "replace", "remove"]
    target: Literal["memory", "user"] = "memory"
    content: str | None = None
    old_text: str | None = None


class SessionSearchInput(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=5)


class AgentRunInput(BaseModel):
    prompt: str
    name: str = "subagent"
    timeout_seconds: float = 300.0


class AgentSpawnInput(BaseModel):
    prompt: str
    name: str = "subagent"
    timeout_seconds: float = 300.0


class AgentResultInput(BaseModel):
    agent_id: str


class AgentCancelInput(BaseModel):
    agent_id: str


class AgentListInput(BaseModel):
    include_completed: bool = True


ToolInputModel = type[BaseModel]


TOOL_INPUT_MODELS: dict[str, ToolInputModel] = {
    "shell.exec": ShellExecInput,
    "shell.spawn": ShellSpawnInput,
    "shell.read": ShellReadInput,
    "shell.kill": ShellKillInput,
    "file.read": FileReadInput,
    "file.write": FileWriteInput,
    "file.edit": FileEditInput,
    "file.multi_edit": FileMultiEditInput,
    "file.glob": FileGlobInput,
    "file.grep": FileGrepInput,
    "file.list": FileListInput,
    "web.fetch": WebFetchInput,
    "task.create": TaskCreateInput,
    "task.get": TaskGetInput,
    "task.list": TaskListInput,
    "task.update": TaskUpdateInput,
    "task.stop": TaskStopInput,
    "schedule.once": ScheduleOnceInput,
    "schedule.cron": ScheduleCronInput,
    "schedule.list": ScheduleListInput,
    "schedule.cancel": ScheduleCancelInput,
    "skill.list": SkillListInput,
    "skill.read": SkillReadInput,
    "memory": MemoryToolInput,
    "session.search": SessionSearchInput,
    "agent.run": AgentRunInput,
    "agent.spawn": AgentSpawnInput,
    "agent.result": AgentResultInput,
    "agent.list": AgentListInput,
    "agent.cancel": AgentCancelInput,
}


class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel] | None = None
    input_schema: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def parameters_schema(self) -> dict[str, Any]:
        if self.input_schema is not None:
            return self.input_schema
        if self.input_model is None:
            raise ValueError(f"Tool {self.name} has no input schema")
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema


class ToolRegistry(BaseModel):
    tools: list[ToolSpec]

    model_config = {"arbitrary_types_allowed": True}

    def by_name(self, name: str) -> ToolSpec:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(name)

    def list_for_model(self) -> list[ToolSpec]:
        return self.tools

    def filter_tools(
        self,
        *,
        include_memory: bool = True,
        include_session_search: bool = True,
    ) -> "ToolRegistry":
        skipped: set[str] = set()
        if not include_memory:
            skipped.add("memory")
        if not include_session_search:
            skipped.add("session.search")
        if not skipped:
            return self
        return ToolRegistry(
            tools=[tool for tool in self.tools if tool.name not in skipped]
        )


ToolName = Literal[
    "shell.exec",
    "shell.spawn",
    "shell.read",
    "shell.kill",
    "file.read",
    "file.write",
    "file.edit",
    "file.multi_edit",
    "file.glob",
    "file.grep",
    "file.list",
    "web.fetch",
    "task.create",
    "task.get",
    "task.list",
    "task.update",
    "task.stop",
    "schedule.once",
    "schedule.cron",
    "schedule.list",
    "schedule.cancel",
    "skill.list",
    "skill.read",
    "memory",
    "session.search",
    "agent.run",
    "agent.spawn",
    "agent.result",
    "agent.list",
    "agent.cancel",
]
ToolInput = (
    ShellExecInput
    | ShellSpawnInput
    | ShellReadInput
    | ShellKillInput
    | FileReadInput
    | FileWriteInput
    | FileEditInput
    | FileMultiEditInput
    | FileGlobInput
    | FileGrepInput
    | FileListInput
    | WebFetchInput
    | TaskCreateInput
    | TaskGetInput
    | TaskListInput
    | TaskUpdateInput
    | TaskStopInput
    | ScheduleOnceInput
    | ScheduleCronInput
    | ScheduleListInput
    | ScheduleCancelInput
    | SkillListInput
    | SkillReadInput
    | MemoryToolInput
    | SessionSearchInput
    | AgentRunInput
    | AgentSpawnInput
    | AgentResultInput
    | AgentListInput
    | AgentCancelInput
    | McpToolInput
)


def parse_llm_tool_input(name: str, arguments: Any) -> ToolInput:
    if name.startswith("mcp."):
        return McpToolInput(arguments=arguments)
    return parse_known_tool_input(name, arguments)


def parse_stored_tool_input(name: str, payload: Any) -> ToolInput:
    if name.startswith("mcp."):
        return McpToolInput.model_validate(payload)
    return parse_known_tool_input(name, payload)


def parse_known_tool_input(name: str, payload: Any) -> ToolInput:
    input_model = TOOL_INPUT_MODELS.get(name)
    if input_model is None:
        raise ValueError(f"unknown tool: {name}")
    return cast(ToolInput, input_model.model_validate(payload))


def default_tool_registry(
    *,
    include_memory: bool = True,
    include_session_search: bool = True,
) -> ToolRegistry:
    """Build the registry of tools exposed to the model.

    `memory` and `session.search` are conditionally included so callers
    that have no MemoryService / SessionSearchService configured can
    omit them. The two flags are independent — exposing one without the
    other is supported.
    """
    return ToolRegistry(
        tools=[
            ToolSpec(
                name="shell.exec",
                description="Run a shell command in the user's workspace.",
                input_model=ShellExecInput,
            ),
            ToolSpec(
                name="shell.spawn",
                description="Start a long-running shell command in the user's workspace.",
                input_model=ShellSpawnInput,
            ),
            ToolSpec(
                name="shell.read",
                description="Read buffered output from a spawned shell command.",
                input_model=ShellReadInput,
            ),
            ToolSpec(
                name="shell.kill",
                description="Stop a spawned shell command.",
                input_model=ShellKillInput,
            ),
            ToolSpec(
                name="file.read",
                description="Read a file from the user's workspace.",
                input_model=FileReadInput,
            ),
            ToolSpec(
                name="file.write",
                description="Write a file in the user's workspace.",
                input_model=FileWriteInput,
            ),
            ToolSpec(
                name="file.edit",
                description="Replace exact text in a file in the user's workspace.",
                input_model=FileEditInput,
            ),
            ToolSpec(
                name="file.multi_edit",
                description="Apply several exact text replacements to one file in the user's workspace.",
                input_model=FileMultiEditInput,
            ),
            ToolSpec(
                name="file.glob",
                description="Find files in the user's workspace matching a glob pattern.",
                input_model=FileGlobInput,
            ),
            ToolSpec(
                name="file.grep",
                description="Search text in files in the user's workspace.",
                input_model=FileGrepInput,
            ),
            ToolSpec(
                name="file.list",
                description="List files and directories under a workspace path.",
                input_model=FileListInput,
            ),
            ToolSpec(
                name="web.fetch",
                description=(
                    "Fetch an HTTP or HTTPS URL and answer the prompt from the page content."
                ),
                input_model=WebFetchInput,
            ),
            ToolSpec(
                name="task.create",
                description="Create a task in the current conversation checklist.",
                input_model=TaskCreateInput,
            ),
            ToolSpec(
                name="task.get",
                description="Read one task from the current conversation checklist.",
                input_model=TaskGetInput,
            ),
            ToolSpec(
                name="task.list",
                description="List tasks in the current conversation checklist.",
                input_model=TaskListInput,
            ),
            ToolSpec(
                name="task.update",
                description="Update a task title or status in the current conversation checklist.",
                input_model=TaskUpdateInput,
            ),
            ToolSpec(
                name="task.stop",
                description="Mark a task stopped in the current conversation checklist.",
                input_model=TaskStopInput,
            ),
            ToolSpec(
                name="schedule.once",
                description=(
                    "Schedule one future synthetic user message. Use this instead of "
                    "sleeping or waiting inside shell commands."
                ),
                input_model=ScheduleOnceInput,
            ),
            ToolSpec(
                name="schedule.cron",
                description=(
                    "Schedule a recurring synthetic user message using a cron expression "
                    "and timezone."
                ),
                input_model=ScheduleCronInput,
            ),
            ToolSpec(
                name="schedule.list",
                description="List active scheduled synthetic user messages.",
                input_model=ScheduleListInput,
            ),
            ToolSpec(
                name="schedule.cancel",
                description="Cancel a scheduled synthetic user message.",
                input_model=ScheduleCancelInput,
            ),
            ToolSpec(
                name="skill.list",
                description="List enabled markdown skills for the user.",
                input_model=SkillListInput,
            ),
            ToolSpec(
                name="skill.read",
                description="Read the body of an enabled markdown skill.",
                input_model=SkillReadInput,
            ),
            ToolSpec(
                name="memory",
                description=(
                    "Save durable information that should outlive this session "
                    "into the persistent memory store.\n"
                    "Save proactively when the user reveals a preference, "
                    "habit, role, personal detail, or correction; when you "
                    "learn an environment fact, project convention, or tool "
                    "quirk that will matter next time.\n"
                    "Two targets:\n"
                    "- 'user': who the user is (name, role, communication "
                    "style, recurring corrections).\n"
                    "- 'memory': your own notes (environment, conventions, "
                    "tool quirks, lessons learned).\n"
                    "Three actions:\n"
                    "- add: append a new entry.\n"
                    "- replace: change one entry — old_text is a short unique "
                    "substring of the entry to update.\n"
                    "- remove: delete one entry — old_text identifies it the "
                    "same way.\n"
                    "Write declarative facts ('User prefers concise responses'), "
                    "not imperatives ('Always respond concisely'). Imperatives "
                    "re-injected into future sessions act as directives.\n"
                    "Do NOT save task progress, completed-work logs, or "
                    "transient state to memory — use session.search to recall "
                    "those from past transcripts."
                ),
                input_model=MemoryToolInput,
            ),
            ToolSpec(
                name="session.search",
                description=(
                    "Search past conversations and return focused summaries of "
                    "matching sessions. Use when the user references something "
                    "from a past conversation or you suspect relevant "
                    "cross-session context exists. The current conversation "
                    "is excluded from results. Returns one short summary per "
                    "matching session, not raw transcripts."
                ),
                input_model=SessionSearchInput,
            ),
            ToolSpec(
                name="agent.run",
                description=(
                    "Run a sub-agent to completion and return its final assistant "
                    "message. The sub-agent can use workspace file, shell, web, "
                    "task, schedule, skill, and MCP tools but cannot spawn further "
                    "sub-agents."
                ),
                input_model=AgentRunInput,
            ),
            ToolSpec(
                name="agent.spawn",
                description=(
                    "Start a background sub-agent. The sub-agent can use workspace "
                    "file, shell, web, task, schedule, skill, and MCP tools but "
                    "cannot spawn further sub-agents."
                ),
                input_model=AgentSpawnInput,
            ),
            ToolSpec(
                name="agent.result",
                description="Read one sub-agent status and result by id.",
                input_model=AgentResultInput,
            ),
            ToolSpec(
                name="agent.list",
                description="List sub-agents created by the current conversation.",
                input_model=AgentListInput,
            ),
            ToolSpec(
                name="agent.cancel",
                description="Cancel a running background sub-agent.",
                input_model=AgentCancelInput,
            ),
        ]
    ).filter_tools(
        include_memory=include_memory,
        include_session_search=include_session_search,
    )
