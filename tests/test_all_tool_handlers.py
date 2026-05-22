from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.context import AgentFileSet, ContextBuilder, Skill
from harness_agent.events import (
    AgentTurnRequested,
    ToolCallCompleted,
    ToolCallRequested,
    UserTextReceived,
)
from harness_agent.handlers import AgentTurnHandler, ConversationProjector
from harness_agent.llm import AssistantText, FakeLlmClient, LlmToolCall
from harness_agent.projections import SQLiteConversationProjection
from harness_agent.runtime import FakeUserRuntime, RuntimeToolResult
from harness_agent.scheduler import SQLiteScheduleStore
from harness_agent.store import SQLiteEventStore
from harness_agent.subagents import SubAgentRecord
from harness_agent.image_generate import GeneratedImage, ImageGenerator
from harness_agent.image_jobs import ImageJobService, SQLiteImageJobStore
from harness_agent.tasks import SQLiteTaskStore
from harness_agent.tool_executor import ToolCallExecutor, ToolCallResultWaiter
from harness_agent.tools import (
    AgentCancelInput,
    AgentListInput,
    AgentResultInput,
    AgentRunInput,
    AgentSpawnInput,
    FileEditInput,
    FileEditOperation,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    ImageGenerateInput,
    ImageStatusInput,
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
    ToolRegistry,
    ToolSpec,
    WebFetchInput,
    default_tool_registry,
)


@pytest.mark.asyncio
async def test_every_exposed_tool_completes_through_agent_turn_handler(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteConversationProjection(tmp_path / "messages.sqlite3")
    task_store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    schedule_store = SQLiteScheduleStore(tmp_path / "schedules.sqlite3")
    existing_task = await task_store.create(
        user_id="u:1",
        conversation_id="cli:all-tools",
        title="existing",
        status="pending",
    )
    existing_schedule = await schedule_store.create_once(
        user_id="u:1",
        conversation_id="cli:all-tools",
        message="[scheduled] existing",
        reply_target=None,
        delay_seconds=60,
    )
    runtime = FakeUserRuntime(
        files={
            "/workspace/agent/SOUL.md": "S",
            "/workspace/agent/AGENTS.md": "A",
            "/workspace/agent/USER.md": "U",
            "/workspace/agent/TOOLS.md": "T",
            "/workspace/file.txt": "alpha\nbeta\n",
        },
        agent_files=AgentFileSet(soul="S", agents="A", user="U", tools="T"),
        skills=[Skill(name="demo", description="Demo skill.", body="body")],
        shell_results=[RuntimeToolResult(stdout="shell-ok\n")],
    )
    responses = [
        LlmToolCall(call_id="shell-exec", name="shell.exec", input=ShellExecInput(command="pwd")),
        LlmToolCall(call_id="shell-spawn", name="shell.spawn", input=ShellSpawnInput(command="sleep 10")),
        LlmToolCall(call_id="shell-read", name="shell.read", input=ShellReadInput(process_id="fake-process")),
        LlmToolCall(call_id="shell-kill", name="shell.kill", input=ShellKillInput(process_id="fake-process")),
        LlmToolCall(call_id="file-write", name="file.write", input=FileWriteInput(path="/workspace/file.txt", content="alpha\nbeta\n")),
        LlmToolCall(call_id="file-read", name="file.read", input=FileReadInput(path="/workspace/file.txt")),
        LlmToolCall(call_id="file-edit", name="file.edit", input=FileEditInput(path="/workspace/file.txt", old="beta", new="gamma")),
        LlmToolCall(
            call_id="file-multi-edit",
            name="file.multi_edit",
            input=FileMultiEditInput(
                path="/workspace/file.txt",
                edits=[FileEditOperation(old="alpha", new="one")],
            ),
        ),
        LlmToolCall(call_id="file-glob", name="file.glob", input=FileGlobInput(pattern="*.txt")),
        LlmToolCall(call_id="file-grep", name="file.grep", input=FileGrepInput(pattern="one", path="/workspace")),
        LlmToolCall(call_id="file-list", name="file.list", input=FileListInput(path="/workspace")),
        LlmToolCall(
            call_id="web-fetch",
            name="web.fetch",
            input=WebFetchInput(url="http://local.test", prompt="Extract the useful text."),
        ),
        LlmToolCall(
            call_id="image-generate",
            name="image.generate",
            input=ImageGenerateInput(
                prompt="a cat with a hat",
                output_path="/workspace/content/cat.png",
            ),
        ),
        LlmToolCall(
            call_id="image-status",
            name="image.status",
            input=ImageStatusInput(image_id="placeholder"),
        ),
        LlmToolCall(call_id="task-create", name="task.create", input=TaskCreateInput(title="new")),
        LlmToolCall(call_id="task-get", name="task.get", input=TaskGetInput(task_id=existing_task.id)),
        LlmToolCall(call_id="task-list", name="task.list", input=TaskListInput(include_stopped=True)),
        LlmToolCall(call_id="task-update", name="task.update", input=TaskUpdateInput(task_id=existing_task.id, title="updated")),
        LlmToolCall(call_id="task-stop", name="task.stop", input=TaskStopInput(task_id=existing_task.id)),
        LlmToolCall(call_id="schedule-once", name="schedule.once", input=ScheduleOnceInput(message="[scheduled] once", delay_seconds=60)),
        LlmToolCall(call_id="schedule-cron", name="schedule.cron", input=ScheduleCronInput(message="[cron] every morning", cron="0 9 * * *")),
        LlmToolCall(call_id="schedule-list", name="schedule.list", input=ScheduleListInput(include_stopped=True)),
        LlmToolCall(call_id="schedule-cancel", name="schedule.cancel", input=ScheduleCancelInput(schedule_id=existing_schedule.id)),
        LlmToolCall(call_id="skill-list", name="skill.list", input=SkillListInput()),
        LlmToolCall(call_id="skill-read", name="skill.read", input=SkillReadInput(name="demo")),
        LlmToolCall(call_id="agent-run", name="agent.run", input=AgentRunInput(prompt="write file")),
        LlmToolCall(call_id="agent-spawn", name="agent.spawn", input=AgentSpawnInput(prompt="background write")),
        LlmToolCall(call_id="agent-result", name="agent.result", input=AgentResultInput(agent_id="agent-1")),
        LlmToolCall(call_id="agent-list", name="agent.list", input=AgentListInput(include_completed=True)),
        LlmToolCall(call_id="agent-cancel", name="agent.cancel", input=AgentCancelInput(agent_id="agent-1")),
        LlmToolCall(call_id="mcp-echo", name="mcp.local.echo", input=McpToolInput(arguments={"text": "mcp-ok"})),
        AssistantText(text="done"),
    ]
    llm = FakeLlmClient(responses)
    bus = EventBus(store)
    registry = ToolRegistry(
        tools=[
            *default_tool_registry().tools,
            ToolSpec(
                name="mcp.local.echo",
                description="Echo text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ]
    )
    tool_results = ToolCallResultWaiter()
    mcp_manager = FakeMcpManager()
    handler = AgentTurnHandler(
        bus=bus,
        context_builder=ContextBuilder(runtime=runtime),
        llm=llm,
        tool_registry=registry,
        projection=projection,
        mcp_manager=mcp_manager,
        tool_results=tool_results,
    )
    image_job_store = SQLiteImageJobStore(tmp_path / "image_jobs.sqlite3")
    image_jobs = ImageJobService(
        store=image_job_store,
        generator=FakeImageGenerator(),
        runtime=runtime,
    )
    tool_executor = ToolCallExecutor(
        runtime=runtime,
        task_store=task_store,
        schedule_store=schedule_store,
        web_fetcher=FakeWebFetcher(),
        image_jobs=image_jobs,
        mcp_manager=mcp_manager,
        sub_agents=FakeSubAgentService(),
    )
    projector = ConversationProjector(projection)
    bus.subscribe(UserTextReceived, projector.handle_user_text)
    bus.subscribe(ToolCallRequested, tool_executor.handle_tool_call_requested)
    bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, ConversationProjector(projection).handle_tool_call_completed)
    bus.subscribe(ToolCallCompleted, handler.handle_tool_call_completed)
    bus.subscribe(UserTextReceived, handler.handle_user_text)
    bus.subscribe(AgentTurnRequested, handler.handle_agent_turn)

    await bus.publish(
        UserTextReceived(
            user_id="u:1",
            conversation_id="cli:all-tools",
            source="cli",
            text="exercise tools",
        )
    )

    completed = [
        event.tool_name
        for event in await store.list_events()
        if event.type == "tool.call.completed"
    ]
    assert completed == [
        "shell.exec",
        "shell.spawn",
        "shell.read",
        "shell.kill",
        "file.write",
        "file.read",
        "file.edit",
        "file.multi_edit",
        "file.glob",
        "file.grep",
        "file.list",
        "web.fetch",
        "image.generate",
        "image.status",
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
        "agent.run",
        "agent.spawn",
        "agent.result",
        "agent.list",
        "agent.cancel",
        "mcp.local.echo",
    ]


class FakeWebFetcher:
    async def fetch(self, input: WebFetchInput) -> RuntimeToolResult:
        return RuntimeToolResult(stdout=f"fetched:{input.url}")


class FakeImageGenerator(ImageGenerator):
    async def generate(self, input: ImageGenerateInput) -> GeneratedImage:
        return GeneratedImage(mime_type="image/png", data=b"\x89PNG\r\n\x1a\nfake")


class FakeMcpManager:
    async def list_tool_specs(self, user_id: str) -> list[ToolSpec]:
        return []

    async def call_tool(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: dict,
    ) -> RuntimeToolResult:
        return RuntimeToolResult(stdout=arguments["text"])


class FakeSubAgentService:
    async def run(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentRunInput,
    ) -> SubAgentRecord:
        return fake_sub_agent_record(
            parent_conversation_id=parent_conversation_id,
            parent_call_id=parent_call_id,
            prompt=input.prompt,
            status="completed",
            result="done",
        )

    async def spawn(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        parent_call_id: str,
        input: AgentSpawnInput,
    ) -> SubAgentRecord:
        return fake_sub_agent_record(
            parent_conversation_id=parent_conversation_id,
            parent_call_id=parent_call_id,
            prompt=input.prompt,
            status="running",
            result=None,
        )

    async def result(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord:
        return fake_sub_agent_record(
            parent_conversation_id=parent_conversation_id,
            parent_call_id="agent-run",
            prompt="write file",
            status="completed",
            result="done",
        )

    async def list_for_parent(
        self,
        *,
        user_id: str,
        parent_conversation_id: str,
        include_completed: bool,
    ) -> list[SubAgentRecord]:
        return [
            fake_sub_agent_record(
                parent_conversation_id=parent_conversation_id,
                parent_call_id="agent-run",
                prompt="write file",
                status="completed",
                result="done",
            )
        ]

    async def cancel(
        self,
        *,
        agent_id: str,
        user_id: str,
        parent_conversation_id: str,
    ) -> SubAgentRecord:
        return fake_sub_agent_record(
            parent_conversation_id=parent_conversation_id,
            parent_call_id="agent-run",
            prompt="write file",
            status="cancelled",
            result=None,
        )


def fake_sub_agent_record(
    *,
    parent_conversation_id: str,
    parent_call_id: str,
    prompt: str,
    status: str,
    result: str | None,
) -> SubAgentRecord:
    now = datetime.now(UTC)
    return SubAgentRecord(
        id="agent-1",
        user_id="u:1",
        parent_conversation_id=parent_conversation_id,
        child_conversation_id=f"{parent_conversation_id}:subagent:agent-1",
        parent_call_id=parent_call_id,
        name="subagent",
        prompt=prompt,
        status=status,
        result=result,
        created_at=now,
        updated_at=now,
    )
