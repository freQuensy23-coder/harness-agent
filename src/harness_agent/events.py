from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from harness_agent.content import ContentRef
from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import ToolInput, parse_stored_tool_input


class EventBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TelegramReplyTarget(BaseModel):
    kind: Literal["telegram"] = "telegram"
    chat_id: int


class CliReplyTarget(BaseModel):
    kind: Literal["cli"] = "cli"
    request_id: str


ReplyTarget = Annotated[
    TelegramReplyTarget | CliReplyTarget,
    Field(discriminator="kind"),
]


class InboundAttachment(BaseModel):
    kind: Literal["image", "file"]
    file_name: str
    mime_type: str | None = None
    size_bytes: int
    workspace_path: str
    content_base64: str
    source_id: str | None = None


class TelegramTextReceived(EventBase):
    type: Literal["telegram.text.received"] = "telegram.text.received"
    telegram_user_id: int
    telegram_chat_id: int
    telegram_message_id: int
    text: str
    attachments: list[InboundAttachment] = Field(default_factory=list[InboundAttachment])


class CliTextReceived(EventBase):
    type: Literal["cli.text.received"] = "cli.text.received"
    cli_user_id: str
    conversation_id: str
    request_id: str
    text: str


class UserTextReceived(EventBase):
    type: Literal["user.text.received"] = "user.text.received"
    user_id: str
    conversation_id: str
    source: Literal["telegram", "cli", "api", "scheduler", "subagent"]
    text: str
    attachments: list[InboundAttachment] = Field(default_factory=list[InboundAttachment])
    reply_target: ReplyTarget | None = None


class AgentTurnRequested(EventBase):
    type: Literal["agent.turn.requested"] = "agent.turn.requested"
    user_id: str
    conversation_id: str
    generation: int
    input_event_id: str
    reply_target: ReplyTarget | None = None


class AgentTurnSuperseded(EventBase):
    type: Literal["agent.turn.superseded"] = "agent.turn.superseded"
    user_id: str
    conversation_id: str
    generation: int
    superseded_by: int
    reason: Literal["newer_user_message"] = "newer_user_message"


class AgentGenerationStarted(EventBase):
    type: Literal["agent.generation.started"] = "agent.generation.started"
    user_id: str
    conversation_id: str
    generation: int
    reply_target: ReplyTarget | None = None


class ToolCallRequested(EventBase):
    type: Literal["tool.call.requested"] = "tool.call.requested"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    reply_target: ReplyTarget | None = None

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class ToolCallError(EventBase):
    type: Literal["tool.call.error"] = "tool.call.error"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    error: str

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class ToolCallCompleted(EventBase):
    type: Literal["tool.call.completed"] = "tool.call.completed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    result: RuntimeToolResult
    attachments: list[ContentRef] = Field(default_factory=list[ContentRef])

    @field_validator("input", mode="before")
    @classmethod
    def parse_input_for_tool(cls, value: Any, info: ValidationInfo) -> Any:
        tool_name = info.data.get("tool_name")
        if tool_name is None:
            return value
        return parse_stored_tool_input(tool_name, value)


class ShellProcessSpawned(EventBase):
    type: Literal["shell.process.spawned"] = "shell.process.spawned"
    user_id: str
    process_id: str
    container_name: str
    command: str
    cwd: str
    base_path: str


class ShellProcessOutputAdvanced(EventBase):
    type: Literal["shell.process.output_advanced"] = "shell.process.output_advanced"
    user_id: str
    process_id: str
    stdout_offset: int
    stderr_offset: int


class ShellProcessTerminated(EventBase):
    type: Literal["shell.process.terminated"] = "shell.process.terminated"
    user_id: str
    process_id: str
    reason: Literal["killed", "exited", "spawn_failed"]


class WebFetchExtractionRequested(EventBase):
    type: Literal["web_fetch.extraction.requested"] = "web_fetch.extraction.requested"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    url: str
    prompt: str
    max_bytes: int


class WebFetchExtractionCompleted(EventBase):
    type: Literal["web_fetch.extraction.completed"] = "web_fetch.extraction.completed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    answer: str


class WebFetchExtractionFailed(EventBase):
    type: Literal["web_fetch.extraction.failed"] = "web_fetch.extraction.failed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    error: str


class ToolResultSpilled(EventBase):
    type: Literal["tool.result.spilled"] = "tool.result.spilled"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    workspace_path: str
    rendered_size_bytes: int


class ToolResultSpillFailed(EventBase):
    type: Literal["tool.result.spill_failed"] = "tool.result.spill_failed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    workspace_path: str
    rendered_size_bytes: int
    error: str


class AssistantTextProduced(EventBase):
    type: Literal["assistant.text.produced"] = "assistant.text.produced"
    user_id: str
    conversation_id: str
    generation: int
    text: str
    reply_target: ReplyTarget | None = None


class ScheduledMessageDue(EventBase):
    type: Literal["scheduled.message.due"] = "scheduled.message.due"
    schedule_id: str
    user_id: str
    conversation_id: str
    text: str
    reply_target: ReplyTarget | None = None


class SubAgentRequested(EventBase):
    type: Literal["subagent.requested"] = "subagent.requested"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str
    timeout_seconds: float


class SubAgentStarted(EventBase):
    type: Literal["subagent.started"] = "subagent.started"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    parent_call_id: str
    name: str
    prompt: str = ""
    timeout_seconds: float = 0.0


class SubAgentTimedOut(EventBase):
    type: Literal["subagent.timed_out"] = "subagent.timed_out"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str


class SubAgentCompleted(EventBase):
    type: Literal["subagent.completed"] = "subagent.completed"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    result: str


class SubAgentFailed(EventBase):
    type: Literal["subagent.failed"] = "subagent.failed"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str
    error: str


class SubAgentCancelled(EventBase):
    type: Literal["subagent.cancelled"] = "subagent.cancelled"
    agent_id: str
    user_id: str
    parent_conversation_id: str
    child_conversation_id: str


class BrowserProfileCreated(EventBase):
    type: Literal["browser.profile.created"] = "browser.profile.created"
    user_id: str
    cloud_profile_id: str


class BrowserProfileEvicted(EventBase):
    type: Literal["browser.profile.evicted"] = "browser.profile.evicted"
    evicted_user_id: str
    cloud_profile_id: str
    requested_by_user_id: str


class BrowserSessionStarted(EventBase):
    type: Literal["browser.session.started"] = "browser.session.started"
    session_id: str
    user_id: str
    conversation_id: str
    generation: int
    parent_call_id: str
    cloud_session_id: str
    cloud_profile_id: str
    live_url: str | None = None


class BrowserSessionPollDue(EventBase):
    type: Literal["browser.session.poll_due"] = "browser.session.poll_due"
    session_id: str
    user_id: str


class BrowserSessionMessageReceived(EventBase):
    type: Literal["browser.session.message_received"] = "browser.session.message_received"
    session_id: str
    user_id: str
    conversation_id: str
    cloud_session_id: str
    cloud_message_id: str
    role: str
    summary: str | None = None
    data: str


class BrowserSessionCompleted(EventBase):
    type: Literal["browser.session.completed"] = "browser.session.completed"
    session_id: str
    user_id: str
    conversation_id: str
    cloud_session_id: str
    output: str | None = None
    step_count: int = 0
    # True when the cloud session went idle but the local record was
    # spawned with keep_alive=True, so the projection must stay 'idle'
    # (still reusable via browser.send) instead of moving to 'completed'.
    keep_alive_idle: bool = False


class BrowserSessionFailed(EventBase):
    type: Literal["browser.session.failed"] = "browser.session.failed"
    session_id: str
    user_id: str
    conversation_id: str
    cloud_session_id: str
    status: str
    error: str | None = None


class BrowserSessionStopped(EventBase):
    type: Literal["browser.session.stopped"] = "browser.session.stopped"
    session_id: str
    user_id: str
    conversation_id: str
    cloud_session_id: str
    requested_by_user_id: str


class BrowserSessionStatusChanged(EventBase):
    """Non-terminal status / live_url change for a browser session.
    Emitted by the poll handler (mid-flight cloud transitions like
    created->running) and by browser.send (status returned from the
    cloud). The projection is updated by handle_session_status_changed,
    so non-terminal state changes are also derived from the event log."""

    type: Literal["browser.session.status_changed"] = "browser.session.status_changed"
    session_id: str
    user_id: str
    conversation_id: str
    cloud_session_id: str
    status: str
    live_url: str | None = None


class BrowserProfileTouched(EventBase):
    """Recency signal for the LRU eviction policy. Published whenever a
    user's profile is observably in use (ensure_profile cache hit,
    browser.send, etc.); the projection's last_used_at column is an
    event-derived field."""

    type: Literal["browser.profile.touched"] = "browser.profile.touched"
    user_id: str
    cloud_profile_id: str


class BrowserSessionTaskSent(EventBase):
    type: Literal["browser.session.task_sent"] = "browser.session.task_sent"
    session_id: str
    user_id: str
    cloud_session_id: str
    task: str
    status: str


AgentEvent = Annotated[
    TelegramTextReceived
    | CliTextReceived
    | UserTextReceived
    | AgentTurnRequested
    | AgentTurnSuperseded
    | AgentGenerationStarted
    | ToolCallRequested
    | ToolCallError
    | ToolCallCompleted
    | ShellProcessSpawned
    | ShellProcessOutputAdvanced
    | ShellProcessTerminated
    | WebFetchExtractionRequested
    | WebFetchExtractionCompleted
    | WebFetchExtractionFailed
    | ToolResultSpilled
    | ToolResultSpillFailed
    | AssistantTextProduced
    | ScheduledMessageDue
    | SubAgentRequested
    | SubAgentStarted
    | SubAgentTimedOut
    | SubAgentCompleted
    | SubAgentFailed
    | SubAgentCancelled
    | BrowserProfileCreated
    | BrowserProfileEvicted
    | BrowserProfileTouched
    | BrowserSessionStarted
    | BrowserSessionPollDue
    | BrowserSessionMessageReceived
    | BrowserSessionCompleted
    | BrowserSessionFailed
    | BrowserSessionStatusChanged
    | BrowserSessionStopped
    | BrowserSessionTaskSent,
    Field(discriminator="type"),
]
