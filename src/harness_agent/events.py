from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from harness_agent.content import ContentRef
from harness_agent.runtime import RuntimeToolResult
from harness_agent.tools import ToolInput


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
    attachments: list[InboundAttachment] = Field(default_factory=list)


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
    source: Literal["telegram", "cli", "api", "scheduler"]
    text: str
    attachments: list[InboundAttachment] = Field(default_factory=list)
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


class ToolCallRequested(EventBase):
    type: Literal["tool.call.requested"] = "tool.call.requested"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    reply_target: ReplyTarget | None = None


class ToolCallCompleted(EventBase):
    type: Literal["tool.call.completed"] = "tool.call.completed"
    user_id: str
    conversation_id: str
    generation: int
    call_id: str
    tool_name: str
    input: ToolInput
    result: RuntimeToolResult
    attachments: list[ContentRef] = Field(default_factory=list)


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


AgentEvent = Annotated[
    TelegramTextReceived
    | CliTextReceived
    | UserTextReceived
    | AgentTurnRequested
    | AgentTurnSuperseded
    | ToolCallRequested
    | ToolCallCompleted
    | AssistantTextProduced
    | ScheduledMessageDue,
    Field(discriminator="type"),
]
