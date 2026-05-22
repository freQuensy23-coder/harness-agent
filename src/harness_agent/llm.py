import json
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, Field

from harness_agent.content import ContentRef
from harness_agent.tools import (
    ToolInput,
    ToolSpec,
    parse_llm_tool_input,
)


class UserMessage(BaseModel):
    kind: Literal["user"] = "user"
    text: str
    attachments: list[ContentRef] = Field(default_factory=list[ContentRef])


class AssistantMessage(BaseModel):
    kind: Literal["assistant"] = "assistant"
    text: str


class AssistantToolCallMessage(BaseModel):
    kind: Literal["assistant_tool_call"] = "assistant_tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolResultMessage(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    content: str


LlmMessage = Annotated[
    UserMessage | AssistantMessage | AssistantToolCallMessage | ToolResultMessage,
    Field(discriminator="kind"),
]


class LlmRequest(BaseModel):
    user_id: str
    conversation_id: str
    generation: int
    system: str
    messages: list[LlmMessage]
    tools: list[ToolSpec]

    model_config = {"arbitrary_types_allowed": True}


class AssistantText(BaseModel):
    kind: Literal["assistant_text"] = "assistant_text"
    text: str


class LlmToolCall(BaseModel):
    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    input: ToolInput


LlmResponse = AssistantText | LlmToolCall


class LlmClient:
    async def respond(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError

    async def respond_text(self, request: LlmRequest) -> AssistantText:
        """Variant of respond() for callers that pass tools=[] and need a
        typed text answer. The default implementation enforces the
        contract at the LLM boundary so business code (e.g. web.fetch
        extraction) never has to type-check the union itself."""
        if request.tools:
            raise ValueError(
                "respond_text only accepts requests with tools=[]; got "
                f"{[tool.name for tool in request.tools]}"
            )
        response = await self.respond(request)
        if not isinstance(response, AssistantText):
            raise RuntimeError(
                "LLM returned a tool call for a respond_text request"
            )
        return response


class OpenAICompatibleChatClient(LlmClient):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def respond(self, request: LlmRequest) -> LlmResponse:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=request.system),
            *(message_to_openai(message) for message in request.messages),
        ]
        tools: list[ChatCompletionToolUnionParam] = [
            tool_to_openai(tool) for tool in request.tools
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                raise RuntimeError("OpenAI-compatible response returned non-function tool call")
            name = canonical_tool_name(tool_call.function.name)
            arguments = cast(dict[str, Any], json.loads(tool_call.function.arguments))
            return LlmToolCall(
                call_id=tool_call.id,
                name=name,
                input=parse_tool_input(name, arguments),
            )
        if message.content is None:
            raise RuntimeError("OpenAI-compatible response contained no content or tool call")
        return AssistantText(text=message.content)


OpenAIResponsesClient = OpenAICompatibleChatClient


class FakeLlmClient(LlmClient):
    def __init__(self, responses: list[LlmResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[LlmRequest] = []

    async def respond(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeLlmClient has no queued response")
        return self._responses.pop(0)


def tool_to_openai(tool: ToolSpec) -> ChatCompletionFunctionToolParam:
    return ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name=api_tool_name(tool.name),
            description=f"Canonical tool: {tool.name}. {tool.description}",
            parameters=tool.parameters_schema(),
        ),
    )


def message_to_openai(message: LlmMessage) -> ChatCompletionMessageParam:
    if message.kind == "user":
        return _user_message_to_openai(message)
    if message.kind == "assistant":
        return ChatCompletionAssistantMessageParam(role="assistant", content=message.text)
    if message.kind == "assistant_tool_call":
        tool_call = ChatCompletionMessageFunctionToolCallParam(
            id=message.call_id,
            type="function",
            function=Function(
                name=api_tool_name(message.name),
                arguments=json.dumps(message.arguments),
            ),
        )
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )
    if message.kind == "tool_result":
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=message.call_id,
            content=message.content,
        )
    raise ValueError(f"unsupported message kind: {message}")


def _user_message_to_openai(message: UserMessage) -> ChatCompletionUserMessageParam:
    text = render_user_text(message)
    image_parts: list[ChatCompletionContentPartParam] = [
        ChatCompletionContentPartImageParam(
            type="image_url",
            image_url=ImageURL(
                url=(
                    f"data:{attachment.mime_type or 'application/octet-stream'};"
                    f"base64,{attachment.content_base64}"
                )
            ),
        )
        for attachment in message.attachments
        if attachment.kind == "image" and attachment.content_base64 is not None
    ]
    if not image_parts:
        return ChatCompletionUserMessageParam(role="user", content=text)
    return ChatCompletionUserMessageParam(
        role="user",
        content=[
            ChatCompletionContentPartTextParam(type="text", text=text),
            *image_parts,
        ],
    )


def render_user_text(message: UserMessage) -> str:
    if not message.attachments:
        return message.text
    lines = [message.text, "", "Attached files saved in workspace:"]
    for attachment in message.attachments:
        lines.append(
            "- "
            f"{attachment.workspace_path} "
            f"({attachment.kind}, {attachment.file_name}, "
            f"{attachment.mime_type or 'unknown'}, {attachment.size_bytes} bytes)"
        )
    return "\n".join(lines)


def parse_tool_input(name: str, arguments: dict[str, Any]) -> ToolInput:
    name = canonical_tool_name(name)
    return parse_llm_tool_input(name, arguments)


def api_tool_name(name: str) -> str:
    return name.replace(".", "__")


def canonical_tool_name(name: str) -> str:
    return name.replace("__", ".")
