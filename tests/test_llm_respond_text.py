"""Unit tests for LlmClient.respond_text -- the typed-text variant that
keeps the isinstance guard at the LLM boundary instead of leaking it
into business code (e.g. web_fetch extraction)."""

import pytest

from harness_agent.llm import (
    AssistantText,
    FakeLlmClient,
    LlmRequest,
    LlmToolCall,
    UserMessage,
)
from harness_agent.tools import FileWriteInput, ToolSpec


def _request_with_tools(tools: list[ToolSpec]) -> LlmRequest:
    return LlmRequest(
        user_id="u:1",
        conversation_id="cli:1",
        generation=1,
        system="sys",
        messages=[UserMessage(text="hello")],
        tools=tools,
    )


@pytest.mark.asyncio
async def test_respond_text_returns_assistant_text_for_text_response() -> None:
    client = FakeLlmClient([AssistantText(text="ok")])
    result = await client.respond_text(_request_with_tools(tools=[]))
    assert result == AssistantText(text="ok")


@pytest.mark.asyncio
async def test_respond_text_rejects_non_empty_tools() -> None:
    """respond_text is the typed-text contract; allowing tools would let
    the LLM legitimately answer with a tool call, defeating the
    isinstance guard the method is supposed to enforce."""
    client = FakeLlmClient([AssistantText(text="ok")])
    forbidden = _request_with_tools(
        tools=[
            ToolSpec(name="file.write", description="dummy", input_model=FileWriteInput)
        ]
    )
    with pytest.raises(ValueError, match="respond_text only accepts requests with tools=\\[\\]"):
        await client.respond_text(forbidden)


@pytest.mark.asyncio
async def test_respond_text_raises_when_model_returns_a_tool_call() -> None:
    """Even with tools=[], a misbehaving model could still produce a
    tool call; respond_text must raise rather than return an unexpected
    union variant up the stack."""
    client = FakeLlmClient(
        [
            LlmToolCall(
                call_id="bad",
                name="file.write",
                input=FileWriteInput(path="/workspace/x.txt", content="x"),
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="tool call for a respond_text request"):
        await client.respond_text(_request_with_tools(tools=[]))
