from types import SimpleNamespace

from harness_agent.llm import (
    AssistantMessage,
    AssistantToolCallMessage,
    LlmRequest,
    ToolResultMessage,
    UserMessage,
    estimate_request_tokens,
    message_to_openai,
    parse_tool_input,
    tool_to_openai,
)
from harness_agent.tools import ShellExecInput, default_tool_registry


def test_tool_to_openai_uses_chat_completions_tool_shape() -> None:
    tool = default_tool_registry().by_name("shell.exec")

    assert tool_to_openai(tool)["type"] == "function"
    assert tool_to_openai(tool)["function"]["name"] == "shell__exec"
    assert "Canonical tool: shell.exec." in tool_to_openai(tool)["function"]["description"]
    assert tool_to_openai(tool)["function"]["parameters"]["properties"]["command"]["type"] == "string"


def test_parse_tool_input_validates_named_tool_arguments() -> None:
    input = parse_tool_input(
        "shell__exec",
        {"command": "pwd", "cwd": "/workspace", "timeout_seconds": 5},
    )

    assert input == ShellExecInput(command="pwd", cwd="/workspace", timeout_seconds=5)


def test_message_to_openai_preserves_tool_call_history_shape() -> None:
    assert message_to_openai(UserMessage(text="hi")) == {
        "role": "user",
        "content": "hi",
    }
    assert message_to_openai(
        AssistantToolCallMessage(
            call_id="call_1",
            name="shell.exec",
            arguments={"command": "pwd"},
        )
    ) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "shell__exec",
                    "arguments": '{"command": "pwd"}',
                },
            }
        ],
    }
    assert message_to_openai(
        ToolResultMessage(
            call_id="call_1",
            name="shell.exec",
            content="ok",
        )
    ) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ok",
    }


def test_estimate_request_tokens_uses_gpt5_tiktoken(monkeypatch) -> None:
    import harness_agent.llm as llm_module

    calls: list[str] = []

    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    def encoding_for_model(model: str) -> FakeEncoding:
        calls.append(model)
        return FakeEncoding()

    monkeypatch.setattr(
        llm_module,
        "tiktoken",
        SimpleNamespace(encoding_for_model=encoding_for_model),
    )

    tokens = estimate_request_tokens(
        LlmRequest(
            user_id="u:1",
            conversation_id="cli:tokens",
            generation=1,
            system="system prompt",
            messages=[UserMessage(text="hello"), AssistantMessage(text="hi")],
            tools=[default_tool_registry().by_name("shell.exec")],
        )
    )

    assert tokens > 0
    assert calls == ["gpt-5"]
