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


def test_estimate_request_tokens_uses_gpt5_tokenizer_for_every_request(monkeypatch) -> None:
    requested_models: list[str] = []

    class RecordingTokenizer:
        def encode(self, value: str) -> list[int]:
            return list(range(len(value.split())))

    def encoding_for_model(model: str) -> RecordingTokenizer:
        requested_models.append(model)
        return RecordingTokenizer()

    monkeypatch.setattr(
        "harness_agent.llm.tiktoken.encoding_for_model",
        encoding_for_model,
    )

    tokens = estimate_request_tokens(
        LlmRequest(
            user_id="u:1",
            conversation_id="cli:1",
            generation=1,
            system="system prompt",
            messages=[UserMessage(text="hello user"), AssistantMessage(text="hi")],
            tools=[default_tool_registry().by_name("shell.exec")],
        )
    )

    assert requested_models == ["gpt-5"]
    assert tokens > 0
