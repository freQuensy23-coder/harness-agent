from harness_agent.llm import (
    AssistantToolCallMessage,
    ToolResultMessage,
    UserMessage,
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
