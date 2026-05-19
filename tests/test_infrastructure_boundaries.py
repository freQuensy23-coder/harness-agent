from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest

from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import AiogramTelegramAdapter, event_from_aiogram_message
from harness_agent.app import HarnessApp
from harness_agent.config import load_config
from harness_agent.context import Skill
from harness_agent.events import (
    AgentTurnRequested,
    AssistantTextProduced,
    TelegramReplyTarget,
)
from harness_agent.runtime import DockerProcessResult, DockerUserRuntime
from harness_agent.tools import FileWriteInput, ShellExecInput
from harness_agent.turns import ConversationTurnCoordinator


def test_yaml_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        """
server:
  host: 127.0.0.1
  port: 8080
llm:
  api_key: test-key
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_config(config_path)


def test_telegram_adapter_only_creates_event() -> None:
    message = SimpleNamespace(
        text="Say hi",
        message_id=999,
        from_user=SimpleNamespace(id=123),
        chat=SimpleNamespace(id=456),
    )

    event = event_from_aiogram_message(message)

    assert event.telegram_user_id == 123
    assert event.telegram_chat_id == 456
    assert event.telegram_message_id == 999
    assert event.text == "Say hi"


@pytest.mark.asyncio
async def test_telegram_adapter_sends_assistant_markdown() -> None:
    bot = RecordingTelegramBot()
    adapter = AiogramTelegramAdapter(token="123456:ABCDEF", bus=None)  # type: ignore[arg-type]
    adapter._bot = bot  # type: ignore[method-assign]

    await adapter.send_assistant_text(
        AssistantTextProduced(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            text="**bold**",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.calls == [
        (
            "send_message",
            {"chat_id": 456, "text": "**bold**", "parse_mode": "Markdown"},
        )
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_falls_back_to_plain_text_for_invalid_markdown() -> None:
    bot = RecordingTelegramBot(fail_first_message=True)
    adapter = AiogramTelegramAdapter(token="123456:ABCDEF", bus=None)  # type: ignore[arg-type]
    adapter._bot = bot  # type: ignore[method-assign]

    await adapter.send_assistant_text(
        AssistantTextProduced(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            text="**broken",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.calls == [
        (
            "send_message",
            {"chat_id": 456, "text": "**broken", "parse_mode": "Markdown"},
        ),
        ("send_message", {"chat_id": 456, "text": "**broken"}),
    ]


@pytest.mark.asyncio
async def test_telegram_writing_action_is_sent_for_current_turn() -> None:
    telegram = RecordingTelegramAdapter()
    app = HarnessApp.__new__(HarnessApp)
    app.telegram = telegram
    app.turn_coordinator = ConversationTurnCoordinator()
    generation = await app.turn_coordinator.request_generation("tg:456")
    event = AgentTurnRequested(
        user_id="u:123",
        conversation_id="tg:456",
        generation=generation,
        input_event_id="event-1",
        reply_target=TelegramReplyTarget(chat_id=456),
    )

    result = await app._send_telegram_writing(event)

    assert result == ()
    assert telegram.events == [event]


@pytest.mark.asyncio
async def test_telegram_adapter_sends_typing_chat_action() -> None:
    bot = RecordingTelegramBot()
    adapter = AiogramTelegramAdapter(token="123456:ABCDEF", bus=None)  # type: ignore[arg-type]
    adapter._bot = bot  # type: ignore[method-assign]

    await adapter.send_writing_action(
        AgentTurnRequested(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            input_event_id="event-1",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.calls == [
        ("send_chat_action", {"chat_id": 456, "action": "typing"}),
    ]


def test_cli_adapter_only_creates_event() -> None:
    event = event_from_cli_send(
        text="Say hi",
        user_id="123",
        conversation_id=None,
    )

    assert event.cli_user_id == "123"
    assert event.conversation_id == "cli:123"
    assert event.text == "Say hi"


@pytest.mark.asyncio
async def test_docker_runtime_routes_shell_without_changing_tool_name() -> None:
    runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="ok\n", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=runner)

    result = await runtime.shell_exec(
        "u:123",
        ShellExecInput(command="pwd", cwd="/workspace"),
    )

    assert result.stdout == "ok\n"
    assert runner.calls == [
        (
            ["docker", "exec", "-w", "/workspace", "harness-u-123", "sh", "-lc", "pwd"],
            None,
        )
    ]


@pytest.mark.asyncio
async def test_docker_runtime_writes_file_with_stdin() -> None:
    runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=runner)

    await runtime.file_write(
        "u:123",
        FileWriteInput(path="/workspace/hello.py", content="print('hi')\n"),
    )

    assert runner.calls == [
        (
            [
                "docker",
                "exec",
                "-i",
                "harness-u-123",
                "sh",
                "-lc",
                "mkdir -p -- /workspace && cat > /workspace/hello.py",
            ],
            b"print('hi')\n",
        )
    ]


class FakeTelegramBadRequest(TelegramBadRequest):
    def __init__(self) -> None:
        Exception.__init__(self, "can't parse entities")


class RecordingTelegramBot:
    def __init__(self, *, fail_first_message: bool = False) -> None:
        self._fail_first_message = fail_first_message
        self.calls: list[tuple[str, dict]] = []

    async def send_message(self, **kwargs) -> None:
        self.calls.append(("send_message", kwargs))
        if self._fail_first_message and len(self.calls) == 1:
            raise FakeTelegramBadRequest()

    async def send_chat_action(self, **kwargs) -> None:
        self.calls.append(("send_chat_action", kwargs))


class RecordingTelegramAdapter:
    def __init__(self) -> None:
        self.events: list[AgentTurnRequested] = []

    async def send_writing_action(self, event: AgentTurnRequested) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_docker_runtime_loads_skills_from_markdown_frontmatter() -> None:
    runner = RecordingDockerRunner(
        [
            DockerProcessResult(
                stdout="/workspace/skills/shell/SKILL.md\n",
                stderr="",
                exit_code=0,
            ),
            DockerProcessResult(
                stdout=(
                    "---\n"
                    "name: shell-work\n"
                    "description: Shell discipline.\n"
                    "---\n"
                    "Stay in /workspace.\n"
                ),
                stderr="",
                exit_code=0,
            ),
        ]
    )
    runtime = DockerUserRuntime(runner=runner)

    assert await runtime.list_skills("u:123") == [
        Skill(
            name="shell-work",
            description="Shell discipline.",
            body="Stay in /workspace.\n",
        )
    ]


class RecordingDockerRunner:
    def __init__(self, results: list[DockerProcessResult]) -> None:
        self._results = results
        self.calls: list[tuple[list[str], bytes | None]] = []

    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        self.calls.append((argv, stdin))
        return self._results.pop(0)
