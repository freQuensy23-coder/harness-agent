from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram.enums import ChatAction, ParseMode
from aiogram.exceptions import TelegramBadRequest

from harness_agent import events as event_models
from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import AiogramTelegramAdapter, event_from_aiogram_message
from harness_agent.bus import EventBus
from harness_agent.config import load_config
from harness_agent.context import Skill
from harness_agent.events import AssistantTextProduced, TelegramReplyTarget
from harness_agent.runtime import DockerProcessResult, DockerUserRuntime
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import FileWriteInput, ShellExecInput


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
async def test_telegram_adapter_sends_assistant_text_as_markdown() -> None:
    coordinator = ConversationTurnCoordinator()
    generation = await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:ABC",
        bus=EventBus(RecordingEventStore()),
        turn_coordinator=coordinator,
    )
    bot = RecordingTelegramBot()
    adapter._bot = bot

    await adapter.handle_assistant_text(
        AssistantTextProduced(
            user_id="user:123",
            conversation_id="tg:456",
            generation=generation,
            text="*bold*",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.sent_messages == [
        {"chat_id": 456, "text": "*bold*", "parse_mode": ParseMode.MARKDOWN}
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_falls_back_to_plain_text_for_invalid_markdown() -> None:
    coordinator = ConversationTurnCoordinator()
    generation = await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:ABC",
        bus=EventBus(RecordingEventStore()),
        turn_coordinator=coordinator,
    )
    bot = RecordingTelegramBot(
        message_errors=[
            TelegramBadRequest(
                method=None,
                message="Bad Request: can't parse entities: Can't find end of entity",
            )
        ]
    )
    adapter._bot = bot

    await adapter.handle_assistant_text(
        AssistantTextProduced(
            user_id="user:123",
            conversation_id="tg:456",
            generation=generation,
            text="**broken",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.sent_messages == [
        {"chat_id": 456, "text": "**broken", "parse_mode": ParseMode.MARKDOWN},
        {"chat_id": 456, "text": "**broken"},
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_sends_typing_when_generation_starts() -> None:
    coordinator = ConversationTurnCoordinator()
    generation = await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:ABC",
        bus=EventBus(RecordingEventStore()),
        turn_coordinator=coordinator,
    )
    bot = RecordingTelegramBot()
    adapter._bot = bot
    generation_started = event_models.AgentGenerationStarted(
        user_id="user:123",
        conversation_id="tg:456",
        generation=generation,
        reply_target=TelegramReplyTarget(chat_id=456),
    )

    await adapter.handle_agent_generation_started(generation_started)

    assert bot.chat_actions == [{"chat_id": 456, "action": ChatAction.TYPING}]


def test_harness_app_does_not_own_telegram_outbound_ux() -> None:
    source = Path("src/harness_agent/app.py").read_text(encoding="utf-8")

    assert "_send_telegram_reply" not in source
    assert "send_assistant_text" not in source
    assert "send_chat_action" not in source


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


class RecordingEventStore:
    async def append(self, event) -> None:
        return None


class RecordingTelegramBot:
    def __init__(self, *, message_errors: list[Exception] | None = None) -> None:
        self.sent_messages: list[dict] = []
        self.chat_actions: list[dict] = []
        self._message_errors = [] if message_errors is None else message_errors

    async def send_message(self, **kwargs) -> None:
        self.sent_messages.append(kwargs)
        if self._message_errors:
            raise self._message_errors.pop(0)

    async def send_chat_action(self, **kwargs) -> None:
        self.chat_actions.append(kwargs)
