from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import event_from_aiogram_message
from harness_agent.config import load_config
from harness_agent.context import Skill
from harness_agent.runtime import DockerProcessResult, DockerUserRuntime
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
