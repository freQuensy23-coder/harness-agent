import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from harness_agent import events
from harness_agent.adapters.cli import event_from_cli_send
from harness_agent.adapters.telegram import AiogramTelegramAdapter, event_from_aiogram_message
from harness_agent.config import load_config
from harness_agent.context import Skill
from harness_agent.events import AssistantTextProduced, TelegramReplyTarget
from harness_agent.runtime import DockerProcessResult, DockerUserRuntime, SQLiteSpawnedProcessStore, InMemorySpawnedProcessStore
from harness_agent.turns import ConversationTurnCoordinator
from harness_agent.tools import (
    FileWriteInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
)


def test_yaml_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        """
server:
  host: 127.0.0.1
  port: 8080
llm:
  api_key: test-key
browser_use:
  api_key: test-bu-key
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_config(config_path)


@pytest.mark.asyncio
async def test_download_attachment_raises_when_telegram_file_has_no_path() -> None:
    """get_file() can succeed with file_path=None for files Telegram
    refuses to serve (too large, deleted, etc). The adapter must raise
    a clear RuntimeError rather than silently passing None downstream."""
    from types import SimpleNamespace

    from harness_agent.adapters.telegram import download_attachment

    class _Bot:
        async def get_file(self, file_id: str) -> object:
            return SimpleNamespace(file_path=None)

        async def download_file(self, file_path: str) -> object:
            raise AssertionError("download_file must not be called when file_path is None")

    with pytest.raises(RuntimeError, match="no file_path"):
        await download_attachment(
            bot=_Bot(),  # type: ignore[arg-type]
            file_id="abc",
            file_unique_id="uid",
            kind="file",
            file_name="x.txt",
            mime_type=None,
            size_bytes=0,
            chat_id=1,
            message_id=2,
        )


@pytest.mark.asyncio
async def test_download_attachment_raises_when_download_file_returns_none() -> None:
    """download_file() can return None if Telegram closes the connection
    or the file vanished between get_file and download. The adapter
    must surface this as a typed RuntimeError, not a TypeError on the
    next attribute access."""
    from types import SimpleNamespace

    from harness_agent.adapters.telegram import download_attachment

    class _Bot:
        async def get_file(self, file_id: str) -> object:
            return SimpleNamespace(file_path="documents/x.txt")

        async def download_file(self, file_path: str) -> None:
            return None

    with pytest.raises(RuntimeError, match="returned no content"):
        await download_attachment(
            bot=_Bot(),  # type: ignore[arg-type]
            file_id="abc",
            file_unique_id="uid",
            kind="file",
            file_name="x.txt",
            mime_type=None,
            size_bytes=0,
            chat_id=1,
            message_id=2,
        )


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
    await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:test", bus=RecordingBus(), turn_coordinator=coordinator
    )
    bot = RecordingTelegramBot()
    adapter._bot = bot

    await adapter.handle_assistant_text(
        AssistantTextProduced(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            text="*bold*",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.messages == [
        {
            "chat_id": 456,
            "text": "*bold*",
            "parse_mode": "Markdown",
        }
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_falls_back_to_plain_text_for_invalid_markdown() -> None:
    coordinator = ConversationTurnCoordinator()
    await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:test", bus=RecordingBus(), turn_coordinator=coordinator
    )
    bot = RecordingTelegramBot(fail_markdown=True)
    adapter._bot = bot

    await adapter.handle_assistant_text(
        AssistantTextProduced(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            text="Broken *markdown",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.messages == [
        {
            "chat_id": 456,
            "text": "Broken *markdown",
            "parse_mode": "Markdown",
        },
        {
            "chat_id": 456,
            "text": "Broken *markdown",
            "parse_mode": None,
        },
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_shows_typing_on_generation_start() -> None:
    assert hasattr(events, "AgentGenerationStarted")
    adapter = AiogramTelegramAdapter(
        token="123:test", bus=RecordingBus(), turn_coordinator=ConversationTurnCoordinator()
    )
    bot = RecordingTelegramBot()
    adapter._bot = bot

    await adapter.handle_generation_started(
        events.AgentGenerationStarted(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.chat_actions == [
        {
            "chat_id": 456,
            "action": "typing",
        }
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_drops_assistant_text_from_superseded_generation() -> None:
    coordinator = ConversationTurnCoordinator()
    # User sends two messages back-to-back. Generation 2 is now current;
    # generation 1's reply must not reach Telegram.
    await coordinator.request_generation("tg:456")
    await coordinator.request_generation("tg:456")
    adapter = AiogramTelegramAdapter(
        token="123:test", bus=RecordingBus(), turn_coordinator=coordinator
    )
    bot = RecordingTelegramBot()
    adapter._bot = bot

    await adapter.handle_assistant_text(
        AssistantTextProduced(
            user_id="u:123",
            conversation_id="tg:456",
            generation=1,
            text="stale reply",
            reply_target=TelegramReplyTarget(chat_id=456),
        )
    )

    assert bot.messages == []


@pytest.mark.asyncio
async def test_docker_runtime_routes_shell_without_changing_tool_name() -> None:
    runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="ok\n", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=runner, spawned_process_store=InMemorySpawnedProcessStore())

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
    runtime = DockerUserRuntime(runner=runner, spawned_process_store=InMemorySpawnedProcessStore())

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
    runtime = DockerUserRuntime(runner=runner, spawned_process_store=InMemorySpawnedProcessStore())

    assert await runtime.list_skills("u:123") == [
        Skill(
            name="shell-work",
            description="Shell discipline.",
            body="Stay in /workspace.\n",
        )
    ]


@pytest.mark.asyncio
async def test_docker_runtime_recovers_spawned_processes_after_restart(tmp_path: Path) -> None:
    store = SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3")
    spawn_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=spawn_runner, spawned_process_store=store)

    spawned = await runtime.shell_spawn(
        "u:123",
        ShellSpawnInput(command="while true; do echo alive; sleep 1; done"),
    )

    process_id = spawned.stdout
    record = await store.get(process_id=process_id, user_id="u:123")
    assert record is not None
    assert record.command == "while true; do echo alive; sleep 1; done"
    assert record.container_name == "harness-u-123"
    assert record.stdout_offset == 0

    read_payload = {
        "stdout": base64.b64encode(b"alive\n").decode("ascii"),
        "stderr": "",
        "stdout_offset": len("alive\n"),
        "stderr_offset": 0,
        "exit_code": 0,
    }
    read_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout=json.dumps(read_payload), stderr="", exit_code=0)]
    )
    restarted_runtime = DockerUserRuntime(runner=read_runner, spawned_process_store=store)

    read = await restarted_runtime.shell_read(
        "u:123",
        ShellReadInput(process_id=process_id),
    )

    assert read.stdout == "alive\n"
    updated = await store.get(process_id=process_id, user_id="u:123")
    assert updated is not None
    assert updated.stdout_offset == len("alive\n")


@pytest.mark.asyncio
async def test_docker_runtime_removes_spawned_process_record_on_kill(tmp_path: Path) -> None:
    store = SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3")
    spawn_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=spawn_runner, spawned_process_store=store)
    spawned = await runtime.shell_spawn("u:123", ShellSpawnInput(command="sleep 60"))

    kill_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    restarted_runtime = DockerUserRuntime(runner=kill_runner, spawned_process_store=store)

    killed = await restarted_runtime.shell_kill(
        "u:123",
        ShellKillInput(process_id=spawned.stdout),
    )

    assert killed.stdout == f"killed {spawned.stdout}\n"
    assert await store.get(process_id=spawned.stdout, user_id="u:123") is None


@pytest.mark.asyncio
async def test_docker_runtime_rolls_back_spawn_record_when_detached_spawn_fails(
    tmp_path: Path,
) -> None:
    """When the detached `docker exec` for the spawn helper returns
    non-zero, the just-created projection row must be removed; otherwise
    a follow-up shell.read/shell.kill would dispatch against a process
    that never started."""
    store = SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3")
    spawn_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="cannot exec", exit_code=1)]
    )
    runtime = DockerUserRuntime(runner=spawn_runner, spawned_process_store=store)

    result = await runtime.shell_spawn("u:123", ShellSpawnInput(command="false"))

    assert result.exit_code == 1
    assert result.stderr == "cannot exec"
    # No record means no orphan; the projection store is in sync with the
    # actual (non-existent) in-container process.
    assert await _any_record(store, user_id="u:123") is None


@pytest.mark.asyncio
async def test_docker_runtime_keeps_spawned_record_until_output_fully_drained(
    tmp_path: Path,
) -> None:
    """If the in-container process has exited but the agent's max_bytes
    only pulled part of the stdout, the runtime must NOT drop the
    projection row -- otherwise the next shell.read has no paths to read
    the remaining bytes from. Drop happens only when offsets reach file
    size."""
    store = SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3")
    spawn_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=spawn_runner, spawned_process_store=store)
    spawned = await runtime.shell_spawn("u:123", ShellSpawnInput(command="echo done"))
    process_id = spawned.stdout

    # First read: process exited, but offsets are < file size (only 3 of 10
    # bytes returned). The record must survive.
    partial_payload = {
        "stdout": base64.b64encode(b"don").decode("ascii"),
        "stderr": "",
        "stdout_offset": 3,
        "stderr_offset": 0,
        "stdout_size": 10,
        "stderr_size": 0,
        "exit_code": 0,
        "exited": True,
    }
    partial_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout=json.dumps(partial_payload), stderr="", exit_code=0)]
    )
    partial_runtime = DockerUserRuntime(
        runner=partial_runner, spawned_process_store=store
    )
    partial_read = await partial_runtime.shell_read(
        "u:123",
        ShellReadInput(process_id=process_id, max_bytes=3),
    )
    assert partial_read.stdout == "don"
    assert await store.get(process_id=process_id, user_id="u:123") is not None

    # Second read: offsets now match file size, the runtime drops the row.
    final_payload = {
        "stdout": base64.b64encode(b"e\nrest....").decode("ascii"),
        "stderr": "",
        "stdout_offset": 10,
        "stderr_offset": 0,
        "stdout_size": 10,
        "stderr_size": 0,
        "exit_code": 0,
        "exited": True,
    }
    final_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout=json.dumps(final_payload), stderr="", exit_code=0)]
    )
    final_runtime = DockerUserRuntime(
        runner=final_runner, spawned_process_store=store
    )
    final_read = await final_runtime.shell_read(
        "u:123",
        ShellReadInput(process_id=process_id, max_bytes=20),
    )
    assert final_read.stdout == "e\nrest...."
    assert await store.get(process_id=process_id, user_id="u:123") is None


@pytest.mark.asyncio
async def test_docker_runtime_drops_spawned_record_when_shell_read_sees_natural_exit(
    tmp_path: Path,
) -> None:
    """When READ_SPAWNED_PROCESS_CODE reports `exited`, the runtime must
    delete the projection row so a follow-up shell.read/shell.kill cannot
    treat the long-gone process as still live."""
    store = SQLiteSpawnedProcessStore(tmp_path / "runtime.sqlite3")
    spawn_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    )
    runtime = DockerUserRuntime(runner=spawn_runner, spawned_process_store=store)
    spawned = await runtime.shell_spawn("u:123", ShellSpawnInput(command="echo done"))
    process_id = spawned.stdout

    read_payload = {
        "stdout": base64.b64encode(b"done\n").decode("ascii"),
        "stderr": "",
        "stdout_offset": len("done\n"),
        "stderr_offset": 0,
        "stdout_size": len("done\n"),
        "stderr_size": 0,
        "exit_code": 0,
        "exited": True,
    }
    read_runner = RecordingDockerRunner(
        [DockerProcessResult(stdout=json.dumps(read_payload), stderr="", exit_code=0)]
    )
    restarted_runtime = DockerUserRuntime(
        runner=read_runner, spawned_process_store=store
    )

    read = await restarted_runtime.shell_read(
        "u:123",
        ShellReadInput(process_id=process_id),
    )
    assert read.stdout == "done\n"
    assert read.exit_code == 0
    assert await store.get(process_id=process_id, user_id="u:123") is None


async def _any_record(store: SQLiteSpawnedProcessStore, *, user_id: str):
    # Convenience: the SQLite store does not expose a "list by user" API,
    # but the SQL table is small enough in tests that we can inline a
    # direct query. We only need to know whether *any* row leaked.
    import aiosqlite

    async with aiosqlite.connect(store._path) as db:  # type: ignore[attr-defined]
        async with db.execute(
            "select process_id from spawned_processes where user_id = ? limit 1",
            (user_id,),
        ) as cursor:
            return await cursor.fetchone()


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


class RecordingBus:
    async def publish(self, event) -> None:
        self.event = event


class RecordingTelegramBot:
    def __init__(self, *, fail_markdown: bool = False) -> None:
        self._fail_markdown = fail_markdown
        self.messages: list[dict] = []
        self.chat_actions: list[dict] = []

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )
        if self._fail_markdown and parse_mode == "Markdown":
            raise TelegramBadRequest(
                method=SendMessage(chat_id=chat_id, text=text),
                message="can't parse entities",
            )

    async def send_chat_action(self, *, chat_id: int, action: str) -> None:
        self.chat_actions.append(
            {
                "chat_id": chat_id,
                "action": action,
            }
        )
