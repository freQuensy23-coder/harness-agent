from pathlib import Path

import pytest

from harness_agent.bus import EventBus
from harness_agent.events import (
    ShellProcessOutputAdvanced,
    ShellProcessSpawned,
    ShellProcessTerminated,
)
from harness_agent.runtime import (
    InMemorySpawnedProcessStore,
    SpawnedProcessRecord,
)
from harness_agent.runtime.eventful_spawned_store import EventfulSpawnedProcessStore
from harness_agent.store import SQLiteEventStore


def _record() -> SpawnedProcessRecord:
    return SpawnedProcessRecord(
        process_id="p-1",
        user_id="u:1",
        container_name="harness-u-1",
        command="tail -f log",
        cwd="/workspace",
        base_path="/workspace/.harness/spawned/p-1",
        stdout_path="/workspace/.harness/spawned/p-1/stdout",
        stderr_path="/workspace/.harness/spawned/p-1/stderr",
        pid_path="/workspace/.harness/spawned/p-1/pid",
        exit_code_path="/workspace/.harness/spawned/p-1/exit_code",
    )


@pytest.mark.asyncio
async def test_create_publishes_spawned_event_before_store_mutation(
    tmp_path: Path,
) -> None:
    inner = InMemorySpawnedProcessStore()
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    decorated = EventfulSpawnedProcessStore(store=inner, bus=bus)
    observed: list[bool] = []

    async def watch(_event: ShellProcessSpawned) -> tuple:
        observed.append(
            await inner.get(process_id="p-1", user_id="u:1") is None
        )
        return ()

    bus.subscribe(ShellProcessSpawned, watch)

    await decorated.create(_record())

    assert observed == [True]
    assert await inner.get(process_id="p-1", user_id="u:1") is not None


@pytest.mark.asyncio
async def test_update_offsets_publishes_event_carrying_offsets(
    tmp_path: Path,
) -> None:
    inner = InMemorySpawnedProcessStore()
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    decorated = EventfulSpawnedProcessStore(store=inner, bus=bus)
    await decorated.create(_record())

    advanced: list[ShellProcessOutputAdvanced] = []

    async def watch(event: ShellProcessOutputAdvanced) -> tuple:
        advanced.append(event)
        return ()

    bus.subscribe(ShellProcessOutputAdvanced, watch)

    await decorated.update_offsets(
        process_id="p-1",
        user_id="u:1",
        stdout_offset=42,
        stderr_offset=7,
    )

    assert len(advanced) == 1
    assert advanced[0].stdout_offset == 42
    assert advanced[0].stderr_offset == 7
    stored = await inner.get(process_id="p-1", user_id="u:1")
    assert stored is not None
    assert stored.stdout_offset == 42
    assert stored.stderr_offset == 7


@pytest.mark.asyncio
async def test_delete_publishes_terminated_event_with_killed_reason(
    tmp_path: Path,
) -> None:
    inner = InMemorySpawnedProcessStore()
    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    decorated = EventfulSpawnedProcessStore(store=inner, bus=bus)
    await decorated.create(_record())

    terminated: list[ShellProcessTerminated] = []

    async def watch(event: ShellProcessTerminated) -> tuple:
        terminated.append(event)
        return ()

    bus.subscribe(ShellProcessTerminated, watch)

    await decorated.delete(process_id="p-1", user_id="u:1")

    assert len(terminated) == 1
    assert terminated[0].process_id == "p-1"
    assert terminated[0].reason == "killed"
    assert await inner.get(process_id="p-1", user_id="u:1") is None


@pytest.mark.asyncio
async def test_runtime_publishes_spawn_failed_reason_when_detached_spawn_fails(
    tmp_path: Path,
) -> None:
    """End-to-end through DockerUserRuntime + EventfulSpawnedProcessStore:
    a failed detached spawn must emit ShellProcessTerminated with
    reason='spawn_failed', not 'killed'."""
    from harness_agent.events import ShellProcessTerminated
    from harness_agent.runtime import DockerProcessResult, DockerUserRuntime
    from harness_agent.tools import ShellSpawnInput

    class _FailingRunner:
        async def run(self, argv: list[str], **_: object) -> DockerProcessResult:
            return DockerProcessResult(stdout="", stderr="cannot exec", exit_code=1)

    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    inner = InMemorySpawnedProcessStore()
    decorated = EventfulSpawnedProcessStore(store=inner, bus=bus)
    terminated: list[ShellProcessTerminated] = []

    async def watch(event: ShellProcessTerminated) -> tuple:
        terminated.append(event)
        return ()

    bus.subscribe(ShellProcessTerminated, watch)

    runtime = DockerUserRuntime(
        runner=_FailingRunner(),  # type: ignore[arg-type]
        spawned_process_store=decorated,
    )
    result = await runtime.shell_spawn("u:1", ShellSpawnInput(command="bad"))

    assert result.exit_code == 1
    assert len(terminated) == 1
    assert terminated[0].reason == "spawn_failed"


@pytest.mark.asyncio
async def test_runtime_publishes_exited_reason_when_shell_read_observes_natural_exit(
    tmp_path: Path,
) -> None:
    """End-to-end through DockerUserRuntime + EventfulSpawnedProcessStore:
    when shell.read returns an exited+drained payload, the runtime drops
    the row via the decorated store and the event log carries
    reason='exited'."""
    import base64
    import json

    from harness_agent.events import (
        ShellProcessSpawned,
        ShellProcessTerminated,
    )
    from harness_agent.runtime import DockerProcessResult, DockerUserRuntime
    from harness_agent.tools import ShellReadInput, ShellSpawnInput

    bus = EventBus(SQLiteEventStore(tmp_path / "events.sqlite3"))
    inner = InMemorySpawnedProcessStore()
    decorated = EventfulSpawnedProcessStore(store=inner, bus=bus)
    terminated: list[ShellProcessTerminated] = []
    spawned_events: list[ShellProcessSpawned] = []

    async def on_term(event: ShellProcessTerminated) -> tuple:
        terminated.append(event)
        return ()

    async def on_spawn(event: ShellProcessSpawned) -> tuple:
        spawned_events.append(event)
        return ()

    bus.subscribe(ShellProcessTerminated, on_term)
    bus.subscribe(ShellProcessSpawned, on_spawn)

    spawn_payloads = [DockerProcessResult(stdout="", stderr="", exit_code=0)]
    drain_payload = {
        "stdout": base64.b64encode(b"done\n").decode("ascii"),
        "stderr": "",
        "stdout_offset": len("done\n"),
        "stderr_offset": 0,
        "stdout_size": len("done\n"),
        "stderr_size": 0,
        "exit_code": 0,
        "exited": True,
    }

    class _ScriptedRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self._spawn_left = list(spawn_payloads)
            self._read_payload = json.dumps(drain_payload)

        async def run(self, argv: list[str], **_: object) -> DockerProcessResult:
            self.calls.append(argv)
            # First call is the detached spawn; later docker exec runs are
            # shell.read which prints the read payload as JSON stdout.
            if self._spawn_left:
                return self._spawn_left.pop(0)
            return DockerProcessResult(
                stdout=self._read_payload, stderr="", exit_code=0
            )

    runner = _ScriptedRunner()
    runtime = DockerUserRuntime(
        runner=runner,  # type: ignore[arg-type]
        spawned_process_store=decorated,
    )
    spawned = await runtime.shell_spawn("u:1", ShellSpawnInput(command="echo done"))
    assert len(spawned_events) == 1
    process_id = spawned.stdout

    read = await runtime.shell_read(
        "u:1", ShellReadInput(process_id=process_id, max_bytes=100)
    )
    assert read.stdout == "done\n"
    assert len(terminated) == 1
    assert terminated[0].reason == "exited"
    assert terminated[0].process_id == process_id


@pytest.mark.asyncio
async def test_event_log_lets_us_reconstruct_lifecycle(tmp_path: Path) -> None:
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    bus = EventBus(event_store)
    decorated = EventfulSpawnedProcessStore(
        store=InMemorySpawnedProcessStore(), bus=bus
    )
    await decorated.create(_record())
    await decorated.update_offsets(
        process_id="p-1", user_id="u:1", stdout_offset=10, stderr_offset=2
    )
    await decorated.delete(process_id="p-1", user_id="u:1")

    persisted = [e.type for e in await event_store.list_events()]
    assert persisted == [
        "shell.process.spawned",
        "shell.process.output_advanced",
        "shell.process.terminated",
    ]
