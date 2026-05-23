"""Shell operations inside the user's Docker workspace.

`exec` is a one-shot foreground command. `spawn`/`read`/`kill` cooperate
to manage long-running background processes whose state survives a
harness restart via the `SpawnedProcessStore`.
"""

import base64
import json
from collections.abc import Awaitable, Callable
from uuid import uuid4

from harness_agent.runtime.models import (
    DockerProcessResult,
    RuntimeToolResult,
    SpawnedProcessRecord,
)
from harness_agent.runtime.paths import workspace_path
from harness_agent.runtime.protocols import SpawnedProcessStore
from harness_agent.runtime.scripts import (
    KILL_SPAWNED_PROCESS_SCRIPT,
    READ_SPAWNED_PROCESS_CODE,
    SPAWN_SHELL_SCRIPT,
)
from harness_agent.tools import (
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
)


ExecInContainer = Callable[..., Awaitable[DockerProcessResult]]
RunInContainer = Callable[..., Awaitable[DockerProcessResult]]
RunInContainerDetached = Callable[..., Awaitable[DockerProcessResult]]
MaybeEnsure = Callable[[str], Awaitable[None]]
ContainerNameFn = Callable[[str], str]


class DockerShell:
    """Foreground and background shell execution.

    `spawn` writes a `SpawnedProcessRecord` before exec so a crash during
    detached startup is recoverable on the next harness boot. `read`
    tracks byte offsets in the store so output is not replayed."""

    def __init__(
        self,
        *,
        exec_in_container: ExecInContainer,
        run_in_container: RunInContainer,
        run_in_container_detached: RunInContainerDetached,
        maybe_ensure: MaybeEnsure,
        container_name: ContainerNameFn,
        spawned_process_store: SpawnedProcessStore,
    ) -> None:
        self._exec = exec_in_container
        self._run_in_container = run_in_container
        self._run_in_container_detached = run_in_container_detached
        self._maybe_ensure = maybe_ensure
        self._container_name = container_name
        self._spawned_processes = spawned_process_store

    async def exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        cwd = workspace_path(input.cwd)
        return await self._exec(
            user_id,
            ["sh", "-lc", input.command],
            workdir=cwd,
            timeout_seconds=input.timeout_seconds,
        )

    async def spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        await self._maybe_ensure(user_id)
        cwd = workspace_path(input.cwd)
        process_id = uuid4().hex
        base_path = f"/workspace/.harness/spawned/{process_id}"
        record = SpawnedProcessRecord(
            process_id=process_id,
            user_id=user_id,
            container_name=self._container_name(user_id),
            command=input.command,
            cwd=cwd,
            base_path=base_path,
            stdout_path=f"{base_path}/stdout",
            stderr_path=f"{base_path}/stderr",
            pid_path=f"{base_path}/pid",
            exit_code_path=f"{base_path}/exit_code",
        )
        await self._spawned_processes.create(record)
        try:
            result = await self._run_in_container_detached(
                record.container_name,
                [
                    "sh",
                    "-c",
                    SPAWN_SHELL_SCRIPT,
                    "spawn",
                    record.base_path,
                    record.stdout_path,
                    record.stderr_path,
                    record.pid_path,
                    record.exit_code_path,
                    record.cwd,
                    record.command,
                ],
                workdir="/workspace",
            )
        except Exception:
            await self._spawned_processes.delete(process_id=process_id, user_id=user_id)
            raise
        if result.exit_code != 0:
            await self._spawned_processes.delete(process_id=process_id, user_id=user_id)
            return result
        return RuntimeToolResult(stdout=process_id)

    async def read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        record = await self._spawned_processes.get(
            process_id=input.process_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        await self._maybe_ensure(user_id)
        result = await self._run_in_container(
            record.container_name,
            [
                "python",
                "-c",
                READ_SPAWNED_PROCESS_CODE,
                record.stdout_path,
                record.stderr_path,
                str(record.stdout_offset),
                str(record.stderr_offset),
                str(input.max_bytes),
                record.pid_path,
                record.exit_code_path,
            ],
        )
        if result.exit_code != 0:
            return result
        payload = json.loads(result.stdout)
        await self._spawned_processes.update_offsets(
            process_id=input.process_id,
            user_id=user_id,
            stdout_offset=int(payload["stdout_offset"]),
            stderr_offset=int(payload["stderr_offset"]),
        )
        return RuntimeToolResult(
            stdout=base64.b64decode(payload["stdout"]).decode("utf-8", errors="replace"),
            stderr=base64.b64decode(payload["stderr"]).decode("utf-8", errors="replace"),
            exit_code=int(payload["exit_code"]),
        )

    async def kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        record = await self._spawned_processes.get(
            process_id=input.process_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        await self._maybe_ensure(user_id)
        result = await self._run_in_container(
            record.container_name,
            [
                "sh",
                "-c",
                KILL_SPAWNED_PROCESS_SCRIPT,
                "kill-spawned",
                record.pid_path,
                record.base_path,
                record.exit_code_path,
            ],
        )
        if result.exit_code != 0:
            return result
        await self._spawned_processes.delete(process_id=input.process_id, user_id=user_id)
        return RuntimeToolResult(stdout=f"killed {input.process_id}\n")
