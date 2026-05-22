"""shell.spawn / shell.read / shell.kill lifecycle, including the
detached-spawn rollback and natural-exit projection cleanup.

State lives in the injected SpawnedProcessStore (typically wrapped by
EventfulSpawnedProcessStore so each mutation publishes a typed event).
"""

import base64
import json
from typing import Any
from uuid import uuid4

from harness_agent.runtime.container import DockerContainerExecutor
from harness_agent.runtime.models import RuntimeToolResult, SpawnedProcessRecord
from harness_agent.runtime.paths import workspace_path
from harness_agent.runtime.protocols import SpawnedProcessStore
from harness_agent.runtime.scripts import (
    KILL_SPAWNED_PROCESS_SCRIPT,
    READ_SPAWNED_PROCESS_CODE,
    SPAWN_SHELL_SCRIPT,
)
from harness_agent.tools import ShellKillInput, ShellReadInput, ShellSpawnInput


class DockerSpawnedProcesses:
    def __init__(
        self,
        *,
        executor: DockerContainerExecutor,
        store: SpawnedProcessStore,
    ) -> None:
        self._executor = executor
        self._store = store

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        if self._executor._ensure_container:  # type: ignore[attr-defined]
            await self._executor.ensure(user_id)
        cwd = workspace_path(input.cwd)
        process_id = uuid4().hex
        base_path = f"/workspace/.harness/spawned/{process_id}"
        record = SpawnedProcessRecord(
            process_id=process_id,
            user_id=user_id,
            container_name=self._executor.container_name(user_id),
            command=input.command,
            cwd=cwd,
            base_path=base_path,
            stdout_path=f"{base_path}/stdout",
            stderr_path=f"{base_path}/stderr",
            pid_path=f"{base_path}/pid",
            exit_code_path=f"{base_path}/exit_code",
        )
        await self._store.create(record)
        try:
            # Run the wrapper under setsid so its pid IS the pgid of the
            # whole subtree (wrapper + child shell + any descendants the
            # user command forks). shell.kill targets that group with
            # `kill -TERM/-KILL -pid` so nothing leaks.
            result = await self._executor.exec_detached(
                record.container_name,
                [
                    "setsid",
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
            await _delete_with_reason(
                self._store,
                process_id=process_id,
                user_id=user_id,
                reason="spawn_failed",
            )
            raise
        if result.exit_code != 0:
            await _delete_with_reason(
                self._store,
                process_id=process_id,
                user_id=user_id,
                reason="spawn_failed",
            )
            return result
        return RuntimeToolResult(stdout=process_id)

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        record = await self._store.get(
            process_id=input.process_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        if self._executor._ensure_container:  # type: ignore[attr-defined]
            await self._executor.ensure(user_id)
        result = await self._executor.exec_in_container(
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
        await self._store.update_offsets(
            process_id=input.process_id,
            user_id=user_id,
            stdout_offset=int(payload["stdout_offset"]),
            stderr_offset=int(payload["stderr_offset"]),
        )
        exit_code = int(payload["exit_code"])
        stdout_text = base64.b64decode(payload["stdout"]).decode("utf-8", errors="replace")
        stderr_text = base64.b64decode(payload["stderr"]).decode("utf-8", errors="replace")
        if _process_has_exited(payload) and _output_fully_drained(payload):
            # Natural termination AND all output drained: only now drop the
            # projection row. Keeping it alive while bytes remain lets the
            # next shell.read pull the rest before we forget the paths.
            await _delete_with_reason(
                self._store,
                process_id=input.process_id,
                user_id=user_id,
                reason="exited",
            )
        return RuntimeToolResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
        )

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        record = await self._store.get(
            process_id=input.process_id,
            user_id=user_id,
        )
        if record is None:
            raise KeyError(f"Unknown process: {input.process_id}")
        if self._executor._ensure_container:  # type: ignore[attr-defined]
            await self._executor.ensure(user_id)
        result = await self._executor.exec_in_container(
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
        await self._store.delete(process_id=input.process_id, user_id=user_id)
        return RuntimeToolResult(stdout=f"killed {input.process_id}\n")


def _process_has_exited(payload: dict[str, Any]) -> bool:
    return bool(payload.get("exited"))


def _output_fully_drained(payload: dict[str, Any]) -> bool:
    try:
        return (
            int(payload["stdout_offset"]) >= int(payload["stdout_size"])
            and int(payload["stderr_offset"]) >= int(payload["stderr_size"])
        )
    except (KeyError, TypeError, ValueError):
        return True


async def _delete_with_reason(
    store: SpawnedProcessStore,
    *,
    process_id: str,
    user_id: str,
    reason: str,
) -> None:
    # EventfulSpawnedProcessStore accepts a `reason` kwarg; the plain
    # SpawnedProcessStore protocol does not. Try the richer call first
    # and fall back so non-eventful implementations still work.
    delete = store.delete
    try:
        await delete(process_id=process_id, user_id=user_id, reason=reason)  # type: ignore[call-arg]
    except TypeError:
        await delete(process_id=process_id, user_id=user_id)
