import asyncio
import base64
import json
import posixpath
import re
import shlex
from pathlib import Path
from pathlib import PurePosixPath
from uuid import uuid4

import yaml
from loguru import logger

from harness_agent.content import WorkspaceFile
from harness_agent.context import AgentFileSet, Skill
from harness_agent.mcp_models import McpServerConfig
from harness_agent.runtime.docker_runner import AsyncioDockerRunner
from harness_agent.runtime.models import (
    DockerProcessResult,
    RuntimeFileRead,
    RuntimeToolResult,
    SpawnedProcessRecord,
)
from harness_agent.runtime.paths import content_path, workspace_path
from harness_agent.runtime.protocols import DockerRunner, SpawnedProcessStore, UserRuntime
from harness_agent.runtime.scripts import (
    KILL_SPAWNED_PROCESS_SCRIPT,
    READ_SPAWNED_PROCESS_CODE,
    SPAWN_SHELL_SCRIPT,
)
from harness_agent.runtime.skills import parse_skill_markdown
from harness_agent.runtime.spawned_store import SQLiteSpawnedProcessStore
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
    ShellKillInput,
    ShellReadInput,
    ShellSpawnInput,
)


class DockerUserRuntime(UserRuntime):
    def __init__(
        self,
        *,
        runner: DockerRunner | None = None,
        spawned_process_store: SpawnedProcessStore | None = None,
        image: str = "python:3.14-slim",
        container_prefix: str = "harness",
        ensure_container: bool = False,
        network: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
    ) -> None:
        if runner is None:
            runner = AsyncioDockerRunner()
        self._runner = runner
        self._image = image
        self._container_prefix = container_prefix
        self._ensure_container = ensure_container
        self._network = network
        self._memory = memory
        self._cpus = cpus
        self._spawned_processes = (
            spawned_process_store
            if spawned_process_store is not None
            else SQLiteSpawnedProcessStore(Path("./data/harness.runtime.sqlite3"))
        )

    async def ensure_user_container(self, user_id: str) -> None:
        name = self._container_name(user_id)
        inspect = await self._runner.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name]
        )
        if inspect.exit_code == 0:
            if inspect.stdout.strip() != "true":
                await self._must_run(["docker", "start", name])
            await self._bootstrap_workspace(user_id)
            return

        argv = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-v",
            f"{name}-workspace:/workspace",
            "-w",
            "/workspace",
        ]
        if self._network is not None:
            argv.extend(["--network", self._network])
        if self._memory is not None:
            argv.extend(["--memory", self._memory])
        if self._cpus is not None:
            argv.extend(["--cpus", self._cpus])
        argv.extend([self._image, "sleep", "infinity"])
        await self._must_run(argv)
        await self._bootstrap_workspace(user_id)

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        return AgentFileSet(
            soul=await self._read_text(user_id, "/workspace/agent/SOUL.md"),
            agents=await self._read_text(user_id, "/workspace/agent/AGENTS.md"),
            user=await self._read_text(user_id, "/workspace/agent/USER.md"),
            tools=await self._read_text(user_id, "/workspace/agent/TOOLS.md"),
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/skills -name SKILL.md -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning("Failed to list skills for {user_id}: {stderr}", user_id=user_id, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        skills: list[Skill] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._read_text(user_id, path)
            skills.append(
                parse_skill_markdown(text, file_name=PurePosixPath(path).parent.name)
            )
        return skills

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/mcp -name '*.yaml' -type f | sort",
            ],
        )
        if result.exit_code != 0:
            logger.warning("Failed to list MCP servers for {user_id}: {stderr}", user_id=user_id, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        servers: list[McpServerConfig] = []
        for path in [line for line in result.stdout.splitlines() if line.strip()]:
            text = await self._read_text(user_id, path)
            servers.append(McpServerConfig.model_validate(yaml.safe_load(text)))
        return servers

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        cwd = workspace_path(input.cwd)
        return await self._exec(
            user_id,
            ["sh", "-lc", input.command],
            workdir=cwd,
            timeout_seconds=input.timeout_seconds,
        )

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        path = content_path(path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=content,
        )

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
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

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
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

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
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

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"head -c {int(input.max_bytes)} -- {shlex.quote(path)}"],
        )

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        path = workspace_path(path)
        code = (
            "import base64, pathlib, sys;"
            "path=pathlib.Path(sys.argv[1]);"
            "limit=sys.argv[2];"
            "stream=path.open('rb');"
            "data=stream.read() if limit == '' else stream.read(int(limit));"
            "stream.close();"
            "sys.stdout.write(base64.b64encode(data).decode('ascii'))"
        )
        limit = "" if max_bytes is None else str(max_bytes)
        result = await self._exec(
            user_id,
            ["python", "-c", code, path, limit],
        )
        if result.exit_code != 0:
            return RuntimeFileRead(result=result)
        return RuntimeFileRead(
            file=WorkspaceFile(
                path=path,
                content=base64.b64decode(result.stdout),
            )
        )

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=input.content.encode("utf-8"),
        )

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        current = await self.file_read(
            user_id,
            FileReadInput(path=input.path, max_bytes=5_000_000),
        )
        if current.exit_code != 0:
            return current
        if input.old not in current.stdout:
            raise ValueError("old text not found")
        count = -1 if input.replace_all else 1
        updated = current.stdout.replace(input.old, input.new, count)
        return await self.file_write(user_id, FileWriteInput(path=input.path, content=updated))

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        current = await self.file_read(
            user_id,
            FileReadInput(path=input.path, max_bytes=5_000_000),
        )
        if current.exit_code != 0:
            return current
        updated = current.stdout
        for edit in input.edits:
            if edit.old not in updated:
                raise ValueError(f"old text not found: {edit.old}")
            count = -1 if edit.replace_all else 1
            updated = updated.replace(edit.old, edit.new, count)
        return await self.file_write(user_id, FileWriteInput(path=input.path, content=updated))

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        cwd = workspace_path(input.cwd)
        code = (
            "import glob, sys;"
            "pattern=sys.argv[1];"
            "print('\\n'.join(sorted(glob.glob(pattern, recursive=True))))"
        )
        return await self._exec(
            user_id,
            ["python", "-c", code, input.pattern],
            workdir=cwd,
        )

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        command = (
            f"grep -RIn -- {shlex.quote(input.pattern)} {shlex.quote(path)}; "
            "code=$?; "
            'if [ "$code" -eq 1 ]; then exit 0; fi; '
            'exit "$code"'
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        command = (
            f"find {shlex.quote(path)} -maxdepth 1 -mindepth 1 "
            "| sort"
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def _read_text(self, user_id: str, path: str) -> str:
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(workspace_path(path))}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

    async def _bootstrap_workspace(self, user_id: str) -> None:
        script = """
mkdir -p /workspace/agent /workspace/skills /workspace/files /workspace/content
mkdir -p /workspace/mcp
touch /workspace/agent/SOUL.md /workspace/agent/AGENTS.md /workspace/agent/USER.md /workspace/agent/TOOLS.md
"""
        await self._must_run(
            [
                "docker",
                "exec",
                self._container_name(user_id),
                "sh",
                "-lc",
                script,
            ]
        )

    async def _exec(
        self,
        user_id: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        await self._maybe_ensure(user_id)
        return await self._run_in_container(
            self._container_name(user_id),
            argv,
            workdir=workdir,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )

    async def _run_in_container(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        docker_argv = ["docker", "exec"]
        if stdin is not None:
            docker_argv.append("-i")
        if workdir is not None:
            docker_argv.extend(["-w", workdir])
        docker_argv.append(container_name)
        docker_argv.extend(argv)
        return await self._runner.run(
            docker_argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )

    async def _run_in_container_detached(
        self,
        container_name: str,
        argv: list[str],
        *,
        workdir: str | None = None,
    ) -> DockerProcessResult:
        docker_argv = ["docker", "exec", "-d"]
        if workdir is not None:
            docker_argv.extend(["-w", workdir])
        docker_argv.append(container_name)
        docker_argv.extend(argv)
        return await self._runner.run(docker_argv)

    async def open_stdio(
        self,
        *,
        user_id: str,
        argv: list[str],
        cwd: str = "/workspace",
    ) -> asyncio.subprocess.Process:
        await self._maybe_ensure(user_id)
        return await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-w",
            workspace_path(cwd),
            self._container_name(user_id),
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _must_run(self, argv: list[str]) -> DockerProcessResult:
        result = await self._runner.run(argv)
        if result.exit_code != 0:
            logger.error("Docker command failed: {argv} stderr={stderr}", argv=argv, stderr=result.stderr)
            raise RuntimeError(result.stderr)
        return result

    async def _maybe_ensure(self, user_id: str) -> None:
        if self._ensure_container:
            await self.ensure_user_container(user_id)

    def _container_name(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", user_id).strip("-")
        return f"{self._container_prefix}-{safe_user_id}"
