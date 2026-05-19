import asyncio
import base64
import json
import posixpath
import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Protocol
from uuid import uuid4

import aiosqlite
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from harness_agent.context import AgentFileSet, Skill, UserContextRuntime
from harness_agent.content import WorkspaceFile
from harness_agent.mcp_models import McpServerConfig
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


class RuntimeToolResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def render_for_llm(self, tool_name: str) -> str:
        return (
            f"{tool_name} stdout:\n{self.stdout}"
            f"stderr:\n{self.stderr}\n"
            f"exit_code: {self.exit_code}"
        )


class DockerProcessResult(RuntimeToolResult):
    pass


class RuntimeFileRead(BaseModel):
    file: WorkspaceFile | None = None
    result: RuntimeToolResult = Field(default_factory=RuntimeToolResult)


class SkillFrontmatter(BaseModel):
    name: str
    description: str


class DockerRunner(Protocol):
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        pass


class SpawnedProcessRecord(BaseModel):
    process_id: str
    user_id: str
    container_name: str
    command: str
    cwd: str
    base_path: str
    stdout_path: str
    stderr_path: str
    pid_path: str
    exit_code_path: str
    stdout_offset: int = 0
    stderr_offset: int = 0


class SpawnedProcessStore(Protocol):
    async def create(self, record: SpawnedProcessRecord) -> SpawnedProcessRecord:
        pass

    async def get(self, *, process_id: str, user_id: str) -> SpawnedProcessRecord | None:
        pass

    async def update_offsets(
        self,
        *,
        process_id: str,
        user_id: str,
        stdout_offset: int,
        stderr_offset: int,
    ) -> None:
        pass

    async def delete(self, *, process_id: str, user_id: str) -> None:
        pass


class SQLiteSpawnedProcessStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def create(self, record: SpawnedProcessRecord) -> SpawnedProcessRecord:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                insert into spawned_processes (
                    process_id,
                    user_id,
                    container_name,
                    command,
                    cwd,
                    base_path,
                    stdout_path,
                    stderr_path,
                    pid_path,
                    exit_code_path,
                    stdout_offset,
                    stderr_offset
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _spawned_process_row(record),
            )
            await db.commit()
        return record

    async def get(self, *, process_id: str, user_id: str) -> SpawnedProcessRecord | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            rows = await db.execute_fetchall(
                """
                select
                    process_id,
                    user_id,
                    container_name,
                    command,
                    cwd,
                    base_path,
                    stdout_path,
                    stderr_path,
                    pid_path,
                    exit_code_path,
                    stdout_offset,
                    stderr_offset
                from spawned_processes
                where process_id = ? and user_id = ?
                """,
                (process_id, user_id),
            )
        if not rows:
            return None
        return _spawned_process_from_row(rows[0])

    async def update_offsets(
        self,
        *,
        process_id: str,
        user_id: str,
        stdout_offset: int,
        stderr_offset: int,
    ) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                update spawned_processes
                set stdout_offset = ?, stderr_offset = ?
                where process_id = ? and user_id = ?
                """,
                (stdout_offset, stderr_offset, process_id, user_id),
            )
            await db.commit()

    async def delete(self, *, process_id: str, user_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                delete from spawned_processes
                where process_id = ? and user_id = ?
                """,
                (process_id, user_id),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                create table if not exists spawned_processes (
                    process_id text not null,
                    user_id text not null,
                    container_name text not null,
                    command text not null,
                    cwd text not null,
                    base_path text not null,
                    stdout_path text not null,
                    stderr_path text not null,
                    pid_path text not null,
                    exit_code_path text not null,
                    stdout_offset integer not null,
                    stderr_offset integer not null,
                    primary key (process_id, user_id)
                )
                """
            )
            await db.commit()


class AsyncioDockerRunner:
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if timeout_seconds is None:
            stdout, stderr = await process.communicate(stdin)
            return DockerProcessResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin),
            timeout=timeout_seconds,
        )
        return DockerProcessResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )


class UserRuntime(UserContextRuntime):
    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        raise NotImplementedError

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        raise NotImplementedError

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        raise NotImplementedError

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        raise NotImplementedError


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
        cwd = _workspace_path(input.cwd)
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
        path = _content_path(path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=content,
        )

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        await self._maybe_ensure(user_id)
        cwd = _workspace_path(input.cwd)
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
                    _SPAWN_SHELL_SCRIPT,
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
                _READ_SPAWNED_PROCESS_CODE,
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
                _KILL_SPAWNED_PROCESS_SCRIPT,
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
        path = _workspace_path(input.path)
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
        path = _workspace_path(path)
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
        path = _workspace_path(input.path)
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
        cwd = _workspace_path(input.cwd)
        code = (
            "import glob, sys;"
            "pattern=sys.argv[1]; limit=int(sys.argv[2]);"
            "print('\\n'.join(sorted(glob.glob(pattern, recursive=True))[:limit]))"
        )
        return await self._exec(
            user_id,
            ["python", "-c", code, input.pattern, str(input.max_results)],
            workdir=cwd,
        )

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        command = (
            f"grep -RIn -- {shlex.quote(input.pattern)} {shlex.quote(path)} "
            f"| head -n {int(input.max_results)}"
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        path = _workspace_path(input.path)
        command = (
            f"find {shlex.quote(path)} -maxdepth 1 -mindepth 1 "
            f"| sort | head -n {int(input.max_results)}"
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def _read_text(self, user_id: str, path: str) -> str:
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(_workspace_path(path))}"],
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
            _workspace_path(cwd),
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


class FakeUserRuntime(UserRuntime):
    def __init__(
        self,
        *,
        files: dict[str, str | bytes] | None = None,
        agent_files: AgentFileSet | None = None,
        skills: Sequence[Skill] = (),
        shell_results: Sequence[RuntimeToolResult] = (),
    ) -> None:
        self._files = {} if files is None else files
        self._agent_files = agent_files
        self._skills = list(skills)
        self._shell_results = list(shell_results)
        self.read_agent_files_calls: list[str] = []
        self.list_skills_calls: list[str] = []
        self.shell_exec_calls: list[ShellExecInput] = []
        self.file_write_calls: list[FileWriteInput] = []
        self.content_write_calls: list[tuple[str, bytes]] = []

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        self.read_agent_files_calls.append(user_id)
        if self._agent_files is not None:
            return self._agent_files
        required = [
            "/workspace/agent/SOUL.md",
            "/workspace/agent/AGENTS.md",
            "/workspace/agent/USER.md",
            "/workspace/agent/TOOLS.md",
        ]
        missing = [path for path in required if path not in self._files]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        return AgentFileSet(
            soul=self._files["/workspace/agent/SOUL.md"],
            agents=self._files["/workspace/agent/AGENTS.md"],
            user=self._files["/workspace/agent/USER.md"],
            tools=self._files["/workspace/agent/TOOLS.md"],
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        self.list_skills_calls.append(user_id)
        return self._skills

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        return []

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        self.content_write_calls.append((path, content))
        self._files[path] = content
        return RuntimeToolResult()

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        if path not in self._files:
            return RuntimeFileRead(
                result=RuntimeToolResult(
                    stderr=f"No such file: {path}\n",
                    exit_code=1,
                )
            )
        content = self._files[path]
        try:
            data = content.encode("latin1")
        except AttributeError:
            data = content
        if max_bytes is None:
            return RuntimeFileRead(file=WorkspaceFile(path=path, content=data))
        return RuntimeFileRead(file=WorkspaceFile(path=path, content=data[:max_bytes]))

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        self.shell_exec_calls.append(input)
        if self._shell_results:
            return self._shell_results.pop(0)
        return RuntimeToolResult()

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        return RuntimeToolResult(stdout="fake-process")

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        return RuntimeToolResult()

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        return RuntimeToolResult()

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        content = self._files[input.path]
        try:
            return RuntimeToolResult(stdout=content.decode("utf-8", errors="replace"))
        except AttributeError:
            pass
        return RuntimeToolResult(stdout=content)

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        self.file_write_calls.append(input)
        self._files[input.path] = input.content
        return RuntimeToolResult()

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        content = self._files[input.path]
        count = -1 if input.replace_all else 1
        self._files[input.path] = content.replace(input.old, input.new, count)
        return RuntimeToolResult()

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        content = self._files[input.path]
        for edit in input.edits:
            count = -1 if edit.replace_all else 1
            content = content.replace(edit.old, edit.new, count)
        self._files[input.path] = content
        return RuntimeToolResult()

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        return RuntimeToolResult(
            stdout="\n".join(path for path in sorted(self._files) if path.startswith(input.cwd))
        )

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        lines: list[str] = []
        for path, content in sorted(self._files.items()):
            if not path.startswith(input.path):
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if input.pattern in line:
                    lines.append(f"{path}:{line_no}:{line}")
        return RuntimeToolResult(stdout="\n".join(lines[: input.max_results]))

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        return RuntimeToolResult(
            stdout="\n".join(path for path in sorted(self._files) if path.startswith(input.path))
        )


def _spawned_process_row(record: SpawnedProcessRecord) -> tuple:
    return (
        record.process_id,
        record.user_id,
        record.container_name,
        record.command,
        record.cwd,
        record.base_path,
        record.stdout_path,
        record.stderr_path,
        record.pid_path,
        record.exit_code_path,
        record.stdout_offset,
        record.stderr_offset,
    )


def _spawned_process_from_row(row: tuple) -> SpawnedProcessRecord:
    return SpawnedProcessRecord(
        process_id=row[0],
        user_id=row[1],
        container_name=row[2],
        command=row[3],
        cwd=row[4],
        base_path=row[5],
        stdout_path=row[6],
        stderr_path=row[7],
        pid_path=row[8],
        exit_code_path=row[9],
        stdout_offset=row[10],
        stderr_offset=row[11],
    )


_SPAWN_SHELL_SCRIPT = """
set -u
base_path=$1
stdout_path=$2
stderr_path=$3
pid_path=$4
exit_code_path=$5
cwd=$6
command=$7

mkdir -p "$base_path"
: > "$stdout_path"
: > "$stderr_path"
rm -f "$exit_code_path"
echo $$ > "$pid_path"

if ! cd "$cwd"; then
    printf '1' > "$exit_code_path"
    exit 1
fi

child=
terminate() {
    if [ -n "${child:-}" ]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    printf '143' > "$exit_code_path"
    exit 143
}
trap terminate TERM INT HUP

sh -lc "$command" > "$stdout_path" 2> "$stderr_path" < /dev/null &
child=$!
wait "$child"
code=$?
printf '%s' "$code" > "$exit_code_path"
exit "$code"
"""


_READ_SPAWNED_PROCESS_CODE = """
import base64
import json
import os
import pathlib
import sys

stdout_path = pathlib.Path(sys.argv[1])
stderr_path = pathlib.Path(sys.argv[2])
stdout_offset = int(sys.argv[3])
stderr_offset = int(sys.argv[4])
max_bytes = int(sys.argv[5])
pid_path = pathlib.Path(sys.argv[6])
exit_code_path = pathlib.Path(sys.argv[7])

def read_available(path, offset):
    if not path.exists():
        return "", offset
    size = path.stat().st_size
    start = min(offset, size)
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(max_bytes)
        next_offset = stream.tell()
    return base64.b64encode(data).decode("ascii"), next_offset

def is_running():
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

stdout, next_stdout_offset = read_available(stdout_path, stdout_offset)
stderr, next_stderr_offset = read_available(stderr_path, stderr_offset)
exit_code = 0
if exit_code_path.exists():
    try:
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
    except ValueError:
        exit_code = 1
elif not is_running():
    exit_code = 1

print(json.dumps({
    "stdout": stdout,
    "stderr": stderr,
    "stdout_offset": next_stdout_offset,
    "stderr_offset": next_stderr_offset,
    "exit_code": exit_code,
}))
"""


_KILL_SPAWNED_PROCESS_SCRIPT = """
set -u
pid_path=$1
base_path=$2
exit_code_path=$3

if [ -f "$pid_path" ]; then
    pid=$(cat "$pid_path" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        i=0
        while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do
            sleep 0.1
            i=$((i + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
fi

printf '143' > "$exit_code_path" 2>/dev/null || true
rm -rf "$base_path"
"""


def parse_skill_markdown(text: str, *, file_name: str) -> Skill:
    if text.startswith("---\n"):
        _, frontmatter, body = text.split("---", 2)
        metadata = SkillFrontmatter.model_validate(yaml.safe_load(frontmatter))
        return Skill(
            name=metadata.name,
            description=metadata.description,
            body=body.lstrip("\n"),
        )
    raise ValueError(f"Missing skill frontmatter in {file_name}")


def _workspace_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if not normalized.startswith("/"):
        normalized = posixpath.normpath(f"/workspace/{normalized}")
    if normalized != "/workspace" and not normalized.startswith("/workspace/"):
        raise ValueError(f"path must stay inside /workspace: {path}")
    return normalized


def _content_path(path: str) -> str:
    normalized = _workspace_path(path)
    if normalized != "/workspace/content" and not normalized.startswith("/workspace/content/"):
        raise ValueError(f"content path must stay inside /workspace/content: {path}")
    return normalized


async def _read_available_now(
    reader: asyncio.StreamReader | None,
    max_bytes: int,
) -> str:
    if reader is None:
        raise RuntimeError("process stream is not available")
    if not reader._buffer:
        return ""
    data = await reader.read(max_bytes)
    return data.decode("utf-8", errors="replace")
