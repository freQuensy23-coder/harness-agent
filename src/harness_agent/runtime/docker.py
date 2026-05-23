"""Docker-backed `UserRuntime` facade.

Owns the per-user container lifecycle and the `docker exec` plumbing;
delegates files, shell, skills, MCP discovery, memory, and session-log
work to the focused helper modules in this package.
"""

import asyncio
import shlex
from pathlib import Path

from loguru import logger

from harness_agent.context import AgentFileSet, Skill
from harness_agent.mcp_models import McpServerConfig
from harness_agent.memory import MemoryTarget
from harness_agent.runtime.docker_files import DockerFiles
from harness_agent.runtime.docker_mcp_discovery import DockerMcpDiscovery
from harness_agent.runtime.docker_runner import AsyncioDockerRunner
from harness_agent.runtime.docker_shell import DockerShell
from harness_agent.runtime.docker_skills import DockerSkills
from harness_agent.runtime.docker_state_files import DockerMemoryFiles, DockerSessionLog
from harness_agent.runtime.models import (
    DockerProcessResult,
    RuntimeFileRead,
    RuntimeToolResult,
)
from harness_agent.runtime.paths import safe_docker_user_part, workspace_path
from harness_agent.runtime.protocols import DockerRunner, SpawnedProcessStore, UserRuntime
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
        self._memory_files = DockerMemoryFiles(exec_in_container=self._exec)
        self._session_log = DockerSessionLog(exec_in_container=self._exec)
        self._files = DockerFiles(exec_in_container=self._exec)
        self._shell = DockerShell(
            exec_in_container=self._exec,
            run_in_container=self._run_in_container,
            run_in_container_detached=self._run_in_container_detached,
            maybe_ensure=self._maybe_ensure,
            container_name=self._container_name,
            spawned_process_store=self._spawned_processes,
        )
        self._skills = DockerSkills(
            exec_in_container=self._exec,
            read_text=self._read_text,
        )
        self._mcp_discovery = DockerMcpDiscovery(
            exec_in_container=self._exec,
            read_text=self._read_text,
        )

    # -- container lifecycle -------------------------------------------------

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

    # -- context (agent files, skills, MCP) ----------------------------------

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        return AgentFileSet(
            soul=await self._read_text(user_id, "/workspace/agent/SOUL.md"),
            agents=await self._read_text(user_id, "/workspace/agent/AGENTS.md"),
            user=await self._read_text(user_id, "/workspace/agent/USER.md"),
            tools=await self._read_text(user_id, "/workspace/agent/TOOLS.md"),
            memory=await self._read_text(user_id, "/workspace/agent/MEMORY.md"),
        )

    async def list_skills(self, user_id: str) -> list[Skill]:
        return await self._skills.list(user_id)

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        return await self._mcp_discovery.list_servers(user_id)

    # -- shell ---------------------------------------------------------------

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        return await self._shell.exec(user_id, input)

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        return await self._shell.spawn(user_id, input)

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        return await self._shell.read(user_id, input)

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        return await self._shell.kill(user_id, input)

    # -- files ---------------------------------------------------------------

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        return await self._files.write_content(user_id, path, content)

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        return await self._files.read(user_id, input)

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        return await self._files.read_bytes(user_id, path, max_bytes)

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        return await self._files.write(user_id, input)

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        return await self._files.edit(user_id, input)

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        return await self._files.multi_edit(user_id, input)

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        return await self._files.glob(user_id, input)

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        return await self._files.grep(user_id, input)

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        return await self._files.list(user_id, input)

    # -- agent memory + session log -----------------------------------------

    async def read_memory_file(self, user_id: str, target: MemoryTarget) -> str:
        return await self._memory_files.read(user_id, target)

    async def write_memory_file(
        self,
        user_id: str,
        target: MemoryTarget,
        content: str,
    ) -> None:
        await self._memory_files.write(user_id, target, content)

    async def append_session_log(
        self,
        user_id: str,
        conversation_id: str,
        line: str,
    ) -> None:
        await self._session_log.append(user_id, conversation_id, line)

    async def list_session_logs(self, user_id: str) -> list[str]:
        return await self._session_log.list(user_id)

    async def read_session_log(self, user_id: str, conversation_id: str) -> str:
        return await self._session_log.read(user_id, conversation_id)

    # -- internal: container plumbing ---------------------------------------

    async def _bootstrap_workspace(self, user_id: str) -> None:
        script = """
mkdir -p /workspace/agent /workspace/skills /workspace/files /workspace/content
mkdir -p /workspace/mcp /workspace/sessions
touch /workspace/agent/SOUL.md /workspace/agent/AGENTS.md /workspace/agent/USER.md /workspace/agent/TOOLS.md /workspace/agent/MEMORY.md
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
        return f"{self._container_prefix}-{safe_docker_user_part(user_id)}"

    async def _read_text(self, user_id: str, path: str) -> str:
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(workspace_path(path))}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout
