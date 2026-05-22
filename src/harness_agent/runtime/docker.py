"""Facade wiring DockerContainerExecutor + workspace files + workspace
metadata + spawned-process collaborators behind the UserRuntime ABC."""

import asyncio

from harness_agent.context import AgentFileSet, Skill
from harness_agent.mcp_models import McpServerConfig
from harness_agent.runtime.container import DockerContainerExecutor
from harness_agent.runtime.models import RuntimeFileRead, RuntimeToolResult
from harness_agent.runtime.protocols import DockerRunner, SpawnedProcessStore, UserRuntime
from harness_agent.runtime.spawned_processes import DockerSpawnedProcesses
from harness_agent.runtime.workspace_files import DockerWorkspaceFiles
from harness_agent.runtime.workspace_metadata import DockerWorkspaceMetadata
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
        spawned_process_store: SpawnedProcessStore,
        runner: DockerRunner,
        image: str = "python:3.14-slim",
        container_prefix: str = "harness",
        ensure_container: bool = False,
        network: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
    ) -> None:
        self._executor = DockerContainerExecutor(
            runner=runner,
            image=image,
            container_prefix=container_prefix,
            ensure_container=ensure_container,
            network=network,
            memory=memory,
            cpus=cpus,
        )
        self._files = DockerWorkspaceFiles(self._executor)
        self._metadata = DockerWorkspaceMetadata(
            executor=self._executor,
            files=self._files,
        )
        self._spawned = DockerSpawnedProcesses(
            executor=self._executor,
            store=spawned_process_store,
        )

    async def ensure_user_container(self, user_id: str) -> None:
        await self._executor.ensure(user_id)

    async def read_agent_files(self, user_id: str) -> AgentFileSet:
        return await self._metadata.read_agent_files(user_id)

    async def list_skills(self, user_id: str) -> list[Skill]:
        return await self._metadata.list_skills(user_id)

    async def list_mcp_servers(self, user_id: str) -> list[McpServerConfig]:
        return await self._metadata.list_mcp_servers(user_id)

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        return await self._files.shell_exec(user_id, input)

    async def write_content_file(
        self,
        user_id: str,
        path: str,
        content: bytes,
    ) -> RuntimeToolResult:
        return await self._files.write_content_file(user_id, path, content)

    async def shell_spawn(self, user_id: str, input: ShellSpawnInput) -> RuntimeToolResult:
        return await self._spawned.shell_spawn(user_id, input)

    async def shell_read(self, user_id: str, input: ShellReadInput) -> RuntimeToolResult:
        return await self._spawned.shell_read(user_id, input)

    async def shell_kill(self, user_id: str, input: ShellKillInput) -> RuntimeToolResult:
        return await self._spawned.shell_kill(user_id, input)

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        return await self._files.file_read(user_id, input)

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        return await self._files.read_file_bytes(user_id, path, max_bytes)

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        return await self._files.file_write(user_id, input)

    async def file_edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        return await self._files.file_edit(user_id, input)

    async def file_multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        return await self._files.file_multi_edit(user_id, input)

    async def file_glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
        return await self._files.file_glob(user_id, input)

    async def file_grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        return await self._files.file_grep(user_id, input)

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        return await self._files.file_list(user_id, input)

    async def open_stdio(
        self,
        *,
        user_id: str,
        argv: list[str],
        cwd: str = "/workspace",
    ) -> asyncio.subprocess.Process:
        return await self._executor.open_stdio(user_id=user_id, argv=argv, cwd=cwd)
