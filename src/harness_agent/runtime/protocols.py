from typing import Protocol

from harness_agent.context import UserContextRuntime
from harness_agent.runtime.models import (
    DockerProcessResult,
    RuntimeFileRead,
    RuntimeToolResult,
    SpawnedProcessRecord,
)
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


class DockerRunner(Protocol):
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        pass


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
