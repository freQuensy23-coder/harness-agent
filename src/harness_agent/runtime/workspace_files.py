"""Workspace file operations (file.read/write/edit/glob/grep/list,
shell.exec, write_content_file) executed via DockerContainerExecutor."""

import base64
import posixpath
import shlex

from harness_agent.content import WorkspaceFile
from harness_agent.runtime.container import DockerContainerExecutor
from harness_agent.runtime.models import RuntimeFileRead, RuntimeToolResult
from harness_agent.runtime.paths import content_path, workspace_path
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
    ShellExecInput,
)


class DockerWorkspaceFiles:
    def __init__(self, executor: DockerContainerExecutor) -> None:
        self._executor = executor

    async def shell_exec(self, user_id: str, input: ShellExecInput) -> RuntimeToolResult:
        cwd = workspace_path(input.cwd)
        return await self._executor.exec(
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
        normalized = content_path(path)
        parent = posixpath.dirname(normalized)
        return await self._executor.exec(
            user_id,
            [
                "sh",
                "-lc",
                f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(normalized)}",
            ],
            stdin=content,
        )

    async def file_read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        return await self._executor.exec(
            user_id,
            ["sh", "-lc", f"head -c {int(input.max_bytes)} -- {shlex.quote(path)}"],
        )

    async def read_file_bytes(
        self,
        user_id: str,
        path: str,
        max_bytes: int | None,
    ) -> RuntimeFileRead:
        normalized = workspace_path(path)
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
        result = await self._executor.exec(
            user_id,
            ["python", "-c", code, normalized, limit],
        )
        if result.exit_code != 0:
            return RuntimeFileRead(result=result)
        return RuntimeFileRead(
            file=WorkspaceFile(
                path=normalized,
                content=base64.b64decode(result.stdout),
            )
        )

    async def file_write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        parent = posixpath.dirname(path)
        return await self._executor.exec(
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
        return await self._executor.exec(
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
        return await self._executor.exec(user_id, ["sh", "-lc", command])

    async def file_list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        command = (
            f"find {shlex.quote(path)} -maxdepth 1 -mindepth 1 "
            "| sort"
        )
        return await self._executor.exec(user_id, ["sh", "-lc", command])

    async def read_text(self, user_id: str, path: str) -> str:
        result = await self._executor.exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(workspace_path(path))}"],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout
