"""File operations inside the user's Docker workspace.

All paths are normalized through `workspace_path` (or `content_path` for
attachments) so a tool call cannot escape `/workspace`. Reads and writes
are performed with a single `docker exec` per call via the shared
`exec_in_container` callable.
"""

import base64
import posixpath
import shlex

from harness_agent.content import WorkspaceFile
from harness_agent.runtime.models import (
    ExecInContainer,
    RuntimeFileRead,
    RuntimeToolResult,
)
from harness_agent.runtime.paths import content_path, workspace_path
from harness_agent.tools import (
    FileEditInput,
    FileGlobInput,
    FileGrepInput,
    FileListInput,
    FileMultiEditInput,
    FileReadInput,
    FileWriteInput,
)


class DockerFiles:
    """Per-workspace file operations exposed to the model and the
    content-ingestion path. `read_bytes` returns raw bytes via a
    Python-in-container helper so binary attachments survive transport."""

    def __init__(self, *, exec_in_container: ExecInContainer) -> None:
        self._exec = exec_in_container

    async def write_content(
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

    async def read(self, user_id: str, input: FileReadInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"head -c {int(input.max_bytes)} -- {shlex.quote(path)}"],
        )

    async def read_bytes(
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

    async def write(self, user_id: str, input: FileWriteInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        parent = posixpath.dirname(path)
        return await self._exec(
            user_id,
            ["sh", "-lc", f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"],
            stdin=input.content.encode("utf-8"),
        )

    async def edit(self, user_id: str, input: FileEditInput) -> RuntimeToolResult:
        current = await self.read(
            user_id,
            FileReadInput(path=input.path, max_bytes=5_000_000),
        )
        if current.exit_code != 0:
            return current
        if input.old not in current.stdout:
            raise ValueError("old text not found")
        count = -1 if input.replace_all else 1
        updated = current.stdout.replace(input.old, input.new, count)
        return await self.write(user_id, FileWriteInput(path=input.path, content=updated))

    async def multi_edit(
        self,
        user_id: str,
        input: FileMultiEditInput,
    ) -> RuntimeToolResult:
        current = await self.read(
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
        return await self.write(user_id, FileWriteInput(path=input.path, content=updated))

    async def glob(self, user_id: str, input: FileGlobInput) -> RuntimeToolResult:
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

    async def grep(self, user_id: str, input: FileGrepInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        # `grep` returns 1 when there are no matches; that's a normal empty
        # result, not a tool failure, so we swallow it but propagate every
        # other non-zero exit code.
        command = (
            f"grep -RIn -- {shlex.quote(input.pattern)} {shlex.quote(path)}; "
            "code=$?; "
            'if [ "$code" -eq 1 ]; then exit 0; fi; '
            'exit "$code"'
        )
        return await self._exec(user_id, ["sh", "-lc", command])

    async def list(self, user_id: str, input: FileListInput) -> RuntimeToolResult:
        path = workspace_path(input.path)
        command = (
            f"find {shlex.quote(path)} -maxdepth 1 -mindepth 1 "
            "| sort"
        )
        return await self._exec(user_id, ["sh", "-lc", command])
