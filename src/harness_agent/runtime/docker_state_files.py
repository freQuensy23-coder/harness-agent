"""Helpers for the per-user memory files (`MEMORY.md`, `USER.md`) and
session-log JSONL files inside the Docker workspace.

Extracted out of `DockerUserRuntime` so the central runtime class stays
focused on container/shell/file primitives. Each helper holds a
reference to the runtime's `_exec(user_id, argv, *, stdin=...)` callable
and reuses the same docker-exec plumbing — there is no duplicate path
to the container.
"""

import shlex
import urllib.parse
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath

from harness_agent.memory import MemoryTarget
from harness_agent.runtime.models import DockerProcessResult
from harness_agent.runtime.paths import safe_conversation_id_part


ExecInContainer = Callable[..., Awaitable[DockerProcessResult]]


class DockerMemoryFiles:
    """Reads and writes the per-user `MEMORY.md` / `USER.md` files via
    an in-container shell pipeline that holds a flock for the duration
    of the atomic-rename. Concurrent docker exec calls serialize on the
    lock file inside the container."""

    def __init__(self, *, exec_in_container: ExecInContainer) -> None:
        self._exec = exec_in_container

    @staticmethod
    def path_for(target: MemoryTarget) -> str:
        if target == "user":
            return "/workspace/agent/USER.md"
        return "/workspace/agent/MEMORY.md"

    async def read(self, user_id: str, target: MemoryTarget) -> str:
        path = self.path_for(target)
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(path)} 2>/dev/null || true"],
        )
        return result.stdout

    async def write(self, user_id: str, target: MemoryTarget, content: str) -> None:
        path = self.path_for(target)
        lock_path = f"{path}.lock"
        script = (
            "set -eu; "
            f"d={shlex.quote(path)}; "
            f"l={shlex.quote(lock_path)}; "
            "touch \"$l\"; "
            "( "
            "  flock -x 9; "
            "  tmp=$(mktemp \"$d.tmp.XXXXXX\"); "
            "  trap 'rm -f \"$tmp\"' EXIT; "
            "  cat > \"$tmp\"; "
            "  mv \"$tmp\" \"$d\"; "
            ") 9>\"$l\""
        )
        result = await self._exec(
            user_id,
            ["sh", "-lc", script],
            stdin=content.encode("utf-8"),
        )
        if result.exit_code != 0:
            raise RuntimeError(
                result.stderr or f"failed to write memory file {path}"
            )


class DockerSessionLog:
    """Append-only JSONL session log per conversation, written under an
    in-container flock so concurrent writers (foreground turn vs
    background memory review) cannot interleave within one append."""

    def __init__(self, *, exec_in_container: ExecInContainer) -> None:
        self._exec = exec_in_container

    async def append(self, user_id: str, conversation_id: str, line: str) -> None:
        safe_id = safe_conversation_id_part(conversation_id)
        if not safe_id:
            return
        path = f"/workspace/sessions/{safe_id}.jsonl"
        lock_path = f"{path}.lock"
        script = (
            "set -eu; "
            "mkdir -p /workspace/sessions; "
            f"d={shlex.quote(path)}; "
            f"l={shlex.quote(lock_path)}; "
            "touch \"$l\"; "
            "( flock -x 9; cat >> \"$d\"; ) 9>\"$l\""
        )
        result = await self._exec(
            user_id,
            ["sh", "-lc", script],
            stdin=(line.rstrip("\n") + "\n").encode("utf-8"),
        )
        if result.exit_code != 0:
            raise RuntimeError(
                result.stderr
                or f"session log append failed for {user_id}/{conversation_id}"
            )

    async def list(self, user_id: str) -> list[str]:
        result = await self._exec(
            user_id,
            [
                "sh",
                "-lc",
                "find /workspace/sessions -maxdepth 1 -name '*.jsonl' -type f | sort",
            ],
        )
        if result.exit_code != 0:
            return []
        ids: list[str] = []
        for raw in result.stdout.splitlines():
            stem = PurePosixPath(raw.strip()).name
            if stem.endswith(".jsonl"):
                encoded = stem[: -len(".jsonl")]
                # Reverse the encoding from `safe_conversation_id_part`
                # so the contract is "list returns the raw conversation IDs".
                ids.append(urllib.parse.unquote(encoded))
        return ids

    async def read(self, user_id: str, conversation_id: str) -> str:
        safe_id = safe_conversation_id_part(conversation_id)
        if not safe_id:
            return ""
        path = f"/workspace/sessions/{safe_id}.jsonl"
        result = await self._exec(
            user_id,
            ["sh", "-lc", f"cat -- {shlex.quote(path)} 2>/dev/null || true"],
        )
        return result.stdout
