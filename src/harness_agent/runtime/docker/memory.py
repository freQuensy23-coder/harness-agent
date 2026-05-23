"""Per-user agent memory files (`MEMORY.md`, `USER.md`) inside the
Docker workspace."""

import shlex

from harness_agent.memory import MemoryTarget
from harness_agent.runtime.models import ExecInContainer


class DockerMemoryFiles:
    """Reads and writes `MEMORY.md` / `USER.md` via an in-container
    shell pipeline that holds a flock for the duration of the
    atomic-rename. Concurrent docker exec calls serialize on the lock
    file inside the container."""

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
