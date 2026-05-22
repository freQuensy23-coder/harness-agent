"""Memory mutation service.

Owns the read-mutate-write loop for `MEMORY.md` / `USER.md`. The
ToolCallExecutor's `memory` handler delegates here; the background
MemoryReviewService does NOT call this directly — it publishes
`ToolCallRequested` events that route to the executor and then back here.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from harness_agent.memory import (
    MemoryDocument,
    MemoryMutationError,
    MemoryTarget,
    scan_memory_content,
)
from harness_agent.runtime import RuntimeToolResult
from harness_agent.runtime.protocols import UserRuntime
from harness_agent.tools import MemoryToolInput


class MemoryService:
    def __init__(self, runtime: UserRuntime) -> None:
        self._runtime = runtime
        self._locks: dict[tuple[str, MemoryTarget], asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    async def execute(
        self,
        user_id: str,
        input: MemoryToolInput,
    ) -> RuntimeToolResult:
        target: MemoryTarget = input.target
        if input.action in ("add", "replace"):
            if input.content is None or not input.content.strip():
                return RuntimeToolResult(
                    stderr=f"'content' is required for action '{input.action}'.\n",
                    exit_code=1,
                )
            rejection = scan_memory_content(input.content)
            if rejection is not None:
                return RuntimeToolResult(stderr=rejection + "\n", exit_code=1)
        if input.action in ("replace", "remove"):
            if input.old_text is None or not input.old_text.strip():
                return RuntimeToolResult(
                    stderr=f"'old_text' is required for action '{input.action}'.\n",
                    exit_code=1,
                )
        async with self._locks[(user_id, target)]:
            raw = await self._runtime.read_memory_file(user_id, target)
            doc = MemoryDocument.parse(target, raw)
            try:
                if input.action == "add":
                    assert input.content is not None
                    message = doc.add(input.content)
                elif input.action == "replace":
                    assert input.content is not None
                    assert input.old_text is not None
                    message = doc.replace(input.old_text, input.content)
                else:
                    assert input.old_text is not None
                    message = doc.remove(input.old_text)
            except MemoryMutationError as exc:
                return RuntimeToolResult(stderr=f"{exc}\n", exit_code=1)
            await self._runtime.write_memory_file(user_id, target, doc.render())
        payload = {
            "success": True,
            "target": target,
            "action": input.action,
            "message": message,
            "entries": doc.entries,
            "entry_count": len(doc.entries),
            "usage": (
                f"{doc.usage_percent()}% — {doc.char_count:,}/{doc.limit:,} chars"
            ),
        }
        return RuntimeToolResult(stdout=json.dumps(payload, ensure_ascii=False))
