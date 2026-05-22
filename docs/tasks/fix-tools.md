# Remove ToolCallResultWaiter, use direct tool dispatch

**Status:** done
**Branch:** fix-tools
**Worktree:** none (in-place)
**Mode:** interactive

## Design

`ToolCallResultWaiter` (tool_executor.py) is dead coordination. `_run_turn` (handlers.py) registers a future via `expect()`, publishes `ToolCallRequested`, then awaits `wait()`. But `bus.publish` is synchronous depth-first — by the time it returns, `tool_call_executor.handle_tool_call_requested` has run inline AND the recursive `publish(ToolCallCompleted)` has already invoked `tool_results.handle_tool_call_completed` which set the future. The `wait()` always returns immediately.

Replace with direct call: `result = await tool_executor.execute(requested)`. Keep `ToolCallRequested` / `ToolCallCompleted` events flowing through the bus for audit (event store) and projection (conversation history), but only as a side effect of the direct dispatch — not as the dispatch mechanism itself.

### Invariants
- IV1: `bus.publish(ToolCallCompleted)` still happens for every tool call (projection + event store consume it).
- IV2: `bus.publish(ToolCallRequested)` still happens for every tool call (event store audit; no one else subscribes after this refactor).
- IV3: Tool execution errors still produce a `ToolCallError` event before `ToolCallCompleted` (preserves error-audit pattern from `handle_tool_call_requested`).
- IV4: Tool execution order in `_run_turn` is unchanged: model → tool → result → next model call.

### Principles
- PC1: Bus stays as audit channel; not control channel for request/response with single consumer.
- PC2: Keep `ToolCallExecutor.handle_tool_call_requested` callable directly (some tests use it) — or refactor tests, depending on cost.

### Assumptions
- AS1: Only `tool_call_executor.handle_tool_call_requested` subscribes to `ToolCallRequested` (verified via grep — no other subscribers).
- AS2: Only `tool_results.handle_tool_call_completed` and `conversation_projector.handle_tool_call_completed` subscribe to `ToolCallCompleted` (verified).
- AS3: Tests reference `ToolCallResultWaiter` directly and need wiring updates (verified: 7 test files).

### Unknowns
- UK1: Should `ToolCallRequested` publish before or after the direct `execute()` call? Decision: BEFORE (natural temporal ordering for audit log: requested → completed).

## Plan

Files to change:
1. `src/harness_agent/tool_executor.py` — extract `execute(event) -> ToolExecutionResult` as public pure-work API. Delete `ToolCallResultWaiter`. Keep `handle_tool_call_requested` as a thin wrapper for back-compat (it now just wraps `execute()` and converts to event batch — still useful for any future bus-driven path).
2. `src/harness_agent/handlers.py` — `AgentTurnHandler.__init__` takes `tool_executor: ToolCallExecutor` (required). Drop `tool_results` param + import. In `_run_turn`: publish `ToolCallRequested`, call `await self._tool_executor.execute(requested)`, on error publish `ToolCallError`, always publish `ToolCallCompleted`.
3. `src/harness_agent/app.py` — drop `self.tool_results`, drop `ToolCallResultWaiter` import. Drop `bus.subscribe(ToolCallRequested, ...)` and `bus.subscribe(ToolCallCompleted, tool_results...)`. Pass `tool_executor=tool_call_executor` to `AgentTurnHandler`.
4. Tests (test_agent_flow, test_all_tool_handlers, test_media_and_scheduler, test_tool_surface, test_subagents) — drop `ToolCallResultWaiter` instantiation; pass `tool_executor=tool_executor` to `AgentTurnHandler`; drop the two now-irrelevant `bus.subscribe` lines for `ToolCallRequested` and `ToolCallCompleted→tool_results`. Keep `conversation_projector` subscription to `ToolCallCompleted`.
5. `tests/test_tool_surface.py:212-229` — that test directly exercises the future-wait pattern. Rewrite to call `executor.execute(requested)` directly.

Test strategy: existing test suite is comprehensive enough; if all pre-existing tests pass with the refactor, behavior is preserved.

## Verify

- pytest: `92 passed in 2.40s` (`tests/` non-integration suite, run via `.venv/bin/python -m pytest tests/`).
- pyright: `372 errors, 2 warnings` — **same count as `origin/main` baseline** (verified by stashing the refactor and re-running). All pre-existing errors are in `tasks.py`, `subagents.py`, `web_fetch.py`, `db.py` (aiosqlite + bs4 stub gaps), none introduced by this refactor.
- IV1 ✓ — `bus.publish(ToolCallCompleted)` fires in `_run_turn` after every `execute()`. `conversation_projector.handle_tool_call_completed` still subscribes (app.py:212).
- IV2 ✓ — `bus.publish(ToolCallRequested)` fires before `execute()`.
- IV3 ✓ — `_run_turn` publishes `ToolCallError` if `execution.error is not None`, before publishing `ToolCallCompleted`. Confirmed in `test_tool_exception_completes_with_error_event` (event sequence `requested → error → completed`).
- IV4 ✓ — execution order preserved; only mechanism changed.

## Conclusion

- Dead coordination removed: `ToolCallResultWaiter` (the `expect`/`wait` future-rendezvous) gone, `ToolCallKey` and `_tool_call_key` helper gone, `import asyncio` no longer needed in `tool_executor.py`.
- `ToolCallExecutor.execute()` is now the public synchronous API: returns `ToolExecutionResult(result, attachments, error)`. `handle_tool_call_requested` kept as a thin wrapper that builds the event tuple — useful for direct-call tests and ad-hoc usage.
- `AgentTurnHandler` takes `tool_executor` as an optional dependency (default `None`); raises a clear `RuntimeError` if the model issues a tool call without an executor configured. Production wiring (`app.py`) passes the executor.
- Bus subscriptions for tool dispatch dropped from `app.py`: no more `bus.subscribe(ToolCallRequested, executor.handle_tool_call_requested)` or `bus.subscribe(ToolCallCompleted, tool_results.handle_tool_call_completed)`. Audit-log subscription (`conversation_projector.handle_tool_call_completed`) preserved.
- Net diff: ~50 lines removed from `tool_executor.py`, 1 dead class deleted, 7 test files simplified (no more `tool_results` plumbing).

### Assumption status
- AS1 ✓ — only `executor.handle_tool_call_requested` subscribed to `ToolCallRequested` before refactor. Verified by grep.
- AS2 ✓ — only `tool_results.handle_tool_call_completed` and `conversation_projector.handle_tool_call_completed` subscribed. Verified by grep.
- AS3 ✓ — 7 test files updated.

### Unknown status
- UK1 resolved: `ToolCallRequested` published **before** `execute()`, `ToolCallCompleted` **after**. Natural temporal ordering preserved in the event store.
