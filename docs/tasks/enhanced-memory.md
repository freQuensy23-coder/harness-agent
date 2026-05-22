# Enhanced Memory

**Status:** done
**Branch:** enhanced-memory
**Worktree:** /Users/a.mametyev/PycharmProjects/agent-codex-enhanced-memory
**Mode:** interactive

## Design

Bring Hermes-style persistent memory into the harness-agent. Three additions:

1. **`memory` tool + MEMORY.md / USER.md**: bounded, file-backed durable notes split into "agent's environment notes" (MEMORY.md, ~2.2KB) and "what we know about the user" (USER.md, ~1.4KB). Each file lives inside the user's Docker workspace at `/workspace/agent/`. The model writes through a single `memory` tool with action ∈ {add, replace, remove} × target ∈ {memory, user}. Replace/remove identify entries by unique substring, not id.

2. **Background review service**: every `nudge_interval` user turns (default 10), spawn a shadow LLM turn restricted to the memory tool, fed the conversation snapshot and a focused review prompt. It either writes one or more memories and exits, or says "Nothing to save." Counter is a `defaultdict[str, int]` keyed by `conversation_id`; reset on threshold *or* when the foreground turn used the memory tool itself.

3. **`session.search` tool over JSONL session logs**: every assistant/user/tool message appends one JSONL line to `/workspace/sessions/<conversation_id>.jsonl`. The `session.search` tool grep-scans these files (excluding current conversation), ranks by hit count, then calls an aux LLM to summarize each matching session focused on the query. Returns summaries, never raw transcripts.

Memory is per-user, scoped to the user's Docker container. JSONL session logs live inside the same workspace so search is local to the user's containerized environment.

### Invariants
- IV1: MEMORY.md and USER.md contents are injected into the system prompt block at every turn. Any content written through the `memory` tool will resurface in the next session's system prompt; therefore the content scanner MUST run host-side before the runtime write.
- IV2: The `memory` tool's atomic write inside the container uses `flock` against a sibling `.lock` file. Foreground and background-review may write concurrently; both must serialize through that lock.
- IV3: The background review service must NEVER emit `AssistantTextProduced` for its shadow turn — its output goes to logs only, not to the user's reply chain.
- IV4: The background review must not block the foreground turn or the conversation's run-slot. It runs in its own asyncio task.
- IV5: Char-budget enforcement is server-side (host Python). MEMORY.md ≤ 2200 chars; USER.md ≤ 1375 chars. Joined with `\n§\n`.
- IV6: `session.search` excludes the current `conversation_id` from results.

### Principles
- PC1: Mutations of memory mid-session are durable on disk but do not trigger system-prompt rebuilds — the model already knows what it wrote.
- PC2: Tool descriptions follow Hermes phrasing (declarative facts, not imperatives) to avoid self-injection of behavior directives.
- PC3: Background review prompt mirrors Hermes literally — known good wording.

### Assumptions
- AS1: One Docker container per user; memory/sessions files are persistent for the lifetime of that container.
- AS2: The OpenAI-compatible API used for foreground turns is also fine for aux summarization calls in `session.search` — we reuse the same `LlmClient`.
- AS3: `aiofiles`-style async file IO isn't needed; we go through `docker exec` for in-container writes which is already async via `asyncio.create_subprocess_exec`.

### Unknowns
- UK1: Where exactly to wire the JSONL session-log append so we don't double-write on retries/superseded generations — leverage `ConversationProjector` paths.
- UK2: Whether tests need a fake runtime hook for the memory tool's flock'd shell snippet — likely yes; `FakeUserRuntime` needs a memory store of its own.

## Plan

### Phase 1 — Memory tool + MEMORY.md plumbing
- Add `memory: str = ""` to `AgentFileSet`.
- Touch `MEMORY.md` in container bootstrap (`runtime/docker.py`).
- Read `MEMORY.md` in `DockerUserRuntime.read_agent_files`.
- New module `src/harness_agent/memory.py`:
  - `MemoryScanResult`, `scan_memory_content` (regex + invisible-unicode).
  - `MemoryDocument` / pure-Python char-budget + dedup logic used by both runtime and tests.
  - Constants `MEMORY_CHAR_LIMIT = 2200`, `USER_CHAR_LIMIT = 1375`, `ENTRY_DELIMITER = "\n§\n"`.
- Extend `UserRuntime` protocol (`runtime/protocols.py`):
  - `read_memory_file(user_id, target)` → str
  - `write_memory_file(user_id, target, new_content)` → None (flock'd, atomic inside container)
- Impl on `DockerUserRuntime` (`runtime/docker.py`): single-pipe shell that takes new content over stdin, locks, writes tmp, mvs.
- Impl on `FakeUserRuntime` (`runtime/fake.py`): in-memory dict.
- `tools.py`: `MemoryToolInput(action, target, content?, old_text?)`, register `memory` tool.
- `tool_executor.py`: handler that
  - reads current target via runtime,
  - applies action through `MemoryDocument`,
  - runs `scan_memory_content` on new content for add/replace,
  - writes back via runtime,
  - returns JSON with usage + entries.
- `context.py`: render memory + user blocks in `ContextBuilder.build` between the `user` block and tool contract.

### Phase 3 — Background memory review
- New module `src/harness_agent/memory_review.py`:
  - `MemoryReviewService(*, bus, llm, tool_executor, projection, tool_registry, interval=10)`
  - `_counters: defaultdict[str, int]`
  - `handle_assistant_text(event)` increments counter; if ≥ threshold and `turn_coordinator.is_current`, spawns asyncio task `_run_review(conversation_id, user_id)`.
  - `handle_tool_call_completed(event)` resets counter if tool was `memory`.
  - `_run_review`:
    - Reads `projection.list_llm_messages(conversation_id)`.
    - Builds an `LlmRequest` with system = review prompt, messages = history + a synthesized user message that is the review instruction, tools = `[memory_tool_spec]`.
    - Inner loop ≤ 5 iterations: dispatches memory tool calls through `ToolCallExecutor.execute_memory(...)` directly (no event publish), or exits on assistant text.
    - Never publishes `AssistantTextProduced`; logs via `logger.info`.
- `events.py`: `MemoryReviewCompleted(conversation_id, actions: list[str])` optional event for telemetry.
- `app.py`: wire MemoryReviewService, subscribe.
- `config.py`: add `memory.nudge_interval` (default 10), `memory.enabled` (default True).

### Phase 4 — JSONL session log + session.search
- New module `src/harness_agent/session_log.py`:
  - `SessionLogWriter`: per-user, per-conversation appender. Writes one JSON line per message to `/workspace/sessions/<conversation_id>.jsonl` inside container.
- Hook into `ConversationProjector`:
  - `handle_user_text`, `handle_assistant_text`, `handle_tool_call_completed` each emit a write.
- `tools.py`: `SessionSearchInput(query, limit=3)`.
- `tool_executor.py`: handler that
  - Lists `/workspace/sessions/*.jsonl` (excludes `<current>.jsonl`).
  - Greps each file for query terms; ranks by hit count.
  - Top-K files → read content → truncate around matches → aux LLM call to summarize focused on query.
  - Returns JSON list of summaries.

### Verify
- `uv run pytest` — green.
- Unit tests added for: `MemoryDocument` char-budget; `scan_memory_content`; tool handler against `FakeUserRuntime`; `MemoryReviewService` counter & threshold trigger; `session.search` against fake JSONL set.

## Conclusion

Three additions landed:

1. **`memory` tool + `MEMORY.md`** alongside the existing `USER.md`.
   Both files are injected into the system prompt every turn (raw text;
   no `═══` framing — kept consistent with how the other agent files are
   inlined). Mutations go through the tool: `MemoryDocument` enforces a
   per-target char budget (memory 2200, user 1375), dedup, and
   substring-based `replace` / `remove`. A regex+invisible-unicode scanner
   refuses prompt-injection / credential-exfil payloads before they hit
   disk because memory content lands in the system prompt of every
   future session. Atomic write inside the container goes through a
   `flock`-guarded shell pipeline. Public `execute_memory(...)` on
   `ToolCallExecutor` is what the background reviewer calls so it does
   not have to route through the event bus.

2. **JSONL session log + `session.search` tool.** `SessionLogWriter`
   subscribes to `UserTextReceived` / `AssistantTextProduced` /
   `ToolCallCompleted` and appends one JSON line per event to
   `/workspace/sessions/<conversation_id>.jsonl`. The `session.search`
   tool greps these files (excluding current conversation), ranks by hit
   count, then runs an auxiliary LLM summarisation per matching session
   focused on the query. Returns summaries only, not raw transcripts.

3. **`MemoryReviewService` — background curation.** `defaultdict[str, int]`
   counter keyed by `conversation_id`. Each `AssistantTextProduced` of
   the current generation increments; threshold (default 10) fires an
   asyncio task that runs a shadow LLM turn with a focused review prompt
   and a tool surface narrowed to `memory` alone. The shadow turn does
   not publish `AssistantTextProduced`; on completion it emits
   `MemoryReviewCompleted(actions, note)` for telemetry. The counter
   also resets when the foreground turn itself called the `memory` tool
   — no point nudging right after the model already saved something.

### What we did NOT do (deliberately)

- **Prefix-cache caching of the system prompt** (was "Phase 2" in the
  design draft). Dropped on user instruction — the `ContextBuilder`
  still rebuilds the system prompt every turn. Worth a follow-up but
  orthogonal to memory.
- **External memory providers / plugin registry.** The Docker-per-user
  isolation makes an in-process plugin slot a different design problem.

### Invariants checked
- IV1 ✓ Scanner runs host-side in `ToolCallExecutor._memory` before
  any runtime write.
- IV2 ✓ `flock` inside container script. Also covered by `asyncio.Lock`
  per `(user_id, target)` in the executor — defence in depth.
- IV3 ✓ `MemoryReviewService._run_review` only emits
  `MemoryReviewCompleted`, never `AssistantTextProduced`.
- IV4 ✓ Review runs in `asyncio.create_task`; does not take the
  conversation's `run_slot`.
- IV5 ✓ `MemoryDocument` enforces both budgets in `add` / `replace`.
- IV6 ✓ `execute_session_search` filters
  `current_conversation_id` out of `list_session_logs` results.

### Review pass — what changed after the codex pre-commit reviewer
- **Memory mutation now goes through the event bus** for the background
  reviewer too. `MemoryReviewService` publishes `ToolCallRequested` with
  `generation=0` (review-scope marker), then awaits `ToolCallCompleted`
  via `ToolCallResultWaiter`. The conversation projector and JSONL writer
  both gate on `turn_coordinator.is_current(...)`, so review-scope
  mutations are auditable on the event log but never leak into the
  conversation transcript or the session log.
- **`ToolCallExecutor` is now a thin dispatcher.** Memory mutation moved
  to `memory_service.py`; session search moved to
  `session_search_service.py`. Both services are constructed in
  `HarnessApp._wire()` and passed in.
- **`SessionSearchService` requires an `LlmClient`** rather than falling
  back to a raw-transcript preview when none is configured.
- **Conversation-id sanitisation is shared and injective.**
  `runtime/paths.py` exports `safe_conversation_id_part`, which
  percent-encodes via RFC 3986's unreserved set. Distinct conversation
  IDs map to distinct on-disk files: `tg:456`, `tg/456`, and `tg-456`
  are three different sessions. `list_session_logs` decodes the
  filenames back into raw IDs so callers can compare like-with-like.
- **Tool log transcript includes stdout / stderr.** A search hit on tool
  output now reaches the summariser as the actual stdout text, not a
  blank `[TOOL:name]` line.

### Test coverage (`uv run pytest`, all green — 61 → 84 tests)
- `tests/test_memory.py` — 13 cases: `MemoryDocument` semantics,
  scanner, end-to-end tool through `FakeUserRuntime` via `MemoryService`.
- `tests/test_session_search.py` — 5 cases: log writer end-to-end via
  bus (including stdout capture), search with summarised matches, search
  with empty store, colon-prefixed conversation id exclusion, summariser
  sees tool stdout.
- `tests/test_memory_review.py` — 5 cases: threshold trigger persists,
  "Nothing to save." note, counter resets on tool call, per-conversation
  isolation, review-scope mutation visible on the event bus.
