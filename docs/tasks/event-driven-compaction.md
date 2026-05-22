# Event-Driven Context Compaction

**Status:** executing
**Branch:** event-driven-compaction
**Worktree:** /Users/a.mametyev/PycharmProjects/agent-codex-event-driven-compaction
**Mode:** interactive

## Design

Replace the monolithic `ContextCompactor` class with a full event cascade. Each step of compaction is its own handler subscribed to one event type, emitting the next event in the chain — same pattern as `SubAgentService` on main. The turn runner observes only one entry point (publish `CompactionRequested`) and one exit (re-read projection after `await bus.publish` returns); it has no knowledge of the intermediate steps.

**Event cascade:**

```
runner → CompactionRequested
            ↓ CompactionService.handle_requested
       CompactionSnapshotReady | CompactionSkipped (terminal)
            ↓ CompactionService.handle_snapshot_ready
       CompactionSummaryReady
            ↓ CompactionService.handle_summary_ready
       CompactionCommitted (terminal) | CompactionConflicted (terminal)
            ↓ CompactionArchiveHandler.handle_committed
       (no follow-up; archive write is terminal)
```

**Service shape:**

- `CompactionService` — three stateless handlers, one per step. No in-memory state; events carry forward only the data the next step needs.
  - `handle_requested(CompactionRequested)` — reads projection, picks boundary (preserved 3-step algorithm), generates `archive_path`, emits `CompactionSnapshotReady(compacted_sequences, tail_sequences, snapshot_max_sequence, archive_path)` or `CompactionSkipped(reason="no_boundary")`.
  - `handle_snapshot_ready(CompactionSnapshotReady)` — re-fetches records by sequence, calls the LLM with the Claude Code `/compact` prompt, parses `<summary>` with preserved retry + analysis-strip fallback, emits `CompactionSummaryReady(summary, ...passes the snapshot fields)`.
  - `handle_summary_ready(CompactionSummaryReady)` — calls `append_compacted_context_if_unchanged` (CAS on `max(sequence)`), emits `CompactionCommitted(archive_path, compacted_sequences)` or `CompactionConflicted(reason="cas_lost")`.
- `CompactionArchiveHandler` — single handler.
  - `handle_committed(CompactionCommitted)` — re-fetches compacted records by sequence, writes JSONL to `/workspace/.old-sessions/{archive_path}` best-effort (retries transient, logs+swallows persistent). Terminal, no follow-up event.
- Drop `ContextCompactor` class and `ContextCompacting` event.

**Events (6 new):** `CompactionRequested`, `CompactionSnapshotReady`, `CompactionSummaryReady`, `CompactionCommitted`, `CompactionConflicted`, `CompactionSkipped`. Each carries `compaction_id` (uuid hex) for traceability, plus the data the next handler in the chain consumes (so handlers stay stateless).

**Behavior preserved 1:1 from the current PR (already approved):**
- Boundary algorithm: `_tail_boundary` → `_nth_user_message_from_end` → fallback `_latest_safe_boundary`.
- `<summary>` extraction: strip `<analysis>` first, then greedy `<summary>(.*)</summary>` match; retry once on missing; fallback to analysis-stripped raw text, else raw last response.
- `append_compacted_context_if_unchanged` (CAS on `max(sequence)`).
- Claude Code `/compact` system prompt verbatim.

**Behavior changes vs current PR:**
- Compaction is no longer invoked inline from `AgentTurnRunner._prepare_messages`. The runner publishes `CompactionRequested` and re-reads `list_llm_messages` after the bus chain returns.
- The `ContextCompacting` audit event is replaced by the start-end pair `CompactionRequested` / `CompactionCommitted` (or `Conflicted`/`Skipped` for terminal failures).
- Archive write is in its own handler, isolated from the commit path; archive write failure cannot break or revert the projection commit.

**Backwards-compat risks:**
- `ContextCompactor` class removed → `app.py` wiring and `tests/test_agent_flow.py` instantiation sites are rewritten. No external API exposure (this is application code in an unmerged PR).
- `ContextCompacting` event removed → existing dev SQLite event stores cannot deserialize their old rows through the `AgentEvent` discriminator. Resolution: wipe `data/` on dev machines (the only place this exists; PR #12 is not in main).

TDD: yes (refactor with deterministic, reusable handlers and a regression-sensitive contract — bus event flow).

### Invariants

- IV1 — Compaction state transitions go through the event bus only; nothing outside `CompactionService` calls its internal methods directly.
- IV2 — Every `Compaction*` event carries a `compaction_id` (uuid hex) so the start-to-terminal chain for one attempt is correlatable in the event store.
- IV3 — The projection's `context_summary` row is the source of truth for "compaction committed"; the on-disk JSONL archive is write-only and never consulted by the system.
- IV4 — `assistant_tool_call` / `tool_result` pairs are never split across the compacted/tail boundary.
- IV5 — The turn runner's only compaction dependency is publishing `CompactionRequested` and re-reading the projection.
- IV6 — Archive write failure (after retries) does not propagate to the turn or revert the committed summary.

### Principles

- PC1 — One handler per event type; each handler is testable by publishing its input event and asserting the returned `EventBatch`.
- PC2 — Boundary selection, response parsing, and CAS are pure functions or focused store methods, testable without the bus.
- PC3 — The redesign is structural, not behavioral: preserve every algorithmic decision already approved on PR #12.

### Assumptions

- AS1 — SQLite `begin immediate` in `append_compacted_context_if_unchanged` is sufficient to serialize concurrent compaction commits per conversation.
- AS2 — User containers have free disk under `/workspace/.old-sessions/` for JSONL files; no rotation policy is in scope for this task.
- AS3 — The bus's recursive sequential `publish` is the desired execution model for the chain (no parallel fan-out needed).

### Unknowns

- UK1 — Whether the supersede behavior (newer user message during in-flight summarize) still produces the same observable outcome (`AgentTurnSuperseded` + no stale summary committed) once `_stop_if_superseded` is no longer invoked inside the old `_prepare_messages` compaction block.

## Plan

Approach: TDD'd bottom-up vertical slice for a full event cascade. Each phase adds one coherent layer with tests written before implementation: projection schema → token estimator + config → 6 events + pure helpers → 3 service handlers chained → archive handler → runner integration → app wiring + e2e. Algorithms (boundary, parser, CAS) are ported verbatim from PR #12 reference files; the redesign is structural — each step is its own bus handler.

Reference files (PR #12 at `/Users/a.mametyev/PycharmProjects/agent-codex-context-compaction`):
- `src/harness_agent/compaction.py` lines 22-106 (`SUMMARY_SYSTEM_PROMPT`), 281-326 (pure helpers), 329-352 (`_records_to_jsonl`).
- `src/harness_agent/projections.py` lines 22-37 (`CONTEXT_SUMMARY_KIND`, `ConversationItemRecord`), 227-325 (active/all items, CAS), 327-367 (`_list_all_context_items`), 465-476 (`_latest_summary_index`, `_summary_payload`).

### PH1 — Projection: context_summary support

- **1.1** `src/harness_agent/projections.py:1-50` (modify)
  - Import `dataclass` from `dataclasses`.
  - Add `CONTEXT_SUMMARY_KIND = "context_summary"`.
  - Add `ConversationItemRecord` dataclass (sequence, user_id, conversation_id, generation, item_kind, text, tool_call_id, tool_name, payload_json, message_json, message).
  - Respects: IV3.
- **1.2** `src/harness_agent/projections.py` (modify, append after existing methods)
  - `list_active_context_items(conversation_id: str) -> list[ConversationItemRecord]` — reads all items, if latest summary exists returns `[summary, *tail_by_sequence, *newer_non_summary]` else returns all.
  - `list_all_context_items(conversation_id: str) -> list[ConversationItemRecord]` — unfiltered rows, used by snapshot/summary/archive handlers to re-fetch by sequence.
  - `list_llm_messages(conversation_id: str)` — re-implement as `[r.message for r in await self.list_active_context_items(...)]`.
  - `append_compacted_context_if_unchanged(*, user_id, conversation_id, generation, summary, archive_path, compacted_sequences: list[int], tail_sequences: list[int], snapshot_max_sequence: int) -> bool` — CAS via `begin immediate`: returns False if `max(sequence)` changed, else inserts a `context_summary` row with payload `{archive_path, compacted_sequences, tail_sequences}`. Takes sequence lists directly (handlers pass them from events; no record re-read needed for CAS).
  - `_latest_summary_index(records)` and `_summary_payload(record)` — module-level helpers.
  - Respects: IV3, IV4, AS1.
- **1.3** `tests/test_compaction.py` (create, ~120 lines)
  - `test_list_active_returns_all_when_no_summary`
  - `test_list_active_returns_summary_plus_tail_after_summary`
  - `test_list_active_drops_compacted_records_outside_payload_tail`
  - `test_append_compacted_writes_summary_row_and_returns_true`
  - `test_append_compacted_returns_false_when_max_sequence_changed`
- Commit: `Add context_summary projection support`

### PH2 — Token estimator + compaction config knobs

- **2.1** `pyproject.toml` (modify) — add `tiktoken>=0.13.0` to dependencies. Run `uv lock`.
- **2.2** `src/harness_agent/llm.py` (modify, after `LlmClient` class around line 75)
  - Import `tiktoken`.
  - `estimate_request_tokens(request: LlmRequest) -> int` — tokenize `request.system` + each `message.model_dump_json()` + each `tool_to_openai(tool)` JSON via gpt-5 encoding, return sum.
  - Export `LlmResponse` type alias publicly.
- **2.3** `src/harness_agent/config.py` (modify, `LlmConfig` block)
  - Add three fields with defaults: `max_tokens_per_model: int = 128_000`, `compaction_reserve_tokens: int = 15_000`, `compaction_keep_last_user_messages: int = 2`.
- **2.4** `harness.yaml.example` (modify) — add the three new keys under `llm:` with example values.
- **2.5** `tests/test_compaction.py` (extend)
  - `test_estimate_request_tokens_counts_system_messages_and_tools` — monkeypatch tokenizer to a recording stub.
  - `test_config_loads_compaction_knobs_with_defaults`
- Commit: `Add token estimator and compaction config knobs`

### PH3 — Compaction events + pure helpers

- **3.1** `src/harness_agent/events.py` (modify, after `AgentGenerationStarted` block)
  - `CompactionRequested(EventBase)`: `compaction_id: str`, `user_id`, `conversation_id`, `generation: int`.
  - `CompactionSnapshotReady(EventBase)`: `compaction_id`, `user_id`, `conversation_id`, `generation`, `compacted_sequences: list[int]`, `tail_sequences: list[int]`, `snapshot_max_sequence: int`, `archive_path: str`.
  - `CompactionSummaryReady(EventBase)`: `compaction_id`, `user_id`, `conversation_id`, `generation`, `compacted_sequences: list[int]`, `tail_sequences: list[int]`, `snapshot_max_sequence: int`, `archive_path: str`, `summary: str`.
  - `CompactionCommitted(EventBase)`: `compaction_id`, `user_id`, `conversation_id`, `generation`, `archive_path: str`, `compacted_sequences: list[int]`.
  - `CompactionConflicted(EventBase)`: `compaction_id`, `user_id`, `conversation_id`, `generation`, `reason: Literal["cas_lost"]`.
  - `CompactionSkipped(EventBase)`: `compaction_id`, `user_id`, `conversation_id`, `generation`, `reason: Literal["no_boundary"]`.
  - Add all six to `AgentEvent` discriminator union.
  - Respects: IV2.
- **3.2** `src/harness_agent/compaction.py` (create, module top — helpers and config only)
  - `SUMMARY_SYSTEM_PROMPT` — verbatim block from PR #12 lines 22-106.
  - `CompactionConfig` frozen dataclass: `max_tokens_per_model: int`, `reserve_tokens: int = 15_000`, `keep_last_user_messages: int = 2`. Property `threshold = max - reserve`.
  - `_tail_boundary(records, keep_last_user_messages) -> int | None` — verbatim from PR #12 lines 293-302.
  - `_nth_user_message_from_end(records, n) -> int | None` — verbatim from PR #12 lines 305-318.
  - `_latest_safe_boundary(records) -> int | None` — verbatim from PR #12 lines 321-326.
  - `_summary_block(text) -> str | None` — verbatim from PR #12 lines 281-286.
  - `_strip_analysis(text) -> str` — verbatim from PR #12 lines 289-290.
  - `_assistant_text(response: LlmResponse) -> str` — verbatim from PR #12 lines 275-278.
  - `_records_to_jsonl(records) -> str` — verbatim from PR #12 lines 329-346 (inline the conditional, drop unused `_json_or_none`).
  - Respects: IV4, PC2, PC3.
- **3.3** `tests/test_compaction.py` (extend)
  - `test_tail_boundary_finds_nth_user_message_from_end`
  - `test_tail_boundary_returns_none_when_n_users_with_first_at_index_zero`
  - `test_tail_boundary_falls_back_to_latest_safe_when_fewer_users`
  - `test_tail_boundary_safe_fallback_skips_tool_result_and_tool_context`
  - `test_summary_block_extracts_trailing_summary_when_analysis_quotes_a_tag`
  - `test_summary_block_preserves_literal_summary_tag_inside_body` — greedy match
  - `test_strip_analysis_removes_analysis_block_nondestructively`
- Commit: `Add compaction events and pure helpers`

### PH4 — CompactionService cascade (3 chained handlers)

- **4.1** `src/harness_agent/compaction.py` (extend with `CompactionService` class — stateless, 3 handlers)
  - `CompactionService(*, projection, llm, config)` — keeps refs only, no in-memory state per compaction.
  - `async handle_requested(event: CompactionRequested) -> EventBatch`:
    1. `records = await projection.list_active_context_items(event.conversation_id)`.
    2. `boundary = _tail_boundary(records, config.keep_last_user_messages)`.
    3. If `boundary is None`: return `(CompactionSkipped(compaction_id=event.compaction_id, ..., reason="no_boundary"),)`.
    4. `compacted = records[:boundary]`, `tail = records[boundary:]`, `snapshot_max_seq = max(r.sequence for r in records)`, `archive_path = f"/workspace/.old-sessions/{event.compaction_id}.jsonl"`.
    5. Return `(CompactionSnapshotReady(compaction_id=event.compaction_id, ..., compacted_sequences=[r.sequence for r in compacted], tail_sequences=[r.sequence for r in tail], snapshot_max_sequence=snapshot_max_seq, archive_path=archive_path),)`.
  - `async handle_snapshot_ready(event: CompactionSnapshotReady) -> EventBatch`:
    1. `all_records = await projection.list_all_context_items(event.conversation_id)`.
    2. `by_seq = {r.sequence: r for r in all_records}`; `compacted = [by_seq[s] for s in event.compacted_sequences if s in by_seq]`.
    3. `summary = await _summarize(llm, compacted, event)` — retry+fallback (PR #12 lines 245-268).
    4. Return `(CompactionSummaryReady(compaction_id=event.compaction_id, ..., summary=summary, compacted_sequences=event.compacted_sequences, tail_sequences=event.tail_sequences, snapshot_max_sequence=event.snapshot_max_sequence, archive_path=event.archive_path),)`.
  - `async handle_summary_ready(event: CompactionSummaryReady) -> EventBatch`:
    1. `committed = await projection.append_compacted_context_if_unchanged(user_id=event.user_id, conversation_id=event.conversation_id, generation=event.generation, summary=event.summary, archive_path=event.archive_path, compacted_sequences=event.compacted_sequences, tail_sequences=event.tail_sequences, snapshot_max_sequence=event.snapshot_max_sequence)`.
    2. If not committed: return `(CompactionConflicted(compaction_id=event.compaction_id, ..., reason="cas_lost"),)`.
    3. Return `(CompactionCommitted(compaction_id=event.compaction_id, ..., archive_path=event.archive_path, compacted_sequences=event.compacted_sequences),)`.
  - Module-level `async _summarize(llm: LlmClient, records: list[ConversationItemRecord], event: CompactionSnapshotReady) -> str` — pure async, retry up to `_SUMMARY_ATTEMPTS = 2`, fallback `_strip_analysis(last_text) or last_text.strip()`.
  - Respects: IV1, IV2, IV3, IV4, AS1, PC1, PC2, PC3.
- **4.2** `tests/test_compaction.py` (extend — one test per handler boundary)
  - `test_handle_requested_emits_snapshot_ready_on_happy_path` — assert returned tuple is one `CompactionSnapshotReady` with right sequences + archive_path.
  - `test_handle_requested_emits_skipped_when_no_boundary` — single-user conversation.
  - `test_handle_snapshot_ready_calls_llm_once_when_summary_tag_present` — fake LLM returns `<summary>X</summary>`, single call, `CompactionSummaryReady(summary="X")` emitted.
  - `test_handle_snapshot_ready_retries_once_when_summary_tag_missing` — fake LLM returns no tag then tag; assert 2 calls.
  - `test_handle_snapshot_ready_falls_back_to_analysis_stripped_when_retries_exhausted`.
  - `test_handle_summary_ready_emits_committed_when_cas_succeeds`.
  - `test_handle_summary_ready_emits_conflicted_when_max_sequence_advanced` — pre-seed projection between snapshot and summary, CAS returns False.
- Commit: `Add CompactionService cascade (requested → snapshot → summary)`

### PH5 — CompactionArchiveHandler

- **5.1** `src/harness_agent/compaction.py` (extend with `CompactionArchiveHandler` class)
  - `CompactionArchiveHandler(*, projection, runtime)`.
  - `async handle_committed(event: CompactionCommitted) -> EventBatch`:
    1. `all_records = await projection.list_all_context_items(event.conversation_id)`.
    2. `by_seq = {r.sequence: r for r in all_records}`; `compacted = [by_seq[s] for s in event.compacted_sequences if s in by_seq]`.
    3. `jsonl = _records_to_jsonl(compacted)`.
    4. Retry-write via `runtime.file_write(FileWriteInput(path=event.archive_path, content=jsonl))` up to `_ARCHIVE_WRITE_ATTEMPTS = 3`; swallow exceptions per attempt.
    5. On persistent failure: `logger.warning(...)`. Always return `()`. No follow-up event.
  - Respects: IV3, IV6, AS2.
- **5.2** `tests/test_compaction.py` (extend, uses `FakeUserRuntime` with `file_write_results`)
  - `test_archive_handler_writes_jsonl_for_committed_sequences`
  - `test_archive_handler_retries_on_transient_failure`
  - `test_archive_handler_swallows_persistent_failure_and_logs`
  - `test_archive_handler_skips_records_not_in_compacted_sequences` — projection has more rows than `compacted_sequences` references; jsonl includes only the named ones.
- Commit: `Add CompactionArchiveHandler`

### PH6 — Runner integration

- **6.1** `src/harness_agent/handlers.py` (modify, `AgentTurnHandler` block at lines 153-323)
  - Add `compaction_config: CompactionConfig | None = None` parameter to `__init__`.
  - In `_run_turn`, before each LLM call inside `while True`, if `compaction_config is not None`: build the `LlmRequest`, call `estimate_request_tokens(request)`, and if `>= compaction_config.threshold`: `compaction_id = uuid4().hex`; `await self._bus.publish(CompactionRequested(compaction_id=..., user_id=..., conversation_id=..., generation=...))`; then `messages = await self._projection.list_llm_messages(event.conversation_id)` to refresh.
  - Keep all existing supersede / tool / sub-agent logic intact.
  - Respects: IV5, IV1, AS3.
- **6.2** `tests/test_agent_flow.py` (extend) — uses `FakeLlmClient` + a fake estimator that returns over-threshold, plus a real `CompactionService` (or a thin handler subscribed to `CompactionRequested` that appends a summary row).
  - `test_runner_publishes_compaction_requested_when_estimate_over_threshold`
  - `test_runner_re_reads_messages_after_compaction_chain_returns`
  - `test_runner_does_not_publish_compaction_when_under_threshold`
  - `test_supersede_during_compaction_does_not_commit_stale_summary` — covers UK1: blocking LLM in service, second `UserTextReceived` arrives mid-cascade, generation bumps via projector, CAS for stale summary fails → `CompactionConflicted` instead of `CompactionCommitted`; the next turn sees full uncompacted history.
- Commit: `Publish CompactionRequested from agent turn`

### PH7 — App wiring + e2e

- **7.1** `src/harness_agent/app.py` (modify, `_wire` block around lines 156-180)
  - Import `CompactionService`, `CompactionArchiveHandler`, `CompactionConfig` from `harness_agent.compaction`.
  - Import the 6 new events from `harness_agent.events`.
  - Build `compaction_config` from `self._config.llm.*` fields; pass to `AgentTurnHandler`.
  - Instantiate `CompactionService(projection=..., llm=..., config=compaction_config)`.
  - Instantiate `CompactionArchiveHandler(projection=..., runtime=...)`.
  - `self.bus.subscribe(CompactionRequested, compaction_service.handle_requested)`.
  - `self.bus.subscribe(CompactionSnapshotReady, compaction_service.handle_snapshot_ready)`.
  - `self.bus.subscribe(CompactionSummaryReady, compaction_service.handle_summary_ready)`.
  - `self.bus.subscribe(CompactionCommitted, archive_handler.handle_committed)`.
- **7.2** `tests/test_agent_flow.py` (extend, full-flow test through `HarnessApp` wiring)
  - `test_e2e_compaction_cascade_in_order` — publish `UserTextReceived`, fake LLM over threshold; assert events in store ordered `CompactionRequested → CompactionSnapshotReady → CompactionSummaryReady → CompactionCommitted → file_write_calls (archive) → AssistantTextProduced`.
- Commit: `Wire CompactionService and archive handler into HarnessApp`

### Test strategy

TDD: yes. Tests come before implementation per phase. `tests/test_compaction.py` holds unit tests (projection, helpers, service handlers, archive). `tests/test_agent_flow.py` holds integration tests (runner, e2e cascade).

Coverage by IV/PC/AS/UK:
- IV1, IV5: PH6 runner tests + PH7 e2e — runner never calls service methods directly.
- IV2: each PH4/PH5 test asserts `compaction_id` propagates through the chain.
- IV3: PH1 projection truth; PH5 archive-write-only (no projection write in archive handler).
- IV4: PH3 boundary tests; PH4 happy-path test verifies tool-pair atomicity in chosen tail.
- IV6: PH5 archive-failure tests confirm no exception propagates.
- PC1: PH4 tests publish each event type in isolation and assert one returned event class.
- PC2: PH3 pure-helper tests; PH1 projection tests.
- PC3: each phase cites verbatim line ranges from PR #12.
- AS1: PH1 CAS tests; PH4 conflicted test.
- AS2: PH5 happy + failure tests.
- UK1: PH6 supersede test resolves it.

### Backwards-compat

Greenfield on `main` (this branch starts off `origin/main` with no compaction code), no consumers of old code. `ContextCompacting` event removed in PR #12 stays removed (never reached main). The `harness.yaml` config keys added in PH2 default to safe values so existing user configs keep working without edits.

### Risks / rollback

- RK1 — Boundary algorithm edge cases not in PR #12 tests (conversation already containing a context_summary row from prior compaction) could regress. Mitigation: explicit test `test_list_active_drops_compacted_records_outside_payload_tail` in PH1 plus PH4 happy-path test with a pre-existing summary row.
- RK2 — Bus's sequential `publish` recursion grows by one per chained event (Requested → SnapshotReady → SummaryReady → Committed → archive returns nothing). Max depth ~5 per compaction, way under Python's 1000-frame limit. PH7 e2e smoke confirms.
- RK3 — Carrying `compacted_sequences` / `tail_sequences` through three events duplicates that data in the event store (~3× sequence-list size). Sequences are int lists, KB-scale; event store volume impact negligible. Accepted trade-off for stateless handlers (no hidden in-memory state keyed by `compaction_id`).

### Interfaces

- IF1 — `ConversationItemRecord` (dataclass in `projections.py`) — produced PH1, consumed PH3 helpers + PH4 + PH5.
- IF2 — `estimate_request_tokens(request: LlmRequest) -> int` — produced PH2, consumed PH6.
- IF3 — `CompactionConfig` dataclass (defined in `compaction.py`) — produced PH3, consumed PH4 (service init), PH6 (runner threshold check), PH7 (wiring).
- IF4 — `CompactionRequested` event class — produced PH3, consumed PH4 (subscribe), PH6 (publish), PH7 (subscribe wiring).
- IF5 — `CompactionSnapshotReady` event class — produced PH3, consumed PH4 (subscribe + emit), PH7 (subscribe wiring).
- IF6 — `CompactionSummaryReady` event class — produced PH3, consumed PH4 (subscribe + emit), PH7 (subscribe wiring).
- IF7 — `CompactionCommitted` event class — produced PH3 (declared), PH4 (emitted), consumed PH5 (subscribe), PH7 (subscribe wiring).
- IF8 — `SQLiteConversationProjection.append_compacted_context_if_unchanged(...) -> bool` — produced PH1, consumed PH4 (`handle_summary_ready`).
- IF9 — `SQLiteConversationProjection.list_active_context_items(conversation_id)` — produced PH1, consumed PH4 (`handle_requested`), runner (re-read in PH6).
- IF10 — `SQLiteConversationProjection.list_all_context_items(conversation_id)` — produced PH1, consumed PH4 (`handle_snapshot_ready`), PH5.

### Interface graph

- PH1                                  -> IF1, IF8, IF9, IF10      @ src/harness_agent/projections.py, tests/test_compaction.py
- PH2                                  -> IF2                       @ src/harness_agent/llm.py, src/harness_agent/config.py, harness.yaml.example, pyproject.toml, uv.lock
- PH3                                  -> IF3, IF4, IF5, IF6, IF7   @ src/harness_agent/events.py, src/harness_agent/compaction.py
- PH4  IF1, IF3, IF4, IF5, IF6, IF8, IF9, IF10 ->                   @ src/harness_agent/compaction.py
- PH5  IF1, IF7, IF10                  ->                            @ src/harness_agent/compaction.py
- PH6  IF2, IF3, IF4, IF9              ->                            @ src/harness_agent/handlers.py, tests/test_agent_flow.py
- PH7  IF3, IF4, IF5, IF6, IF7         ->                            @ src/harness_agent/app.py, tests/test_agent_flow.py

## Verify
<empty — filled by up:uverify>

## Conclusion
<empty — filled by up:ureview>
