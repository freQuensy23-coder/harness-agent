# System and Development Rules

This document is the development reference for the harness agent. Keep it current
when system boundaries, event contracts, persistence, or tool behavior change.

## System Overview

The harness is an async Python agent runtime that accepts input from adapters,
normalizes it into typed events, persists those events, projects conversation
history, calls an OpenAI-compatible chat model, and executes requested tools in a
per-user Docker workspace.

The top-level composition happens in `HarnessApp`:

- Configuration is loaded from YAML into typed Pydantic settings.
- SQLite stores are created for events, conversation projection, LLM audit rows,
  schedules, tasks, and sub-agents.
- `EventBus` wires every handler together through typed event subscriptions.
- `DockerUserRuntime` owns per-user containers and file/process operations.
- `ToolCallExecutor` routes model tool calls to runtime, scheduling, task,
  web-fetch, MCP, skill, and sub-agent services.
- Reply handlers deliver only current-generation responses back to the adapter
  that originated the request.

## Runtime Boundary

The host process owns orchestration and persistence. It stays outside user
containers and is responsible for:

- Event dispatch and event persistence.
- Conversation, task, schedule, sub-agent, and LLM audit stores.
- Telegram and CLI adapters.
- Identity resolution from adapter-specific users to internal users and
  conversations.
- LLM requests and tool result coordination.
- Docker container lifecycle and MCP stdio process management.

Each user gets a persistent Docker workspace. User-controlled files and commands
run there, including:

- `/workspace/agent/*.md`
- `/workspace/skills/*/SKILL.md`
- `/workspace/content/**`
- User-created files and scripts.
- Short-lived and spawned shell commands.
- Long-lived MCP stdio server processes.

MCP server configuration exists at two scopes. User-scope configs live in the
user's workspace at `/workspace/mcp/*.yaml` and the user owns them. Global-scope
configs live in `harness.yaml` under `mcp.servers` and apply to every user; the
user cannot edit or disable them from the workspace. The host merges both lists
with global entries winning on name conflict, then starts each server process
inside the user's container via the same stdio transport.

User MCP YAML parsing is per-file: a single malformed file is skipped with a
warning so it cannot suppress other user servers or globals.

## Event Flow

The system is event based. New behavior should be modeled as explicit event
types and handlers instead of hidden side effects between services.

1. Adapter input becomes an adapter event such as `CliTextReceived` or
   `TelegramTextReceived`.
2. `IdentityHandler` converts adapter events into canonical `UserTextReceived`
   events with user and conversation identifiers.
3. `ContentIngestionHandler` stores inbound attachments in the user's workspace.
4. `ConversationProjector` appends user messages, assistant messages, tool
   calls, and tool results to the conversation projection.
5. `AgentTurnHandler` requests a new conversation generation and emits an
   `AgentTurnRequested` event.
6. The agent turn builds context, loads projected messages, asks the LLM, and
   publishes either `AssistantTextProduced` or `ToolCallRequested`.
7. `ToolCallExecutor` executes tool requests and publishes `ToolCallCompleted`.
8. Reply handlers send `AssistantTextProduced` events back to Telegram or CLI
   only when the event generation is still current.

The scheduler uses the same event path. Schedule tools write rows to the schedule
store, the pump emits `ScheduledMessageDue`, and `SchedulerDueHandler` turns that
into a synthetic `UserTextReceived` event.

Sub-agents also use the same event path. A parent tool call creates a sub-agent
record, publishes a child `UserTextReceived`, waits for the child conversation's
assistant text, and then publishes completion, failure, or cancellation events.

## Async Execution Model

The system runtime must remain 100% async. Handlers, stores, adapters, LLM
clients, runtime operations, scheduling, MCP sessions, tool execution, and
sub-agent coordination should expose async APIs and should be awaited by their
callers.

Development rules:

- Do not add blocking network, database, subprocess, sleep, or file I/O to event
  handlers, turn execution, tool execution, adapters, schedulers, or stores.
- Use async libraries for runtime I/O. The current code uses `aiosqlite`,
  `AsyncOpenAI`, async subprocess APIs, aiogram, and async HTTP clients for this
  reason.
- If a synchronous operation is unavoidable at startup or configuration time,
  keep it outside request and event handling paths and document the boundary.
- Long-running work should use tasks, spawned process handles, schedules, or
  sub-agent records instead of blocking the event loop.
- Event handlers should return follow-up events and let `EventBus` publish them.
  Avoid directly invoking another domain handler to bypass the event model.
- Before emitting user-visible replies or projecting assistant/tool results,
  verify the conversation generation is still current when applicable.

## Persistence Model

The event store is the source of received and produced event records. Projection
stores are derived views optimized for runtime reads:

- `SQLiteEventStore` persists every event published through `EventBus`.
- `SQLiteConversationProjection` stores ordered model messages and tool
  exchanges for each conversation.
- `SQLiteTaskStore` stores model-visible checklist tasks.
- `SQLiteScheduleStore` stores once and cron schedules and atomically claims due
  rows.
- `SQLiteSubAgentStore` stores parent/child conversation relationships and
  sub-agent state transitions.
- `SQLiteLlmAuditStore` records model requests and responses for inspection.

When adding state, prefer an explicit store with typed models and focused tests.
Do not hide durable state only in memory unless the state is intentionally
process-local coordination such as waiters or locks.

## Tool Surface

The model sees canonical tool names from `default_tool_registry`. Tool calls are
validated with typed Pydantic input models before execution and are persisted in
conversation history with their original structured input and rendered result.

Development rules:

- Add new model tools through `tools.py` and route execution through
  `ToolCallExecutor`.
- Keep tool names stable once persisted because event and conversation records
  reference them.
- Runtime file and shell tools must stay scoped to the user's Docker workspace.
- MCP tools are exposed as `mcp.<server>.<tool>` and should preserve the same
  validation and result rendering expectations as built-in tools.
- Tools that produce files or images should return attachments through
  `ToolExecutionResult` so the projection and model context remain explicit.

## Testing Rules

Every system element should have focused unit tests. New work is incomplete until
the affected event type, handler, store, adapter, runtime boundary, tool route,
scheduler path, or sub-agent path has tests that exercise the behavior directly.

Development rules:

- Add or update async tests for async code paths with `pytest.mark.asyncio`.
- Test event contracts by publishing the input event and asserting the emitted
  event batch, persisted projection, or reply behavior.
- Test stores with temporary SQLite paths and assert schema creation, inserts,
  reads, ordering, and state transitions.
- Test tool routes at the executor boundary with fake runtimes or stores before
  relying on end-to-end tests.
- Add regression tests for superseded generation behavior whenever replies,
  projections, or side effects are changed.
- Keep Docker or real LLM dependent tests separate from fast unit tests, and use
  fakes for unit-level coverage.
- Do not merge feature code that changes runtime behavior without a test proving
  the changed element.

## Change Checklist

Use this checklist for future improvements:

- The change preserves the 100% async runtime rule.
- The change uses explicit typed events for cross-service behavior.
- The change updates stores or projections deliberately and documents durable
  state ownership.
- The change keeps host/container boundaries intact.
- The change updates model tool schemas and executor routing together when tool
  behavior changes.
- The change adds unit tests for each changed system element.
- The change updates this document and the README when architecture or
  development rules change.
