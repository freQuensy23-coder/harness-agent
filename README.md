Harness Agent
=============

Python 3.14 async agent harness with typed events, SQLite persistence, Telegram input, OpenAI-compatible Chat Completions API, and one persistent Docker container per user.

Boundary
--------

Outside Docker:

- event bus
- SQLite event/message/task stores
- Telegram adapter
- OpenAI-compatible LLM calls
- tool router
- Docker controller
- MCP stdio client manager

Inside each user Docker container:

- `/workspace/agent/*.md` (`SOUL.md`, `AGENTS.md`, `USER.md`, `MEMORY.md`, `TOOLS.md`)
- `/workspace/skills/*/SKILL.md`
- `/workspace/sessions/<conversation_id>.jsonl` — per-conversation message log searched by `session.search`
- `/workspace/content/**` for Telegram uploads
- user files
- scripts
- short-lived shell commands
- long-running spawned shell commands
- long-lived MCP stdio server processes

MCP stdio servers are configured in YAML and run inside the user's Docker container.

Turn Concurrency
----------------

Per conversation, each inbound user message creates a new generation. A running older generation stops at the next safe point before reply delivery or tool side effects. Intermediate stale generations do not call the LLM; the latest generation runs against persisted history containing all user messages.

Conversation history stores user messages, assistant messages, assistant tool calls, and tool results in order.

Telegram uploads are represented as typed inbound events. The content ingestion handler writes every upload to `/workspace/content/...`; images are also attached to the user message sent to the LLM.

Scheduler messages are also events. `schedule.once` and `schedule.cron` create rows in SQLite; the scheduler pump emits `scheduled.message.due`, which becomes a synthetic `user.text.received` event through the bus.

Persistent Memory
-----------------

Two file-backed stores live in each user's workspace and are inlined into every system prompt:

- `/workspace/agent/USER.md` — what the agent knows about the user (role, preferences, communication style). Char budget: 1375.
- `/workspace/agent/MEMORY.md` — the agent's own durable notes (environment, conventions, tool quirks). Char budget: 2200.

The model writes through the `memory` tool: `action ∈ {add, replace, remove}` × `target ∈ {memory, user}`. Replace / remove identify an entry by a unique substring. Entries are joined by `\n§\n`. Content is scanned host-side for prompt-injection and credential-exfiltration patterns before any disk write — memory text lands in the system prompt of every future session, so payloads cannot be allowed to persist. Mutations inside the container are atomic via `flock` + tmpfile + `mv`.

A background `MemoryReviewService` watches `AssistantTextProduced` events. Per-conversation counter (`defaultdict[str, int]`) increments on each assistant message of the current generation. At the configured threshold (`memory.nudge_interval`, default 10) the service fires an asyncio task that runs a shadow LLM turn restricted to the `memory` tool, fed the conversation history plus a focused review prompt. The shadow turn never emits `AssistantTextProduced`; it surfaces only `MemoryReviewCompleted(actions, note)` for telemetry. The counter also resets when the foreground turn itself used the `memory` tool.

Session Recall
--------------

Every user message, assistant message, and tool exchange is also appended to `/workspace/sessions/<conversation_id>.jsonl` (one JSON record per line). `session.search` is a two-stage tool: stage 1 grep-scans every JSONL file (excluding the current conversation), ranks by hit count; stage 2 calls an auxiliary LLM per matching session with a focused summarisation prompt. The tool returns short summaries — never raw transcripts — so cross-session recall does not consume the active context window.

Tools Exposed To LLM
--------------------

- `shell.exec`
- `shell.spawn`
- `shell.read`
- `shell.kill`
- `file.read`
- `file.write`
- `file.edit`
- `file.multi_edit`
- `file.glob`
- `file.grep`
- `file.list`
- `web.fetch`
- `task.create`
- `task.get`
- `task.list`
- `task.update`
- `task.stop`
- `schedule.once`
- `schedule.cron`
- `schedule.list`
- `schedule.cancel`
- `skill.list`
- `skill.read`
- `memory`
- `session.search`
- `mcp.<server>.<tool>`

MCP Configuration
-----------------

MCP servers can be defined at two scopes. The same stdio transport runs both:
the server process launches inside the user's Docker container via `docker exec -i`.

User scope — config lives inside that user's Docker workspace, one YAML per server.
The user can edit, add, or remove these from their workspace.

```yaml
# /workspace/mcp/local.yaml
name: local
command: ["python", "/workspace/mcp_server.py"]
cwd: /workspace
```

Global scope — config lives in `harness.yaml` and applies to every user. The user
cannot edit or disable these from their workspace. On name conflict, the global
entry wins over a user entry with the same name.

```yaml
# harness.yaml
mcp:
  servers:
    - name: local
      command: ["python", "/workspace/mcp_server.py"]
      cwd: /workspace
```

Documentation
-------------

- [System and development rules](docs/system-and-development-rules.md)

Run
---

```bash
cp harness.yaml.example harness.yaml
# put llm.api_key in harness.yaml
uv run harness-agent --config harness.yaml ask "Say hi"
```

Direct CLI adapter:

```bash
uv run python cli.py --config harness.yaml --send "hi" --user_id 123
```

Telegram:

```bash
# set telegram.enabled: true and telegram.bot_token in harness.yaml
uv run harness-agent --config harness.yaml telegram
```

Tests
-----

Unit tests — no Docker, no network, no external APIs:

```bash
uv run pytest
```

Integration tests — require a running Docker daemon, and the OpenRouter
config at `~/.config/harness-agent/openrouter.yaml` for the real-LLM cases:

```bash
uv run pytest tests/integration
```
