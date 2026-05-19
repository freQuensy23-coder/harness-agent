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

- `/workspace/agent/*.md`
- `/workspace/skills/*/SKILL.md`
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
- `mcp.<server>.<tool>`

User MCP YAML
-------------

User MCP config lives inside that user's Docker workspace, not in global server config.

```yaml
# /workspace/mcp/local.yaml
name: local
command: ["python", "/workspace/mcp_server.py"]
cwd: /workspace
```

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

```bash
uv run pytest
```
