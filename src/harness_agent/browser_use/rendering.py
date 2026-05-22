"""JSON renderers for browser.* tool results consumed by the LLM."""

import json
from typing import Any

from harness_agent.browser_use.cloud_dtos import CloudMessage
from harness_agent.browser_use.records import BrowserSessionRecord


def render_browser_session(
    record: BrowserSessionRecord,
    *,
    messages: list[CloudMessage] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "session_id": record.session_id,
        "status": record.status,
        "task": record.task,
        "model": record.model,
        "keep_alive": record.keep_alive,
        "live_url": record.live_url,
        "output": record.output,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
    if messages is not None:
        payload["messages"] = [
            {
                "role": m.role,
                "summary": m.summary,
                "data": m.data,
            }
            for m in messages
        ]
    return json.dumps(payload, indent=2, default=str)


def render_browser_sessions(records: list[BrowserSessionRecord]) -> str:
    return json.dumps(
        [
            {
                "session_id": r.session_id,
                "status": r.status,
                "task": r.task,
                "keep_alive": r.keep_alive,
                "live_url": r.live_url,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in records
        ],
        indent=2,
        default=str,
    )
