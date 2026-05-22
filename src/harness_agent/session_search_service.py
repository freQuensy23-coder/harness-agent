"""Cross-session recall over JSONL session logs.

Two stages: rank logs by query hit count, then summarise the top-K
through the auxiliary LLM. Raw transcripts never reach the agent — the
contract is "summaries only" because the JSONL may capture tool stdout
with secrets the user didn't expect to see verbatim.
"""

import json
import re

from harness_agent.llm import (
    LlmClient,
    LlmRequest,
    UserMessage as LlmUserMessage,
)
from harness_agent.runtime import RuntimeToolResult
from harness_agent.runtime.protocols import UserRuntime
from harness_agent.tools import SessionSearchInput


_SUMMARIZER_SYSTEM_PROMPT = (
    "You are reviewing a past conversation transcript to help recall what "
    "happened. Summarise the conversation focused on the search topic. "
    "Include: 1) what the user asked about or wanted to accomplish, "
    "2) what actions were taken and what came of them, "
    "3) decisions, solutions, or conclusions, "
    "4) specific commands, paths, URLs, or technical details worth recalling, "
    "5) anything notable that was left unresolved. "
    "Be thorough but concise; preserve specifics. Past tense, factual recap."
)


class SessionSearchService:
    def __init__(self, runtime: UserRuntime, llm: LlmClient) -> None:
        self._runtime = runtime
        self._llm = llm

    async def execute(
        self,
        *,
        user_id: str,
        current_conversation_id: str,
        input: SessionSearchInput,
    ) -> RuntimeToolResult:
        query = input.query.strip()
        if not query:
            return RuntimeToolResult(stderr="query is required.\n", exit_code=1)
        terms = [t for t in re.split(r"\s+", query.lower()) if t] or [query.lower()]
        # Runtime contract: list returns RAW conversation IDs, so we
        # compare raw-to-raw and never double-encode.
        all_ids = await self._runtime.list_session_logs(user_id)
        candidates = [cid for cid in all_ids if cid != current_conversation_id]
        if not candidates:
            return _empty(query, "No prior sessions available to search.")
        scored: list[tuple[str, str, int]] = []
        for cid in candidates:
            content = await self._runtime.read_session_log(user_id, cid)
            if not content:
                continue
            lowered = content.lower()
            score = sum(lowered.count(term) for term in terms)
            if score > 0:
                scored.append((cid, content, score))
        scored.sort(key=lambda row: row[2], reverse=True)
        top = scored[: input.limit]
        if not top:
            return _empty(query, "No matching sessions found.")
        results: list[dict[str, str]] = []
        for cid, content, score in top:
            transcript = truncate_around_terms(
                format_session_transcript(content), terms
            )
            try:
                summary = await self._summarize(
                    user_id=user_id,
                    conversation_id=cid,
                    query=query,
                    transcript=transcript,
                )
            except Exception as exc:
                summary = f"[summary unavailable: {type(exc).__name__}]"
            results.append(
                {"conversation_id": cid, "match_score": str(score), "summary": summary}
            )
        return RuntimeToolResult(
            stdout=json.dumps(
                {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                },
                ensure_ascii=False,
            )
        )

    async def _summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        query: str,
        transcript: str,
    ) -> str:
        response = await self._llm.respond(
            LlmRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                generation=0,
                system=_SUMMARIZER_SYSTEM_PROMPT,
                messages=[
                    LlmUserMessage(
                        text=(
                            f"Search topic: {query}\n\n"
                            f"CONVERSATION TRANSCRIPT:\n{transcript}\n\n"
                            f"Summarise the conversation focused on: {query}"
                        )
                    )
                ],
                tools=[],
            )
        )
        if response.kind == "assistant_text":
            return response.text
        return "[summary unavailable: model returned a tool call instead of text]"


def _empty(query: str, message: str) -> RuntimeToolResult:
    return RuntimeToolResult(
        stdout=json.dumps(
            {"success": True, "query": query, "results": [], "message": message},
            ensure_ascii=False,
        )
    )


def format_session_transcript(jsonl_content: str) -> str:
    lines: list[str] = []
    for raw in jsonl_content.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            # A corrupt JSONL line must never leak raw text to the
            # summariser — it could carry secrets from another session.
            lines.append("[unparseable line]")
            continue
        role = str(entry.get("role", "")).upper() or "?"
        text = entry.get("text") or entry.get("content") or ""
        tool = entry.get("tool_name")
        if role == "TOOL" and tool:
            stdout = entry.get("stdout") or ""
            stderr = entry.get("stderr") or ""
            tool_input = entry.get("input")
            input_repr = (
                json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
                if tool_input is not None
                else ""
            )
            payload = stdout or text
            if stderr:
                payload = (
                    f"{payload}\n[stderr] {stderr}" if payload else f"[stderr] {stderr}"
                )
            if len(payload) > 400:
                payload = payload[:200] + "…[truncated]…" + payload[-200:]
            header = f"[TOOL:{tool}]"
            if input_repr:
                header = f"{header} input={input_repr}"
            lines.append(f"{header} {payload}" if payload else header)
            continue
        if role == "ASSISTANT_TOOL_CALL" and tool:
            lines.append(f"[ASSISTANT calls {tool}]")
            continue
        lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


def truncate_around_terms(
    transcript: str,
    terms: list[str],
    max_chars: int = 8000,
) -> str:
    if len(transcript) <= max_chars:
        return transcript
    lowered = transcript.lower()
    positions: list[int] = []
    for term in terms:
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(term)
    if not positions:
        return transcript[:max_chars] + "\n…[truncated]"
    center = sum(positions) // len(positions)
    half = max_chars // 2
    start = max(0, center - half)
    end = min(len(transcript), start + max_chars)
    return (
        ("…[truncated]\n" if start > 0 else "")
        + transcript[start:end]
        + ("\n…[truncated]" if end < len(transcript) else "")
    )
