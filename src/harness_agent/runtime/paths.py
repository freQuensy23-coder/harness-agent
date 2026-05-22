import posixpath
import urllib.parse


def safe_conversation_id_part(conversation_id: str) -> str:
    """Encode a conversation_id into an injective filesystem-safe form.

    Percent-encoding via RFC 3986's unreserved set guarantees that
    distinct conversation_ids always produce distinct on-disk names —
    `tg:456` and `tg-456` are kept apart, so a Telegram session and a
    CLI session never share the same JSONL log.

    The encoding is reversible. `DockerUserRuntime.list_session_logs`
    relies on `urllib.parse.unquote` to recover the raw conversation
    IDs from filenames so callers compare like-with-like.
    """
    return urllib.parse.quote(conversation_id, safe="-._~")


def workspace_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if not normalized.startswith("/"):
        normalized = posixpath.normpath(f"/workspace/{normalized}")
    if normalized != "/workspace" and not normalized.startswith("/workspace/"):
        raise ValueError(f"path must stay inside /workspace: {path}")
    return normalized


def content_path(path: str) -> str:
    normalized = workspace_path(path)
    if normalized != "/workspace/content" and not normalized.startswith("/workspace/content/"):
        raise ValueError(f"content path must stay inside /workspace/content: {path}")
    return normalized
