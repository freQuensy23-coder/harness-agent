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


def safe_docker_user_part(user_id: str) -> str:
    """Encode a user_id into an injective Docker-name-safe form.

    Docker container names match `[a-zA-Z0-9_.-]+` (after the first
    char). Percent-encoding is unusable because `%` is rejected, so we
    use `_` as the escape: literal `_` doubles to `__`, and any byte
    outside the Docker-safe set becomes `_HH`. Every `_` in the output
    is followed either by another `_` (literal) or by two hex digits
    (escaped byte), so distinct user_ids never collapse to the same
    container name.
    """
    out: list[str] = []
    for byte in user_id.encode("utf-8"):
        if byte == 0x5F:
            out.append("__")
        elif (
            0x30 <= byte <= 0x39
            or 0x41 <= byte <= 0x5A
            or 0x61 <= byte <= 0x7A
            or byte == 0x2E
            or byte == 0x2D
        ):
            out.append(chr(byte))
        else:
            out.append(f"_{byte:02x}")
    encoded = "".join(out)
    return encoded or "_"


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
