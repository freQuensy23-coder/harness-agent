import posixpath


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
