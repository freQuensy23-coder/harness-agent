import base64
import hashlib
import mimetypes
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel


class ContentRef(BaseModel):
    kind: Literal["image", "file"]
    file_name: str
    mime_type: str
    size_bytes: int
    sha256: str
    workspace_path: str
    content_base64: str | None = None


class WorkspaceFile(BaseModel):
    path: str
    content: bytes


def content_ref_from_workspace_file(file: WorkspaceFile) -> ContentRef:
    mime_type = detect_mime_type(file.path, file.content)
    return ContentRef(
        kind="image" if is_image_mime_type(mime_type) else "file",
        file_name=PurePosixPath(file.path).name,
        mime_type=mime_type,
        size_bytes=len(file.content),
        sha256=hashlib.sha256(file.content).hexdigest(),
        workspace_path=file.path,
        content_base64=base64.b64encode(file.content).decode("ascii")
        if is_image_mime_type(mime_type)
        else None,
    )


def detect_mime_type(path: str, content: bytes) -> str:
    magic_mime_type = detect_magic_mime_type(content)
    if magic_mime_type is not None:
        return magic_mime_type
    path_mime_type, _ = mimetypes.guess_type(path)
    if path_mime_type is not None:
        return path_mime_type
    return "application/octet-stream"


def detect_magic_mime_type(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def is_image_mime_type(mime_type: str) -> bool:
    return mime_type.startswith("image/")
