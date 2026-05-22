"""Persistent memory: MEMORY.md and USER.md, char-budgeted and scanned."""

import re
from typing import Literal

from pydantic import BaseModel


MemoryTarget = Literal["memory", "user"]

MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
ENTRY_DELIMITER = "\n§\n"


def char_limit_for(target: MemoryTarget) -> int:
    return USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT


_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"authorized_keys", "ssh_backdoor"),
    (r"\$HOME/\.ssh|~/\.ssh", "ssh_access"),
)

_INVISIBLE_CHARS = (
    "​", "‌", "‍", "⁠", "﻿", "‪", "‫", "‬", "‭", "‮",
)


def scan_memory_content(content: str) -> str | None:
    """Memory content is injected into every future system prompt, so
    refuse prompt-injection / exfil payloads before they land on disk."""
    for char in _INVISIBLE_CHARS:
        if char in content:
            return (
                "Blocked: content contains invisible unicode character "
                f"U+{ord(char):04X} (possible injection)."
            )
    for pattern, identifier in _INJECTION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return (
                f"Blocked: content matches threat pattern '{identifier}'. "
                "Memory entries are injected into the system prompt and must "
                "not contain injection or exfiltration payloads."
            )
    return None


class MemoryMutationError(ValueError):
    pass


class MemoryDocument(BaseModel):
    target: MemoryTarget
    entries: list[str]

    @classmethod
    def parse(cls, target: MemoryTarget, raw: str) -> "MemoryDocument":
        if not raw.strip():
            return cls(target=target, entries=[])
        seen: list[str] = []
        for entry in (e.strip() for e in raw.split(ENTRY_DELIMITER)):
            if entry and entry not in seen:
                seen.append(entry)
        return cls(target=target, entries=seen)

    def render(self) -> str:
        return ENTRY_DELIMITER.join(self.entries) if self.entries else ""

    @property
    def char_count(self) -> int:
        return len(self.render())

    @property
    def limit(self) -> int:
        return char_limit_for(self.target)

    def usage_percent(self) -> int:
        return min(100, int((self.char_count / self.limit) * 100)) if self.limit else 0

    def add(self, content: str) -> str:
        content = content.strip()
        if not content:
            raise MemoryMutationError("Content cannot be empty.")
        if content in self.entries:
            return "Entry already exists (no duplicate added)."
        projected = ENTRY_DELIMITER.join([*self.entries, content])
        if len(projected) > self.limit:
            raise MemoryMutationError(
                f"Memory at {self.char_count:,}/{self.limit:,} chars. "
                f"Adding this entry ({len(content)} chars) would exceed the limit. "
                "Replace or remove existing entries first."
            )
        self.entries.append(content)
        return "Entry added."

    def replace(self, old_text: str, new_content: str) -> str:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            raise MemoryMutationError("old_text cannot be empty.")
        if not new_content:
            raise MemoryMutationError(
                "content cannot be empty. Use 'remove' to delete entries."
            )
        idx = self._find_unique(old_text)
        candidate = list(self.entries)
        candidate[idx] = new_content
        projected = ENTRY_DELIMITER.join(candidate)
        if len(projected) > self.limit:
            raise MemoryMutationError(
                f"Replacement would put memory at {len(projected):,}/{self.limit:,} chars. "
                "Shorten the new content or remove other entries first."
            )
        self.entries = candidate
        return "Entry replaced."

    def remove(self, old_text: str) -> str:
        old_text = old_text.strip()
        if not old_text:
            raise MemoryMutationError("old_text cannot be empty.")
        idx = self._find_unique(old_text)
        del self.entries[idx]
        return "Entry removed."

    def _find_unique(self, old_text: str) -> int:
        matches = [(i, e) for i, e in enumerate(self.entries) if old_text in e]
        if not matches:
            raise MemoryMutationError(f"No entry matched '{old_text}'.")
        if len(matches) > 1 and len({e for _, e in matches}) > 1:
            previews = ", ".join(
                f"{e[:60]}{'...' if len(e) > 60 else ''}" for _, e in matches
            )
            raise MemoryMutationError(
                f"Multiple entries matched '{old_text}'. Be more specific. "
                f"Matched: {previews}"
            )
        return matches[0][0]
