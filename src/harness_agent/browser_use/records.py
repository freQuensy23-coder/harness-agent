"""Internal projection records persisted by SQLiteBrowserProfileStore /
SQLiteBrowserSessionStore. Distinct from the cloud DTOs so we control
the schema even if the upstream API changes."""

from datetime import datetime

from pydantic import BaseModel


LOCAL_STATUS_COMPLETED = "completed"
ACTIVE_LOCAL_STATUSES: frozenset[str] = frozenset({"created", "running", "idle"})


class BrowserProfileRecord(BaseModel):
    user_id: str
    cloud_profile_id: str
    created_at: datetime
    last_used_at: datetime


class BrowserSessionRecord(BaseModel):
    session_id: str
    user_id: str
    conversation_id: str
    generation: int
    parent_call_id: str
    cloud_session_id: str
    cloud_profile_id: str
    status: str
    keep_alive: bool
    task: str
    model: str
    live_url: str | None = None
    output: str | None = None
    error: str | None = None
    last_cloud_message_id: str | None = None
    created_at: datetime
    updated_at: datetime
