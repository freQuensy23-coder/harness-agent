"""Wire-format DTOs for the browser-use cloud REST API. Fields use
`validation_alias`/`serialization_alias` so both snake_case and the
cloud's camelCase keys round-trip; `populate_by_name=True` lets test
code build instances with python attribute names."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CloudSessionStatus = Literal["created", "idle", "running", "stopped", "timed_out", "error"]


class CloudSessionState(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    status: CloudSessionStatus
    step_count: int = Field(default=0, validation_alias="stepCount", serialization_alias="stepCount")
    output: Any = None
    last_step_summary: str | None = Field(
        default=None, validation_alias="lastStepSummary", serialization_alias="lastStepSummary"
    )
    live_url: str | None = Field(
        default=None, validation_alias="liveUrl", serialization_alias="liveUrl"
    )
    profile_id: str | None = Field(
        default=None, validation_alias="profileId", serialization_alias="profileId"
    )
    total_cost_usd: str | None = Field(
        default=None, validation_alias="totalCostUsd", serialization_alias="totalCostUsd"
    )
    is_task_successful: bool | None = Field(
        default=None, validation_alias="isTaskSuccessful", serialization_alias="isTaskSuccessful"
    )


class CloudProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    name: str | None = None
    user_id: str | None = Field(
        default=None, validation_alias="userId", serialization_alias="userId"
    )


class CloudMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    session_id: str = Field(validation_alias="sessionId", serialization_alias="sessionId")
    role: str
    data: str
    summary: str | None = None
    type: str | None = None


class CloudMessagesPage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    messages: list[CloudMessage] = Field(default_factory=list[CloudMessage])
    has_more: bool = Field(
        default=False, validation_alias="hasMore", serialization_alias="hasMore"
    )
