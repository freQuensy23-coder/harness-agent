from typing import Literal

from harness_agent.bus import EventBus
from harness_agent.events import (
    ShellProcessOutputAdvanced,
    ShellProcessSpawned,
    ShellProcessTerminated,
)
from harness_agent.runtime.models import SpawnedProcessRecord
from harness_agent.runtime.protocols import SpawnedProcessStore


TerminationReason = Literal["killed", "spawn_failed", "exited"]


class EventfulSpawnedProcessStore:
    """Decorator that publishes typed spawned-process lifecycle events on
    the bus before delegating each mutation to the underlying store.

    Keeps DockerUserRuntime ignorant of EventBus while making the
    spawned-process lifecycle visible to the audit log. The wrapped
    store is the durable projection of the published events.
    """

    def __init__(self, *, store: SpawnedProcessStore, bus: EventBus) -> None:
        self._store = store
        self._bus = bus

    async def create(self, record: SpawnedProcessRecord) -> SpawnedProcessRecord:
        await self._bus.publish(
            ShellProcessSpawned(
                user_id=record.user_id,
                process_id=record.process_id,
                container_name=record.container_name,
                command=record.command,
                cwd=record.cwd,
                base_path=record.base_path,
            )
        )
        return await self._store.create(record)

    async def get(
        self,
        *,
        process_id: str,
        user_id: str,
    ) -> SpawnedProcessRecord | None:
        return await self._store.get(process_id=process_id, user_id=user_id)

    async def update_offsets(
        self,
        *,
        process_id: str,
        user_id: str,
        stdout_offset: int,
        stderr_offset: int,
    ) -> None:
        await self._bus.publish(
            ShellProcessOutputAdvanced(
                user_id=user_id,
                process_id=process_id,
                stdout_offset=stdout_offset,
                stderr_offset=stderr_offset,
            )
        )
        await self._store.update_offsets(
            process_id=process_id,
            user_id=user_id,
            stdout_offset=stdout_offset,
            stderr_offset=stderr_offset,
        )

    async def delete(
        self,
        *,
        process_id: str,
        user_id: str,
        reason: TerminationReason = "killed",
    ) -> None:
        await self._bus.publish(
            ShellProcessTerminated(
                user_id=user_id,
                process_id=process_id,
                reason=reason,
            )
        )
        await self._store.delete(process_id=process_id, user_id=user_id)
