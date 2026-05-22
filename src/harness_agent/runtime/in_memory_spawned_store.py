from harness_agent.runtime.models import SpawnedProcessRecord


class InMemorySpawnedProcessStore:
    """Process-local spawned-process metadata.

    Use this in tests and ephemeral CLI runs where a SQLite file on disk
    would just leak temp state. Production composes
    `SQLiteSpawnedProcessStore` instead.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], SpawnedProcessRecord] = {}

    async def create(self, record: SpawnedProcessRecord) -> SpawnedProcessRecord:
        self._records[(record.process_id, record.user_id)] = record
        return record

    async def get(
        self,
        *,
        process_id: str,
        user_id: str,
    ) -> SpawnedProcessRecord | None:
        return self._records.get((process_id, user_id))

    async def update_offsets(
        self,
        *,
        process_id: str,
        user_id: str,
        stdout_offset: int,
        stderr_offset: int,
    ) -> None:
        existing = self._records.get((process_id, user_id))
        if existing is None:
            return
        self._records[(process_id, user_id)] = existing.model_copy(
            update={"stdout_offset": stdout_offset, "stderr_offset": stderr_offset}
        )

    async def delete(self, *, process_id: str, user_id: str) -> None:
        self._records.pop((process_id, user_id), None)
