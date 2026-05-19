import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator


class ConversationTurnCoordinator:
    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._generations: dict[str, int] = {}
        self._run_locks: dict[str, asyncio.Lock] = {}

    async def request_generation(self, conversation_id: str) -> int:
        async with self._state_lock:
            generation = self._generations.get(conversation_id, 0) + 1
            self._generations[conversation_id] = generation
            if conversation_id not in self._run_locks:
                self._run_locks[conversation_id] = asyncio.Lock()
            return generation

    async def current_generation(self, conversation_id: str) -> int:
        async with self._state_lock:
            return self._generations.get(conversation_id, 0)

    async def is_current(self, conversation_id: str, generation: int) -> bool:
        return await self.current_generation(conversation_id) == generation

    @asynccontextmanager
    async def run_slot(self, conversation_id: str) -> AsyncIterator[None]:
        async with self._state_lock:
            if conversation_id not in self._run_locks:
                self._run_locks[conversation_id] = asyncio.Lock()
            run_lock = self._run_locks[conversation_id]
        async with run_lock:
            yield
