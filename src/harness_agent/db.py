from collections.abc import Iterable
from typing import Any, cast

import aiosqlite


async def fetchall_rows(
    db: aiosqlite.Connection,
    sql: str,
    parameters: Iterable[Any] = (),
) -> list[tuple[Any, ...]]:
    return cast(list[tuple[Any, ...]], await db.execute_fetchall(sql, parameters))
