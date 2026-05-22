"""Unit tests for AsyncioDockerRunner subprocess timeout handling.

The runner shells out to `docker` in production, but the
async subprocess primitives are the same for any executable, so we
exercise the timeout branch against a host-side `sleep` to keep the
test deterministic and Docker-free.
"""

import asyncio
import os
import signal

import pytest

from harness_agent.runtime.docker_runner import AsyncioDockerRunner


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.asyncio
async def test_run_reaps_subprocess_on_timeout() -> None:
    """A long-running child that exceeds timeout_seconds must be killed
    AND reaped before run() returns/raises. Otherwise a timed-out
    docker exec leaks one zombie per failed tool call."""
    runner = AsyncioDockerRunner()
    pid_holder: dict[str, int] = {}

    original = asyncio.create_subprocess_exec

    async def _capturing_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await original(*args, **kwargs)  # type: ignore[arg-type]
        pid_holder["pid"] = process.pid
        return process

    asyncio.create_subprocess_exec = _capturing_create  # type: ignore[assignment]
    try:
        with pytest.raises(TimeoutError):
            await runner.run(["sleep", "10"], timeout_seconds=1)
    finally:
        asyncio.create_subprocess_exec = original  # type: ignore[assignment]

    assert "pid" in pid_holder
    for _ in range(50):
        if not _process_is_alive(pid_holder["pid"]):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"timed-out subprocess pid {pid_holder['pid']} is still alive after run() returned"
    )


@pytest.mark.asyncio
async def test_run_kills_subprocess_that_ignores_sigterm_on_timeout() -> None:
    runner = AsyncioDockerRunner()
    pid_holder: dict[str, int] = {}

    original = asyncio.create_subprocess_exec

    async def _capturing_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await original(*args, **kwargs)  # type: ignore[arg-type]
        pid_holder["pid"] = process.pid
        return process

    asyncio.create_subprocess_exec = _capturing_create  # type: ignore[assignment]
    try:
        with pytest.raises(TimeoutError):
            await runner.run(
                ["sh", "-c", "trap '' TERM; sleep 30"],
                timeout_seconds=1,
            )
    finally:
        asyncio.create_subprocess_exec = original  # type: ignore[assignment]

    for _ in range(100):
        if not _process_is_alive(pid_holder["pid"]):
            return
        await asyncio.sleep(0.1)
    try:
        os.kill(pid_holder["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise AssertionError(
        f"TERM-trapping subprocess pid {pid_holder['pid']} survived runner.run() timeout"
    )


@pytest.mark.asyncio
async def test_run_returns_normally_when_command_completes_under_timeout() -> None:
    runner = AsyncioDockerRunner()
    result = await runner.run(["sh", "-c", "echo hi"], timeout_seconds=10)
    assert result.exit_code == 0
    assert result.stdout == "hi\n"


@pytest.mark.asyncio
async def test_run_returns_normally_with_no_timeout_argument() -> None:
    runner = AsyncioDockerRunner()
    result = await runner.run(["sh", "-c", "printf hello"])
    assert result.exit_code == 0
    assert result.stdout == "hello"
