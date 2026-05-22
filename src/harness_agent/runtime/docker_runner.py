import asyncio

from harness_agent.runtime.models import DockerProcessResult


_TIMEOUT_REAP_GRACE_SECONDS = 5.0


class AsyncioDockerRunner:
    async def run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> DockerProcessResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if timeout_seconds is None:
            stdout, stderr = await process.communicate(stdin)
        else:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                # Don't leak the child: SIGTERM first, escalate to SIGKILL
                # if it ignores TERM, then reap so process.returncode is
                # populated and OS resources are released.
                await _kill_and_reap(process)
                raise
        if process.returncode is None:
            raise RuntimeError(f"subprocess {argv[0]} did not exit")
        return DockerProcessResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=_TIMEOUT_REAP_GRACE_SECONDS)
        return
    except TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return
    try:
        await process.wait()
    except Exception:
        # Best effort; we still raised the original TimeoutError upstream.
        return
