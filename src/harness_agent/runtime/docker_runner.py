import asyncio

from harness_agent.runtime.models import DockerProcessResult


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
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin),
                timeout=timeout_seconds,
            )
        if process.returncode is None:
            raise RuntimeError(f"subprocess {argv[0]} did not exit")
        return DockerProcessResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )
