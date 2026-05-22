"""Behavioral tests for the spawned-process shell + Python scripts in
src/harness_agent/runtime/scripts.py.

The scripts are normally executed via `docker exec` inside a user
container. Here we run them on the host filesystem with sh/python, which
is the same surface they expose: argv positions, file paths, exit codes.
"""

import asyncio
import base64
import json
import subprocess
from pathlib import Path

import pytest

from harness_agent.runtime.scripts import (
    KILL_SPAWNED_PROCESS_SCRIPT,
    READ_SPAWNED_PROCESS_CODE,
    SPAWN_SHELL_SCRIPT,
)


def _spawn_paths(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base_path": base,
        "stdout_path": base / "stdout",
        "stderr_path": base / "stderr",
        "pid_path": base / "pid",
        "exit_code_path": base / "exit_code",
    }


def _run_spawn(paths: dict[str, Path], *, cwd: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "sh",
            "-c",
            SPAWN_SHELL_SCRIPT,
            "spawn",
            str(paths["base_path"]),
            str(paths["stdout_path"]),
            str(paths["stderr_path"]),
            str(paths["pid_path"]),
            str(paths["exit_code_path"]),
            str(cwd),
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _run_read(
    paths: dict[str, Path],
    *,
    stdout_offset: int,
    stderr_offset: int,
    max_bytes: int,
) -> dict:
    completed = subprocess.run(
        [
            "python3",
            "-c",
            READ_SPAWNED_PROCESS_CODE,
            str(paths["stdout_path"]),
            str(paths["stderr_path"]),
            str(stdout_offset),
            str(stderr_offset),
            str(max_bytes),
            str(paths["pid_path"]),
            str(paths["exit_code_path"]),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    return json.loads(completed.stdout)


def _decode(b64: str) -> str:
    return base64.b64decode(b64).decode("utf-8")


def test_spawn_runs_command_writes_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    result = _run_spawn(
        paths,
        cwd=tmp_path,
        command="printf hello; printf oops 1>&2; exit 7",
    )
    assert result.returncode == 7
    assert paths["stdout_path"].read_text() == "hello"
    assert paths["stderr_path"].read_text() == "oops"
    assert paths["exit_code_path"].read_text() == "7"
    pid_text = paths["pid_path"].read_text().strip()
    assert pid_text.isdigit()


def test_spawn_invalid_cwd_writes_failure_exit_code(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    result = _run_spawn(
        paths,
        cwd=tmp_path / "does-not-exist",
        command="echo never",
    )
    assert result.returncode == 1
    assert paths["exit_code_path"].read_text() == "1"
    # stdout/stderr files were created (the script touched them) but stay empty.
    assert paths["stdout_path"].read_text() == ""


def test_read_during_spawn_startup_does_not_report_exited(tmp_path: Path) -> None:
    """Race regression: shell.read can fire between the detached spawn
    helper's docker exec dispatch and its `echo $$ > pid_path`. If no pid
    and no exit_code file exist yet, the read MUST NOT report `exited`
    -- otherwise the runtime deletes the still-starting process record."""
    paths = _spawn_paths(tmp_path / "spawn")
    payload = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=1024)
    assert _decode(payload["stdout"]) == ""
    assert _decode(payload["stderr"]) == ""
    assert payload["stdout_offset"] == 0
    assert payload["stderr_offset"] == 0
    assert payload["exited"] is False
    # exit_code stays at the "still running" default of 0 so the LLM
    # reading the result does not see a spurious failure either.
    assert payload["exit_code"] == 0


def test_read_treats_pid_dead_without_exit_code_as_failed_exit(tmp_path: Path) -> None:
    """If the spawn helper wrote a pid but died before producing an
    exit code (eg killed by OOM), shell.read must report exited."""
    paths = _spawn_paths(tmp_path / "spawn")
    paths["pid_path"].write_text("99999999")  # unlikely PID -> dead
    payload = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=1024)
    assert payload["exited"] is True
    assert payload["exit_code"] == 1


def test_read_advances_offsets_byte_by_byte(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    paths["stdout_path"].write_bytes(b"abcdef")
    paths["stderr_path"].write_bytes(b"XYZ")
    paths["pid_path"].write_text("99999999")  # unlikely PID -> not running
    paths["exit_code_path"].write_text("0")

    first = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=3)
    assert _decode(first["stdout"]) == "abc"
    assert first["stdout_offset"] == 3
    assert _decode(first["stderr"]) == "XYZ"
    assert first["stderr_offset"] == 3
    assert first["exit_code"] == 0
    assert first["exited"] is True  # exit_code file present -> terminal
    # Drain signal: the offsets we got back must be comparable to the file
    # sizes so the runtime can keep the row alive until everything is read.
    assert first["stdout_size"] == 6
    assert first["stderr_size"] == 3
    assert first["stdout_offset"] < first["stdout_size"]


def test_read_reports_drained_when_offsets_reach_file_sizes(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    paths["stdout_path"].write_bytes(b"abcdef")
    paths["stderr_path"].write_bytes(b"")
    paths["pid_path"].write_text("99999999")
    paths["exit_code_path"].write_text("0")

    # First read pulls 3 bytes; not drained.
    first = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=3)
    assert first["stdout_offset"] == 3
    assert first["stdout_size"] == 6
    # Second read pulls the remaining 3; offsets now equal file size.
    second = _run_read(paths, stdout_offset=3, stderr_offset=0, max_bytes=10)
    assert _decode(second["stdout"]) == "def"
    assert second["stdout_offset"] == 6
    assert second["stdout_size"] == 6
    assert second["stderr_offset"] == 0
    assert second["stderr_size"] == 0
    assert second["exited"] is True


def test_read_treats_missing_exit_code_with_dead_pid_as_failure(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    paths["stdout_path"].write_bytes(b"")
    paths["stderr_path"].write_bytes(b"")
    paths["pid_path"].write_text("99999999")  # not running

    payload = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=10)
    assert payload["exit_code"] == 1


def test_read_treats_unparseable_exit_code_as_failure(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    paths["stdout_path"].write_bytes(b"")
    paths["stderr_path"].write_bytes(b"")
    paths["pid_path"].write_text("99999999")
    paths["exit_code_path"].write_text("not-an-int")

    payload = _run_read(paths, stdout_offset=0, stderr_offset=0, max_bytes=10)
    assert payload["exit_code"] == 1


def test_kill_removes_base_directory_with_no_live_process(tmp_path: Path) -> None:
    base = tmp_path / "spawn"
    paths = _spawn_paths(base)
    paths["stdout_path"].write_text("")
    paths["pid_path"].write_text("99999999")  # no live process to kill
    paths["exit_code_path"].write_text("0")

    result = subprocess.run(
        [
            "sh",
            "-c",
            KILL_SPAWNED_PROCESS_SCRIPT,
            "kill-spawned",
            str(paths["pid_path"]),
            str(paths["base_path"]),
            str(paths["exit_code_path"]),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0
    # Whole base directory is removed; exit_code_path was inside it.
    assert not base.exists()


async def _spawn_under_new_session(
    paths: dict[str, Path],
    *,
    cwd: Path,
    command: str,
) -> asyncio.subprocess.Process:
    """Start the spawn helper in its own process group, mirroring how
    production runs it under `setsid` inside the user's container. The
    wrapper's pid then equals its pgid, so `kill -- -pid` reaches the
    whole subtree."""
    return await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        SPAWN_SHELL_SCRIPT,
        "spawn",
        str(paths["base_path"]),
        str(paths["stdout_path"]),
        str(paths["stderr_path"]),
        str(paths["pid_path"]),
        str(paths["exit_code_path"]),
        str(cwd),
        command,
        start_new_session=True,
    )


async def _wait_for_pid_file(paths: dict[str, Path]) -> int:
    for _ in range(200):
        if paths["pid_path"].exists():
            text = paths["pid_path"].read_text().strip()
            if text:
                return int(text)
        await asyncio.sleep(0.02)
    raise AssertionError("spawn script never wrote a pid")


async def _run_kill_script(paths: dict[str, Path]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "sh",
        "-c",
        KILL_SPAWNED_PROCESS_SCRIPT,
        "kill-spawned",
        str(paths["pid_path"]),
        str(paths["base_path"]),
        str(paths["exit_code_path"]),
    )


@pytest.mark.asyncio
async def test_kill_terminates_a_live_child_process(tmp_path: Path) -> None:
    paths = _spawn_paths(tmp_path / "spawn")
    proc = await _spawn_under_new_session(paths, cwd=tmp_path, command="sleep 30")
    try:
        await _wait_for_pid_file(paths)
        kill = await _run_kill_script(paths)
        await asyncio.wait_for(kill.wait(), timeout=10)
        assert kill.returncode == 0
        await asyncio.wait_for(proc.wait(), timeout=10)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.asyncio
async def test_kill_escalates_to_descendants_that_ignore_sigterm(tmp_path: Path) -> None:
    """The user-supplied command can fork descendants and/or trap SIGTERM.
    kill-spawned must kill the whole process group (TERM then KILL),
    not just the wrapper pid, otherwise descendants keep running after
    the projection row is gone."""
    paths = _spawn_paths(tmp_path / "spawn")
    # Spawn a TERM-ignoring shell that itself starts a long-running child.
    proc = await _spawn_under_new_session(
        paths,
        cwd=tmp_path,
        command="trap '' TERM; sleep 60 & wait $!",
    )
    try:
        await _wait_for_pid_file(paths)
        kill = await _run_kill_script(paths)
        await asyncio.wait_for(kill.wait(), timeout=15)
        assert kill.returncode == 0
        # Both the wrapper and the trapping child must be gone soon
        # after kill returns; if any survived, proc.wait() would hang.
        await asyncio.wait_for(proc.wait(), timeout=15)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
