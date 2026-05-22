SPAWN_SHELL_SCRIPT = """
set -u
base_path=$1
stdout_path=$2
stderr_path=$3
pid_path=$4
exit_code_path=$5
cwd=$6
command=$7

mkdir -p "$base_path"
: > "$stdout_path"
: > "$stderr_path"
rm -f "$exit_code_path"
# Write the wrapper pid; the wrapper becomes the process-group leader for
# the child shell + any descendants it forks. pid is also the pgid (since
# this script runs under `setsid` from the runtime).
echo $$ > "$pid_path"

if ! cd "$cwd"; then
    printf '1' > "$exit_code_path"
    exit 1
fi

child=
terminate() {
    if [ -n "${child:-}" ]; then
        # Kill the whole process group so descendants don't survive.
        kill -TERM -"$$" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    printf '143' > "$exit_code_path"
    exit 143
}
trap terminate TERM INT HUP

sh -lc "$command" > "$stdout_path" 2> "$stderr_path" < /dev/null &
child=$!
wait "$child"
code=$?
printf '%s' "$code" > "$exit_code_path"
exit "$code"
"""


READ_SPAWNED_PROCESS_CODE = """
import base64
import json
import os
import pathlib
import sys

stdout_path = pathlib.Path(sys.argv[1])
stderr_path = pathlib.Path(sys.argv[2])
stdout_offset = int(sys.argv[3])
stderr_offset = int(sys.argv[4])
max_bytes = int(sys.argv[5])
pid_path = pathlib.Path(sys.argv[6])
exit_code_path = pathlib.Path(sys.argv[7])

def read_available(path, offset):
    # Offsets are bytes already returned by shell.read. Store byte positions, not
    # decoded character counts, because output files and max_bytes are byte-based.
    if not path.exists():
        return "", offset
    size = path.stat().st_size
    start = min(offset, size)
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(max_bytes)
        next_offset = stream.tell()
    return base64.b64encode(data).decode("ascii"), next_offset

def is_running():
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

stdout, next_stdout_offset = read_available(stdout_path, stdout_offset)
stderr, next_stderr_offset = read_available(stderr_path, stderr_offset)
exit_code = 0
exited = False
if exit_code_path.exists():
    # Spawn helper wrote a final exit code: process is durably done.
    exited = True
    try:
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
    except ValueError:
        exit_code = 1
elif pid_path.exists() and not is_running():
    # Pid was written and the kernel says that pid is gone, but the
    # spawn helper never wrote an exit code: treat as terminal failure.
    exited = True
    exit_code = 1
# Otherwise (no pid file yet) we are between docker exec dispatch and
# the spawn helper's `echo $$ > pid_path`. Do NOT mark exited; a follow-up
# read will see the pid file or the exit code.

def _file_size(path):
    return path.stat().st_size if path.exists() else 0

print(json.dumps({
    "stdout": stdout,
    "stderr": stderr,
    "stdout_offset": next_stdout_offset,
    "stderr_offset": next_stderr_offset,
    "stdout_size": _file_size(stdout_path),
    "stderr_size": _file_size(stderr_path),
    "exit_code": exit_code,
    "exited": exited,
}))
"""


KILL_SPAWNED_PROCESS_SCRIPT = """
set -u
pid_path=$1
base_path=$2
exit_code_path=$3

# The spawn helper runs under `setsid`, so the pid we stored is also
# the pgid for the whole subtree (wrapper + child shell + any
# descendants the user command forked). Send TERM then KILL to the
# group so nothing survives the kill.
if [ -f "$pid_path" ]; then
    pid=$(cat "$pid_path" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM -"$pid" 2>/dev/null || true
        i=0
        while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do
            sleep 0.1
            i=$((i + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -"$pid" 2>/dev/null || true
            # Final reap loop; if the wrapper is still alive after KILL
            # the kernel has bigger problems and we bail out without
            # destroying the base dir so the next shell.read can still
            # report the live pid.
            j=0
            while kill -0 "$pid" 2>/dev/null && [ "$j" -lt 50 ]; do
                sleep 0.1
                j=$((j + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                exit 1
            fi
        fi
    fi
fi

printf '143' > "$exit_code_path" 2>/dev/null || true
rm -rf "$base_path"
"""
