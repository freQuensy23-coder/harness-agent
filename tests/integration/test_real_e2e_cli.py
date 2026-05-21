import hashlib
import json
import sqlite3
import struct
import subprocess
import zlib
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


def test_cli_real_e2e_openrouter_docker_sqlite(tmp_path: Path) -> None:
    openrouter_config_path = Path.home() / ".config" / "harness-agent" / "openrouter.yaml"
    if not openrouter_config_path.exists():
        pytest.skip(f"missing YAML config: {openrouter_config_path}")
    openrouter_config = yaml.safe_load(openrouter_config_path.read_text(encoding="utf-8"))

    prefix = f"harness-real-{uuid4().hex[:8]}"
    user_id = f"real-{uuid4().hex[:8]}"
    config_path = tmp_path / "harness.yaml"
    data_path = tmp_path / "harness.sqlite3"
    config_path.write_text(
        "\n".join(
            [
                "database:",
                f"  path: {data_path}",
                "llm:",
                "  base_url: https://openrouter.ai/api/v1",
                f"  api_key: \"{openrouter_config['llm']['api_key']}\"",
                "  model: z-ai/glm-5v-turbo",
                "telegram:",
                "  enabled: false",
                "  bot_token: null",
                "runtime:",
                "  docker:",
                "    image: python:3.14-slim",
                f"    container_prefix: {prefix}",
                "    network: bridge",
                "    memory: 1g",
                "    cpus: \"1\"",
            ]
        ),
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "cli.py",
                "--config",
                str(config_path),
                "--send",
                (
                    "Use file.write to create /workspace/real_e2e.txt containing "
                    "only this exact five-letter lowercase text: proven. Then use "
                    "file.read to read it. Answer only the file content."
                ),
                "--user_id",
                user_id,
                "--conversation_id",
                f"cli:{user_id}",
            ],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=True,
        )
        assert completed.stdout.strip() == "proven"
        assert (tmp_path / "harness.events.sqlite3").exists()
        assert (tmp_path / "harness.llm.sqlite3").exists()
        assert (tmp_path / "harness.messages.sqlite3").exists()
        with sqlite3.connect(tmp_path / "harness.messages.sqlite3") as db:
            rows = db.execute(
                """
                select item_kind, tool_name
                from conversation_items
                order by sequence asc
                """
            ).fetchall()
        assert ("assistant_tool_call", "file.write") in rows
        assert ("tool_result", "file.write") in rows
        assert ("assistant_tool_call", "file.read") in rows
        assert ("tool_result", "file.read") in rows
        with sqlite3.connect(tmp_path / "harness.messages.sqlite3") as db:
            tool_history_json = [
                row[0]
                for row in db.execute(
                    """
                    select message_json
                    from conversation_items
                    where item_kind in ('assistant_tool_call', 'tool_result')
                    order by sequence asc
                    """
                ).fetchall()
            ]
        with sqlite3.connect(tmp_path / "harness.llm.sqlite3") as db:
            first_turn_llm_count = db.execute(
                "select count(*) from llm_requests"
            ).fetchone()[0]

        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "cli.py",
                "--config",
                str(config_path),
                "--send",
                "Say hi. Answer only hi.",
                "--user_id",
                user_id,
                "--conversation_id",
                f"cli:{user_id}",
            ],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=True,
        )
        with sqlite3.connect(tmp_path / "harness.llm.sqlite3") as db:
            second_turn_message_json = json.loads(
                db.execute(
                    """
                    select message_json
                    from llm_requests
                    where sequence > ?
                    order by sequence asc
                    limit 1
                    """,
                    (first_turn_llm_count,),
                ).fetchone()[0]
            )
        second_turn_tool_history_json = [
            message
            for message in second_turn_message_json
            if json.loads(message)["kind"] in {"assistant_tool_call", "tool_result"}
        ]
        assert second_turn_tool_history_json == tool_history_json
    finally:
        container_name = f"{prefix}-u-{user_id}"
        volume_name = f"{container_name}-workspace"
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "volume", "rm", volume_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_cli_real_e2e_openrouter_docker_image_file_read(tmp_path: Path) -> None:
    openrouter_config_path = Path.home() / ".config" / "harness-agent" / "openrouter.yaml"
    if not openrouter_config_path.exists():
        pytest.skip(f"missing YAML config: {openrouter_config_path}")
    openrouter_config = yaml.safe_load(openrouter_config_path.read_text(encoding="utf-8"))

    prefix = f"harness-vision-{uuid4().hex[:8]}"
    user_id = f"vision-{uuid4().hex[:8]}"
    container_name = f"{prefix}-u-{user_id}"
    volume_name = f"{container_name}-workspace"
    config_path = tmp_path / "harness.yaml"
    data_path = tmp_path / "harness.sqlite3"
    config_path.write_text(
        "\n".join(
            [
                "database:",
                f"  path: {data_path}",
                "llm:",
                "  base_url: https://openrouter.ai/api/v1",
                f"  api_key: \"{openrouter_config['llm']['api_key']}\"",
                "  model: z-ai/glm-5v-turbo",
                "telegram:",
                "  enabled: false",
                "  bot_token: null",
                "runtime:",
                "  docker:",
                "    image: python:3.14-slim",
                f"    container_prefix: {prefix}",
                "    network: bridge",
                "    memory: 1g",
                "    cpus: \"1\"",
            ]
        ),
        encoding="utf-8",
    )
    image_bytes = red_square_png()

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{volume_name}:/workspace",
                "-w",
                "/workspace",
                "python:3.14-slim",
                "sleep",
                "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_name,
                "sh",
                "-lc",
                "mkdir -p /workspace/content && cat > /workspace/content/red-square.png",
            ],
            input=image_bytes,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )

        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "cli.py",
                "--config",
                str(config_path),
                "--send",
                (
                    "Use file.read to open /workspace/content/red-square.png. "
                    "After the image is attached to context, inspect the image and "
                    "answer exactly: RED SQUARE"
                ),
                "--user_id",
                user_id,
                "--conversation_id",
                f"cli:{user_id}",
            ],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=True,
        )
        assert completed.stdout.strip().upper() == "RED SQUARE"
        with sqlite3.connect(tmp_path / "harness.messages.sqlite3") as db:
            rows = db.execute(
                """
                select item_kind, tool_name, message_json
                from conversation_items
                order by sequence asc
                """
            ).fetchall()
        assert ("assistant_tool_call", "file.read") in [
            (row[0], row[1]) for row in rows
        ]
        tool_context = [
            json.loads(row[2])
            for row in rows
            if row[0] == "tool_context" and row[1] == "file.read"
        ][0]
        attachment = tool_context["attachments"][0]
        assert attachment["kind"] == "image"
        assert attachment["mime_type"] == "image/png"
        assert attachment["sha256"] == hashlib.sha256(image_bytes).hexdigest()
        assert attachment["content_base64"]
        with sqlite3.connect(tmp_path / "harness.llm.sqlite3") as db:
            llm_messages = json.loads(
                db.execute(
                    """
                    select message_json
                    from llm_requests
                    order by sequence desc
                    limit 1
                    """
                ).fetchone()[0]
            )
        parsed_messages = [json.loads(message) for message in llm_messages]
        assert sum(len(message.get("attachments", [])) for message in parsed_messages) == 1
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "volume", "rm", volume_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_cli_real_e2e_openrouter_docker_subagent_file_work(tmp_path: Path) -> None:
    openrouter_config_path = Path.home() / ".config" / "harness-agent" / "openrouter.yaml"
    if not openrouter_config_path.exists():
        pytest.skip(f"missing YAML config: {openrouter_config_path}")
    openrouter_config = yaml.safe_load(openrouter_config_path.read_text(encoding="utf-8"))

    prefix = f"harness-subagent-{uuid4().hex[:8]}"
    user_id = f"subagent-{uuid4().hex[:8]}"
    container_name = f"{prefix}-u-{user_id}"
    volume_name = f"{container_name}-workspace"
    config_path = tmp_path / "harness.yaml"
    data_path = tmp_path / "harness.sqlite3"
    config_path.write_text(
        "\n".join(
            [
                "database:",
                f"  path: {data_path}",
                "llm:",
                "  base_url: https://openrouter.ai/api/v1",
                f"  api_key: \"{openrouter_config['llm']['api_key']}\"",
                "  model: z-ai/glm-5v-turbo",
                "telegram:",
                "  enabled: false",
                "  bot_token: null",
                "runtime:",
                "  docker:",
                "    image: python:3.14-slim",
                f"    container_prefix: {prefix}",
                "    network: bridge",
                "    memory: 1g",
                "    cpus: \"1\"",
            ]
        ),
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "cli.py",
                "--config",
                str(config_path),
                "--send",
                (
                    "Use agent.run exactly once. Give the sub-agent this task: "
                    "use file.write to create /workspace/subagent_real.txt containing "
                    "only delegated, then use file.read to read /workspace/subagent_real.txt, "
                    "then answer only delegated. After the sub-agent returns, answer only delegated."
                ),
                "--user_id",
                user_id,
                "--conversation_id",
                f"cli:{user_id}",
            ],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=True,
        )
        assert completed.stdout.strip().lower() == "delegated"
        docker_file = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "cat",
                "/workspace/subagent_real.txt",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=True,
        )
        assert docker_file.stdout == "delegated"
        with sqlite3.connect(tmp_path / "harness.events.sqlite3") as db:
            event_types = [
                row[0]
                for row in db.execute(
                    """
                    select type
                    from events
                    order by sequence asc
                    """
                ).fetchall()
            ]
        assert "subagent.started" in event_types
        assert "subagent.completed" in event_types
        with sqlite3.connect(tmp_path / "harness.messages.sqlite3") as db:
            rows = db.execute(
                """
                select conversation_id, item_kind, tool_name
                from conversation_items
                order by sequence asc
                """
            ).fetchall()
        assert ("assistant_tool_call", "agent.run") in [
            (row[1], row[2]) for row in rows
        ]
        assert ("assistant_tool_call", "file.write") in [
            (row[1], row[2]) for row in rows
        ]
        assert ("assistant_tool_call", "file.read") in [
            (row[1], row[2]) for row in rows
        ]
        child_conversations = [
            row[0]
            for row in rows
            if row[1] == "assistant_tool_call" and row[2] == "file.write"
        ]
        assert child_conversations[0].startswith(f"cli:{user_id}:subagent:")
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "volume", "rm", volume_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def red_square_png() -> bytes:
    width = 80
    height = 80
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )
