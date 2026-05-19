import json
import sqlite3
import subprocess
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
                "  model: z-ai/glm-5.1",
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
                "Use file.write to create /workspace/real_e2e.txt containing exactly proven, then use file.read to read it, then answer only the file content.",
                "--user_id",
                user_id,
                "--conversation_id",
                f"cli:{user_id}",
            ],
            cwd=Path(__file__).parents[1],
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
            cwd=Path(__file__).parents[1],
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
