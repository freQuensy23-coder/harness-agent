#!/usr/bin/env python3

import argparse
import asyncio
from pathlib import Path

from harness_agent.app import HarnessApp
from harness_agent.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="cli.py")
    parser.add_argument("--config", type=Path, default=Path("harness.yaml"))
    parser.add_argument("--send", required=True)
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--conversation_id")
    args = parser.parse_args()

    app = HarnessApp(config=load_config(args.config))
    reply = asyncio.run(
        app.send_cli(
            text=args.send,
            user_id=args.user_id,
            conversation_id=args.conversation_id,
        )
    )
    print(reply)


if __name__ == "__main__":
    main()
