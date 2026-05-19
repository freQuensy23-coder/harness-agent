import argparse
import asyncio
from pathlib import Path


def main() -> None:
    from harness_agent.app import HarnessApp
    from harness_agent.config import load_config

    parser = argparse.ArgumentParser(prog="harness-agent")
    parser.add_argument("--config", type=Path, default=Path("harness.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("telegram")

    cli = subparsers.add_parser("ask")
    cli.add_argument("text")
    cli.add_argument("--user-id", default="local")
    cli.add_argument("--conversation-id")

    args = parser.parse_args()
    config = load_config(args.config)
    app = HarnessApp(config=config)

    if args.command == "telegram":
        asyncio.run(app.run_telegram())
    if args.command == "ask":
        reply = asyncio.run(
            app.send_cli(
                text=args.text,
                user_id=args.user_id,
                conversation_id=args.conversation_id,
            )
        )
        print(reply)
