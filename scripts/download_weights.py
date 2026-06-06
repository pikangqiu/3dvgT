#!/usr/bin/env python3
"""Download model weights used by reference pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="thatbrguy/g3t")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/g3t"))
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional Hugging Face allow pattern; repeat for multiple patterns.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _print_download_plan(args)
    if args.dry_run:
        print("status: dry-run")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        if error.name != "huggingface_hub":
            raise
        print("error: huggingface_hub is required; install requirements.txt or run scripts/setup_env.sh")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
    )
    print(f"Downloaded {args.repo_id} to {path}")
    return 0


def _print_download_plan(args: argparse.Namespace) -> None:
    print(f"repo_id: {args.repo_id}")
    print(f"output_dir: {args.output_dir}")
    print(f"revision: {args.revision or '<default>'}")
    print(f"allow_patterns: {_format_patterns(args.allow_pattern)}")


def _format_patterns(patterns: list[str] | None) -> str:
    if not patterns:
        return "<all files>"
    return ", ".join(patterns)


if __name__ == "__main__":
    raise SystemExit(main())
