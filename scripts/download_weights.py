#!/usr/bin/env python3
"""Download model weights used by reference pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="thatbrguy/g3t")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/g3t"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.output_dir)
    print(f"Downloaded {args.repo_id} to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

