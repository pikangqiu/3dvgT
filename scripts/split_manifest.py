#!/usr/bin/env python3
"""Split a JSONL manifest into train/eval manifests by scene token."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.manifest_split import split_manifest_by_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="0")
    parser.add_argument(
        "--eval-scenes",
        default=None,
        help="Comma-separated scene tokens to place in eval; overrides --eval-fraction.",
    )
    args = parser.parse_args()

    eval_scene_tokens = (
        {token.strip() for token in args.eval_scenes.split(",") if token.strip()}
        if args.eval_scenes
        else None
    )
    try:
        report = split_manifest_by_scene(
            args.manifest,
            train_output_path=args.train_output,
            eval_output_path=args.eval_output,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            eval_scene_tokens=eval_scene_tokens,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"manifest: {report.manifest_path}")
    print(f"train_output: {report.train_output_path}")
    print(f"eval_output: {report.eval_output_path}")
    print(f"samples: {report.sample_count}")
    print(f"train_samples: {report.train_sample_count}")
    print(f"eval_samples: {report.eval_sample_count}")
    print(f"train_scenes: {report.train_scene_count}")
    print(f"eval_scenes: {report.eval_scene_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
