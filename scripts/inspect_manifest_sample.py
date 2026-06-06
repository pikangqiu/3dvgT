#!/usr/bin/env python3
"""Write a visual and JSON preview for one manifest sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_project.data.manifest_preview import build_manifest_sample_preview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manifest-preview"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    preview = build_manifest_sample_preview(
        args.manifest,
        args.output_dir,
        sample_index=args.sample_index,
        tile_size=args.tile_size,
    )
    if args.json:
        print(json.dumps(preview.summary, indent=2, sort_keys=True))
        return 0

    print(f"token: {preview.token}")
    print(f"summary: {preview.summary_path}")
    print(f"contact_sheet: {preview.contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
