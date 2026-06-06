#!/usr/bin/env python3
"""Report the current Python/package/device environment for real runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.environment_report import (
    collect_environment_report,
    format_environment_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = collect_environment_report()
    payload = report.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(format_environment_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
