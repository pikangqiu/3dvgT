#!/usr/bin/env python3
"""Audit whether the repository has the requested training scaffold pieces."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.project_audit import audit_project_files, format_audit_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit_project_files(args.root)
    if args.json:
        print(report.to_json())
    else:
        print(format_audit_report(report))
    return 0 if report.scaffold_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
