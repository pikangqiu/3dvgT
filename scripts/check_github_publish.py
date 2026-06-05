#!/usr/bin/env python3
"""Check whether GitHub publishing can proceed."""

from __future__ import annotations

from vggt_project.github_publish import format_publish_preflight, run_publish_preflight


def main() -> int:
    report = run_publish_preflight()
    print(format_publish_preflight(report))
    return 0 if report.ready_to_publish else 1


if __name__ == "__main__":
    raise SystemExit(main())
