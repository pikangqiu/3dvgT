#!/usr/bin/env python3
"""Check that local paper/code references are available."""

from __future__ import annotations

from vggt_project.references import collect_reference_status


def main() -> int:
    statuses = collect_reference_status()
    for status in statuses:
        marker = "OK" if status.exists else "MISSING"
        print(f"[{marker}] {status.name}: {status.path} ({status.details})")
    return 0 if all(status.exists for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())

