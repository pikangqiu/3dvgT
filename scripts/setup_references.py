#!/usr/bin/env python3
"""Clone external reference repositories used by the project."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.reference_setup import setup_reference_repositories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plans = setup_reference_repositories(root=args.root, dry_run=args.dry_run)
    for plan in plans:
        marker = "exists" if plan.exists else "missing"
        print(f"[{marker}] {plan.spec.name}: {plan.path}")
        print(f"  url: {plan.spec.url}")
        print(f"  purpose: {plan.spec.purpose}")
        if not plan.exists:
            print(f"  command: {' '.join(plan.command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
