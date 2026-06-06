#!/usr/bin/env python3
"""Print the recommended baseline and benchmark protocol."""

from __future__ import annotations

import argparse

from vggt_project.experiment_protocol import build_experiment_protocol, format_experiment_protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    protocol = build_experiment_protocol()
    if args.json:
        print(protocol.to_json())
    else:
        print(format_experiment_protocol(protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
