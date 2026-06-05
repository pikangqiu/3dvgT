"""Split JSONL manifests into train/eval sets without leaking scenes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestSplitReport:
    manifest_path: Path
    train_output_path: Path
    eval_output_path: Path
    sample_count: int
    train_sample_count: int
    eval_sample_count: int
    train_scene_count: int
    eval_scene_count: int


def split_manifest_by_scene(
    manifest_path: Path,
    *,
    train_output_path: Path,
    eval_output_path: Path,
    eval_fraction: float = 0.2,
    seed: str = "0",
    eval_scene_tokens: Iterable[str] | None = None,
) -> ManifestSplitReport:
    """Split records by scene token so no scene appears in both outputs."""

    records = _read_jsonl_records(manifest_path)
    scenes = sorted({str(record["scene_token"]) for record in records})
    if not scenes:
        raise ValueError(f"manifest has no records: {manifest_path}")

    eval_scenes = (
        set(eval_scene_tokens)
        if eval_scene_tokens is not None
        else _choose_eval_scenes(scenes, eval_fraction=eval_fraction, seed=seed)
    )
    unknown_scenes = eval_scenes - set(scenes)
    if unknown_scenes:
        raise ValueError(f"eval_scene_tokens not found in manifest: {sorted(unknown_scenes)}")

    train_records = [record for record in records if record["scene_token"] not in eval_scenes]
    eval_records = [record for record in records if record["scene_token"] in eval_scenes]
    if not train_records:
        raise ValueError("train split is empty; reduce eval_fraction or eval_scene_tokens")
    if not eval_records:
        raise ValueError("eval split is empty; increase eval_fraction or set eval_scene_tokens")

    _write_jsonl_records(train_records, train_output_path)
    _write_jsonl_records(eval_records, eval_output_path)
    return ManifestSplitReport(
        manifest_path=manifest_path,
        train_output_path=train_output_path,
        eval_output_path=eval_output_path,
        sample_count=len(records),
        train_sample_count=len(train_records),
        eval_sample_count=len(eval_records),
        train_scene_count=len({record["scene_token"] for record in train_records}),
        eval_scene_count=len({record["scene_token"] for record in eval_records}),
    )


def _choose_eval_scenes(scenes: list[str], *, eval_fraction: float, seed: str) -> set[str]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    eval_count = max(1, round(len(scenes) * eval_fraction))
    eval_count = min(eval_count, len(scenes) - 1)
    ranked = sorted(scenes, key=lambda scene: _stable_hash(f"{seed}:{scene}"))
    return set(ranked[:eval_count])


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _read_jsonl_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
