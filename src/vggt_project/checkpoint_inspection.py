"""Checkpoint structure inspection helpers for downloaded model weights."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


STATE_DICT_KEYS = ("model", "state_dict", "model_state_dict")
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin")


@dataclass(frozen=True)
class CheckpointSummary:
    container_key: str
    tensor_count: int
    prefix_counts: dict[str, int]
    sample_keys: tuple[str, ...]
    tensor_shapes: dict[str, tuple[int, ...]]
    tensor_dtypes: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def summarize_checkpoint(checkpoint: Mapping[str, Any], *, sample_limit: int = 20) -> CheckpointSummary:
    """Summarize a raw or nested checkpoint mapping without mutating it."""

    container_key, state = _extract_state_mapping(checkpoint)
    keys = tuple(sorted(str(key) for key in state.keys()))
    prefix_counts = Counter(_prefix(key) for key in keys)
    sample_keys = keys[:sample_limit]
    return CheckpointSummary(
        container_key=container_key,
        tensor_count=len(keys),
        prefix_counts=dict(sorted(prefix_counts.items())),
        sample_keys=sample_keys,
        tensor_shapes={key: _shape_tuple(state[key]) for key in sample_keys},
        tensor_dtypes={key: _dtype_text(state[key]) for key in sample_keys},
    )


def load_checkpoint_summary(path: Path, *, sample_limit: int = 20) -> CheckpointSummary:
    """Load a torch checkpoint on CPU and summarize its state-dict structure."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("torch is required to inspect checkpoint files; install the training environment") from error

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must load to a mapping: {path}")
    return summarize_checkpoint(checkpoint, sample_limit=sample_limit)


def find_checkpoint_candidates(path: Path) -> tuple[Path, ...]:
    """Return likely PyTorch checkpoint files from a file or directory path."""

    if path.is_file():
        return (path,) if path.suffix.lower() in CHECKPOINT_SUFFIXES else ()
    if not path.is_dir():
        return ()
    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    return tuple(sorted(candidates, key=lambda candidate: str(candidate.relative_to(path))))


def format_checkpoint_summary(summary: CheckpointSummary) -> str:
    """Render checkpoint summary for CLI use."""

    lines = [
        f"container_key: {summary.container_key}",
        f"tensor_count: {summary.tensor_count}",
        "prefix_counts:",
    ]
    if summary.prefix_counts:
        lines.extend(f"- {prefix}: {count}" for prefix, count in summary.prefix_counts.items())
    else:
        lines.append("- none")
    lines.append("sample_tensors:")
    if not summary.sample_keys:
        lines.append("- none")
    for key in summary.sample_keys:
        lines.append(f"- {key}: shape={summary.tensor_shapes[key]} dtype={summary.tensor_dtypes[key]}")
    return "\n".join(lines)


def _extract_state_mapping(checkpoint: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return key, value
    return "raw", checkpoint


def _prefix(key: str) -> str:
    return key.split(".", 1)[0] if key else ""


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    try:
        return tuple(int(dimension) for dimension in shape)
    except TypeError:
        return ()


def _dtype_text(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"
