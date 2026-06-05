"""Project configuration types.

These dataclasses describe the intended research system without pulling in heavy
training dependencies. They are the stable boundary between paper references,
dataset preparation, and model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ReferencePaths:
    """Local paths for external reference material."""

    root: Path = Path(".")
    g3t: Path = Path("refs/g3t")
    look_from_above_notes: Path = Path("refs/look-from-above")
    pseudomaptrainer: Path = Path("refs/look-from-above-components/PseudoMapTrainer")
    maptr: Path = Path("refs/look-from-above-components/MapTR")
    look_from_above_paper: Path = Path("859_Look_from_Above_Satellite_.pdf")

    def resolve(self) -> "ReferencePaths":
        root = self.root.resolve()
        return ReferencePaths(
            root=root,
            g3t=(root / self.g3t).resolve(),
            look_from_above_notes=(root / self.look_from_above_notes).resolve(),
            pseudomaptrainer=(root / self.pseudomaptrainer).resolve(),
            maptr=(root / self.maptr).resolve(),
            look_from_above_paper=(root / self.look_from_above_paper).resolve(),
        )


@dataclass(frozen=True)
class ReconstructionObjective:
    """Loss and metric priorities for the first project target."""

    primary_losses: tuple[str, ...] = (
        "pointmap_reconstruction",
        "depth_reconstruction",
        "pose_consistency",
        "gravity_alignment",
    )
    auxiliary_losses: tuple[str, ...] = (
        "bev_occupancy_optional",
        "vector_map_optional",
        "valid_area_masking",
    )
    primary_metrics: tuple[str, ...] = (
        "pointmap_accuracy",
        "depth_error",
        "camera_pose_error",
        "gravity_alignment_error",
        "long_sequence_alignment_drift",
    )


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project intent."""

    name: str = "vggt_satellite_reconstruction"
    deadline: str = "2026-06-25"
    task_framing: str = "3d_reconstruction_first"
    references: ReferencePaths = field(default_factory=ReferencePaths)
    objective: ReconstructionObjective = field(default_factory=ReconstructionObjective)
