"""Interfaces for future satellite/BEV-conditioned G3T modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ReconstructionBatch:
    """Batch payload passed from dataset code into the model."""

    multi_view_images: Any
    camera_calibration: Any
    ego_poses: Any
    bev_features: Any
    satellite_patch: Any
    valid_area_mask: Any | None = None
    lidar_or_depth_target: Any | None = None
    vector_map_target: Any | None = None


@dataclass(frozen=True)
class FusionEncoderOutput:
    """Shared latent after BEV/satellite fusion."""

    scene_latent: Any
    bev_latent: Any
    satellite_latent: Any
    frame: str


@dataclass(frozen=True)
class ReconstructionPrediction:
    """Outputs expected from G3T-style reconstruction heads."""

    gravity_aligned_pointmap: Any
    depth: Any
    local_camera_to_gravity_pose: Any
    relative_yaw_translation: Any
    bev_occupancy: Any | None = None
    vector_map: Any | None = None


class ReconstructionSystem(Protocol):
    """Minimal protocol implemented by concrete model systems."""

    def encode(self, batch: ReconstructionBatch) -> FusionEncoderOutput:
        """Encode multi-view/BEV/satellite inputs into a shared scene latent."""

    def reconstruct(self, encoded: FusionEncoderOutput) -> ReconstructionPrediction:
        """Predict gravity-aligned 3D reconstruction outputs."""

