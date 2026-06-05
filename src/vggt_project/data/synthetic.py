"""Synthetic data used to verify the training/evaluation plumbing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSpec:
    num_samples: int = 16
    bev_channels: int = 8
    satellite_channels: int = 3
    height: int = 32
    width: int = 32
    point_count: int = 128


def make_synthetic_dataset(spec: SyntheticSpec):
    """Create a tiny deterministic torch dataset.

    Torch is imported lazily so docs/tests that only inspect the scaffold do not
    need a full training environment.
    """

    import torch
    from torch.utils.data import TensorDataset

    generator = torch.Generator().manual_seed(7)
    bev = torch.randn(
        spec.num_samples,
        spec.bev_channels,
        spec.height,
        spec.width,
        generator=generator,
    )
    satellite = torch.randn(
        spec.num_samples,
        spec.satellite_channels,
        spec.height,
        spec.width,
        generator=generator,
    )
    pointmap = torch.randn(spec.num_samples, spec.point_count, 3, generator=generator)
    depth = torch.rand(spec.num_samples, 1, spec.height, spec.width, generator=generator)
    local_pose = torch.randn(spec.num_samples, 4, generator=generator)
    yaw_translation = torch.randn(spec.num_samples, 4, generator=generator)
    valid_mask = torch.ones(spec.num_samples, 1, spec.height, spec.width)
    return TensorDataset(bev, satellite, pointmap, depth, local_pose, yaw_translation, valid_mask)


def tensor_tuple_to_batch(tensors: tuple) -> dict:
    """Convert the synthetic tuple into the project batch dictionary."""

    bev, satellite, pointmap, depth, local_pose, yaw_translation, valid_mask = tensors
    return {
        "bev_features": bev,
        "satellite_patch": satellite,
        "target_pointmap": pointmap,
        "target_depth": depth,
        "target_local_camera_to_gravity_pose": local_pose,
        "target_relative_yaw_translation": yaw_translation,
        "valid_area_mask": valid_mask,
    }

