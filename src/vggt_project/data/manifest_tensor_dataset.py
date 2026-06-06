"""Tensor dataset backed by a validated JSONL manifest.

This dataset is a real-file smoke path: it reads camera and satellite images from
disk and converts them to tensors shaped for the current scaffold model. It can
load depth/mask targets and derive coarse ego-pose targets from manifest metadata.
Pointmap targets can be sample-level or camera-level arrays when preprocessing
writes them into the manifest.
"""

from __future__ import annotations

import math
from pathlib import Path

from vggt_project.data.manifest import load_manifest


class ManifestTensorDataset:
    """Read manifest image paths into scaffold training batches."""

    def __init__(self, manifest_path: Path, image_size: int = 32, point_count: int = 128) -> None:
        self.samples = load_manifest(manifest_path)
        self.image_size = image_size
        self.point_count = point_count
        self.scene_origins = _scene_translation_origins(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        import torch

        sample = self.samples[index]
        camera_tensors = [_load_rgb_tensor(camera.image_path, self.image_size) for camera in sample.cameras]
        camera_stack = torch.stack(camera_tensors, dim=0)
        camera_mean = camera_stack.mean(dim=0)
        bev_features = _expand_channels(camera_mean, channels=8)
        satellite_patch = _load_rgb_tensor(sample.satellite_patch_path, self.image_size)
        target_camera_depths = _load_camera_depth_tensors(sample, self.image_size)
        target_depth = (
            target_camera_depths.mean(dim=0)
            if target_camera_depths.numel() > 0
            else torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        )
        target_camera_pointmaps = _load_camera_pointmap_tensors(sample, self.point_count)
        target_pointmap = (
            _load_pointmap_tensor(sample.pointmap_path, self.point_count)
            if sample.pointmap_path is not None
            else target_camera_pointmaps.mean(dim=0)
            if target_camera_pointmaps.numel() > 0
            else torch.zeros(self.point_count, 3, dtype=torch.float32)
        )
        valid_area_mask = (
            _load_gray_tensor(sample.valid_area_mask_path, self.image_size)
            if sample.valid_area_mask_path is not None
            else torch.ones(1, self.image_size, self.image_size, dtype=torch.float32)
        )

        return {
            "bev_features": bev_features,
            "camera_images": camera_stack,
            "satellite_patch": satellite_patch,
            "target_pointmap": target_pointmap,
            "target_camera_pointmaps": target_camera_pointmaps,
            "target_depth": target_depth,
            "target_camera_depths": target_camera_depths,
            "target_local_camera_to_gravity_pose": _pose_quaternion_tensor(sample),
            "target_camera_local_camera_to_gravity_poses": _camera_pose_quaternion_tensor(sample),
            "target_relative_yaw_translation": _relative_yaw_translation_tensor(
                sample,
                self.scene_origins.get(sample.scene_token),
            ),
            "valid_area_mask": valid_area_mask,
            "sample_token": sample.token,
        }


def _load_rgb_tensor(path: Path, image_size: int):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_gray_tensor(path: Path, image_size: int):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("L").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _load_camera_depth_tensors(sample, image_size: int):
    import torch

    if sample.lidar_depth_paths is not None:
        by_camera = [
            sample.lidar_depth_paths[camera.camera_name]
            for camera in sample.cameras
            if camera.camera_name in sample.lidar_depth_paths
        ]
        if by_camera:
            return torch.stack([_load_gray_tensor(path, image_size) for path in by_camera], dim=0)
    if sample.lidar_depth_path is not None:
        return _load_gray_tensor(sample.lidar_depth_path, image_size).unsqueeze(0)
    return torch.zeros(0, 1, image_size, image_size, dtype=torch.float32)


def _load_camera_pointmap_tensors(sample, point_count: int):
    import torch

    if sample.pointmap_paths is None:
        return torch.zeros(0, point_count, 3, dtype=torch.float32)
    by_camera = [
        sample.pointmap_paths[camera.camera_name]
        for camera in sample.cameras
        if camera.camera_name in sample.pointmap_paths
    ]
    if not by_camera:
        return torch.zeros(0, point_count, 3, dtype=torch.float32)
    return torch.stack([_load_pointmap_tensor(path, point_count) for path in by_camera], dim=0)


def _expand_channels(tensor, channels: int):
    repeat_count = (channels + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat(repeat_count, 1, 1)[:channels]


def _load_pointmap_tensor(path: Path, point_count: int):
    import numpy as np
    import torch

    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "pointmap" not in loaded:
            raise ValueError(f"pointmap npz must contain a 'pointmap' array: {path}")
        array = loaded["pointmap"]
    else:
        array = loaded
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"pointmap must have shape Nx3, got {array.shape}: {path}")

    tensor = torch.from_numpy(array).contiguous()
    if tensor.shape[0] >= point_count:
        return tensor[:point_count]

    padded = torch.zeros(point_count, 3, dtype=torch.float32)
    padded[: tensor.shape[0]] = tensor
    return padded


def _scene_translation_origins(samples) -> dict[str, tuple[float, float, float]]:
    origins: dict[str, tuple[float, float, float]] = {}
    for sample in sorted(samples, key=lambda item: (item.scene_token, item.timestamp_us)):
        if sample.ego_translation is not None and sample.scene_token not in origins:
            origins[sample.scene_token] = sample.ego_translation
    return origins


def _pose_quaternion_tensor(sample):
    import torch

    if sample.ego_rotation is None:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    normalized = _normalize_quaternion(sample.ego_rotation)
    return torch.tensor(normalized, dtype=torch.float32)


def _camera_pose_quaternion_tensor(sample):
    return _pose_quaternion_tensor(sample).unsqueeze(0).repeat(len(sample.cameras), 1)


def _relative_yaw_translation_tensor(sample, origin: tuple[float, float, float] | None):
    import torch

    if sample.ego_rotation is None or sample.ego_translation is None or origin is None:
        return torch.zeros(4, dtype=torch.float32)
    yaw = _yaw_from_quaternion(_normalize_quaternion(sample.ego_rotation))
    translation = tuple(sample.ego_translation[index] - origin[index] for index in range(3))
    return torch.tensor((yaw, *translation), dtype=torch.float32)


def _normalize_quaternion(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quaternion)


def _yaw_from_quaternion(quaternion: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
