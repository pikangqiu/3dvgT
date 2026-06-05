"""Tensor dataset backed by a validated JSONL manifest.

This dataset is a real-file smoke path: it reads camera and satellite images from
disk and converts them to tensors shaped for the current scaffold model. It does
not provide real 3D supervision yet; targets are zero placeholders until depth,
pointmap, pose, or occupancy labels are connected.
"""

from __future__ import annotations

from pathlib import Path

from vggt_project.data.manifest import load_manifest


class ManifestTensorDataset:
    """Read manifest image paths into scaffold training batches."""

    def __init__(self, manifest_path: Path, image_size: int = 32, point_count: int = 128) -> None:
        self.samples = load_manifest(manifest_path)
        self.image_size = image_size
        self.point_count = point_count

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

        return {
            "bev_features": bev_features,
            "satellite_patch": satellite_patch,
            "target_pointmap": torch.zeros(self.point_count, 3, dtype=torch.float32),
            "target_depth": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32),
            "target_local_camera_to_gravity_pose": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "target_relative_yaw_translation": torch.zeros(4, dtype=torch.float32),
            "valid_area_mask": torch.ones(1, self.image_size, self.image_size, dtype=torch.float32),
            "sample_token": sample.token,
        }


def _load_rgb_tensor(path: Path, image_size: int):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _expand_channels(tensor, channels: int):
    repeat_count = (channels + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat(repeat_count, 1, 1)[:channels]

