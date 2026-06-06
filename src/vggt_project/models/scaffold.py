"""Minimal trainable model scaffold for satellite/BEV-conditioned reconstruction."""

from __future__ import annotations


class SatelliteBEVG3TScaffold:
    """Factory wrapper that delays torch imports until training time."""

    @staticmethod
    def build(
        bev_channels: int = 8,
        satellite_channels: int = 3,
        latent_dim: int = 128,
        point_count: int = 128,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bev_encoder = nn.Sequential(
                    nn.Conv2d(bev_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.satellite_encoder = nn.Sequential(
                    nn.Conv2d(satellite_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.camera_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(128, latent_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(inplace=True),
                )
                self.point_head = nn.Linear(latent_dim, point_count * 3)
                self.depth_head = nn.Linear(latent_dim, 32 * 32)
                self.camera_depth_head = nn.Linear(latent_dim + 64, 32 * 32)
                self.camera_point_head = nn.Linear(latent_dim + 64, point_count * 3)
                self.camera_local_pose_head = nn.Linear(latent_dim + 64, 4)
                self.local_pose_head = nn.Linear(latent_dim, 4)
                self.relative_pose_head = nn.Linear(latent_dim, 4)

            def forward(self, batch: dict) -> dict:
                bev = batch["bev_features"]
                satellite = batch["satellite_patch"]
                bev_latent = self.bev_encoder(bev).flatten(1)
                satellite_latent = self.satellite_encoder(satellite).flatten(1)
                scene_latent = self.fusion(torch.cat([bev_latent, satellite_latent], dim=1))

                depth = self.depth_head(scene_latent).view(bev.shape[0], 1, 32, 32)
                if depth.shape[-2:] != bev.shape[-2:]:
                    depth = F.interpolate(depth, size=bev.shape[-2:], mode="bilinear", align_corners=False)
                camera_outputs = self._predict_camera_outputs(batch, scene_latent, bev.shape[-2:])

                local_pose = self.local_pose_head(scene_latent)
                local_pose = F.normalize(local_pose, dim=1)

                return {
                    "scene_latent": scene_latent,
                    "gravity_aligned_pointmap": self.point_head(scene_latent).view(
                        bev.shape[0], point_count, 3
                    ),
                    "depth": depth,
                    "camera_depths": camera_outputs["camera_depths"],
                    "camera_pointmaps": camera_outputs["camera_pointmaps"],
                    "camera_local_camera_to_gravity_poses": camera_outputs[
                        "camera_local_camera_to_gravity_poses"
                    ],
                    "local_camera_to_gravity_pose": local_pose,
                    "relative_yaw_translation": self.relative_pose_head(scene_latent),
                }

            def _predict_camera_outputs(self, batch: dict, scene_latent, output_size: tuple[int, int]):
                camera_images = batch.get("camera_images")
                if camera_images is None or camera_images.numel() == 0:
                    return {
                        "camera_depths": None,
                        "camera_pointmaps": None,
                        "camera_local_camera_to_gravity_poses": None,
                    }

                batch_size, camera_count, channels, height, width = camera_images.shape
                camera_flat = camera_images.reshape(batch_size * camera_count, channels, height, width)
                camera_latent = self.camera_encoder(camera_flat).flatten(1)
                scene_repeated = scene_latent.unsqueeze(1).expand(
                    batch_size,
                    camera_count,
                    scene_latent.shape[1],
                )
                fused = torch.cat(
                    [scene_repeated.reshape(batch_size * camera_count, -1), camera_latent],
                    dim=1,
                )
                camera_depths = self.camera_depth_head(fused).view(batch_size * camera_count, 1, 32, 32)
                if camera_depths.shape[-2:] != output_size:
                    camera_depths = F.interpolate(
                        camera_depths,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                camera_pointmaps = self.camera_point_head(fused).view(
                    batch_size,
                    camera_count,
                    point_count,
                    3,
                )
                camera_poses = F.normalize(self.camera_local_pose_head(fused), dim=1).view(
                    batch_size,
                    camera_count,
                    4,
                )
                return {
                    "camera_depths": camera_depths.reshape(batch_size, camera_count, 1, *output_size),
                    "camera_pointmaps": camera_pointmaps,
                    "camera_local_camera_to_gravity_poses": camera_poses,
                }

        return _Model()
