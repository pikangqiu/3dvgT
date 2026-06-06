"""Adapter template for wiring G3T/VGGT-style heads into the project trainer.

This module is intentionally importable through `runtime.model.adapter_module_path`.
It keeps the same prediction contract as the scaffold model while leaving a
focused replacement point for a real G3T/VGGT backbone.
"""

from __future__ import annotations


def build_local_reference_adapter(
    *,
    reference_root,
    reference_model: str = "g3t",
    point_count: int = 128,
    model_kwargs: dict | None = None,
):
    """Instantiate a local refs/g3t G3T or VGGT model and wrap it for project training."""

    import importlib
    import sys
    from pathlib import Path

    root = Path(reference_root)
    if not root.exists():
        raise FileNotFoundError(f"reference_root does not exist: {root}")
    root_text = str(root.resolve())
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    module_snapshot = _snapshot_vggt_modules(sys.modules)
    _clear_vggt_modules(sys.modules)
    try:
        model_name = reference_model.lower()
        if model_name == "g3t":
            module = importlib.import_module("vggt.models.g3t")
            reference_cls = getattr(module, "G3T")
        elif model_name == "vggt":
            module = importlib.import_module("vggt.models.vggt")
            reference_cls = getattr(module, "VGGT")
        else:
            raise ValueError("reference_model must be 'g3t' or 'vggt'")
        reference = reference_cls(**(model_kwargs or {}))
        return G3TVGGTReferenceAdapter(
            reference,
            point_count=point_count,
            reference_root=root,
            reference_model_name=model_name,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"Could not import {reference_model} from {root}; install refs/g3t requirements first"
        ) from error
    finally:
        _clear_vggt_modules(sys.modules)
        sys.modules.update(module_snapshot)
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def map_reference_prediction_to_contract(reference_prediction: dict, batch: dict, *, point_count: int = 128) -> dict:
    """Map G3T/VGGT-style dense outputs into the project reconstruction contract."""

    import torch

    bev = batch["bev_features"]
    batch_size = bev.shape[0]
    output_size = bev.shape[-2:]
    camera_count = _camera_count(batch)

    camera_depths = _camera_depths_from_reference(reference_prediction, batch_size, camera_count, output_size, bev)
    camera_pointmaps = _camera_pointmaps_from_reference(
        reference_prediction,
        batch_size,
        camera_count,
        point_count,
        bev,
    )
    camera_local_poses = _camera_local_poses_from_reference(
        reference_prediction,
        batch_size,
        camera_count,
        bev,
    )
    return {
        "gravity_aligned_pointmap": _scene_pointmap_from_camera_pointmaps(camera_pointmaps, point_count),
        "depth": camera_depths.mean(dim=1),
        "camera_depths": camera_depths,
        "camera_pointmaps": camera_pointmaps,
        "local_camera_to_gravity_pose": _scene_pose_from_camera_poses(camera_local_poses),
        "camera_local_camera_to_gravity_poses": camera_local_poses,
        "relative_yaw_translation": _relative_pose_from_reference(reference_prediction, batch_size, bev),
    }


def build_model(
    *,
    point_count: int = 128,
    bev_channels: int = 8,
    satellite_channels: int = 3,
    latent_dim: int = 128,
    use_reference_adapter: bool = False,
    reference_root=None,
    reference_model: str = "g3t",
):
    """Build a trainable adapter that satisfies the reconstruction contract."""

    if use_reference_adapter:
        if reference_root is None:
            raise ValueError("reference_root is required when use_reference_adapter is true")
        return build_local_reference_adapter(
            reference_root=reference_root,
            reference_model=reference_model,
            point_count=point_count,
        )

    import torch
    from torch import nn

    from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

    class G3TVGGTAdapterTemplate(nn.Module):
        """Thin trainable bridge until concrete G3T/VGGT heads are connected."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = SatelliteBEVG3TScaffold.build(
                bev_channels=bev_channels,
                satellite_channels=satellite_channels,
                latent_dim=latent_dim,
                point_count=point_count,
            )

        def forward(self, batch: dict) -> dict:
            return self.backbone(batch)

        def freeze_backbone(self) -> None:
            # Keep camera-specific prediction heads trainable for adapter smoke tests.
            for name, parameter in self.backbone.named_parameters():
                if name.startswith(("bev_encoder.", "satellite_encoder.", "camera_encoder.", "fusion.")):
                    parameter.requires_grad = False

        @property
        def adapter_status(self) -> dict[str, str]:
            return {
                "status": "template",
                "backbone": "SatelliteBEVG3TScaffold",
                "reference_output_mapping": "available",
                "next_step": "instantiate concrete G3T/VGGT modules and route outputs through G3TVGGTReferenceAdapter",
            }

    return G3TVGGTAdapterTemplate()


def _torch_nn_module_base():
    import torch

    return torch.nn.Module


def _snapshot_vggt_modules(modules: dict) -> dict:
    return {name: module for name, module in modules.items() if _is_vggt_module_name(name)}


def _clear_vggt_modules(modules: dict) -> None:
    for name in tuple(modules):
        if _is_vggt_module_name(name):
            del modules[name]


def _is_vggt_module_name(name: str) -> bool:
    return name == "vggt" or name.startswith("vggt.")


class G3TVGGTReferenceAdapter(_torch_nn_module_base()):
    """Wrap a concrete G3T/VGGT module and expose the project prediction contract."""

    def __init__(
        self,
        reference_model,
        *,
        point_count: int = 128,
        reference_root=None,
        reference_model_name: str = "custom",
    ) -> None:
        import torch

        super().__init__()
        if not isinstance(reference_model, torch.nn.Module):
            raise TypeError("reference_model must be a torch.nn.Module")
        self.reference_model = reference_model
        self.point_count = point_count
        self.reference_root = reference_root
        self.reference_model_name = reference_model_name

    def forward(self, batch: dict) -> dict:
        reference_prediction = self.reference_model(batch["camera_images"])
        return map_reference_prediction_to_contract(
            reference_prediction,
            batch,
            point_count=self.point_count,
        )

    @property
    def adapter_status(self) -> dict[str, str]:
        return {
            "status": "reference",
            "reference_model": str(self.reference_model_name),
            "reference_root": str(self.reference_root) if self.reference_root is not None else "",
            "reference_output_mapping": "available",
        }


def _camera_count(batch: dict) -> int:
    camera_images = batch.get("camera_images")
    if camera_images is None or camera_images.numel() == 0:
        return 1
    return int(camera_images.shape[1])


def _camera_depths_from_reference(reference_prediction, batch_size, camera_count, output_size, reference_tensor):
    import torch
    from torch.nn import functional as F

    depth = reference_prediction.get("depth")
    if depth is None:
        return torch.zeros(
            batch_size,
            camera_count,
            1,
            *output_size,
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )
    if depth.ndim == 5 and depth.shape[-1] == 1:
        camera_depths = depth.permute(0, 1, 4, 2, 3).contiguous()
    elif depth.ndim == 4:
        camera_depths = depth.unsqueeze(2)
    else:
        raise ValueError(f"unsupported reference depth shape: {tuple(depth.shape)}")
    if camera_depths.shape[-2:] != output_size:
        flat = camera_depths.reshape(camera_depths.shape[0] * camera_depths.shape[1], 1, *camera_depths.shape[-2:])
        flat = F.interpolate(flat, size=output_size, mode="bilinear", align_corners=False)
        camera_depths = flat.reshape(batch_size, camera_depths.shape[1], 1, *output_size)
    return camera_depths[:, :camera_count]


def _camera_pointmaps_from_reference(reference_prediction, batch_size, camera_count, point_count, reference_tensor):
    import torch

    world_points = reference_prediction.get("world_points")
    if world_points is None:
        return torch.zeros(batch_size, camera_count, point_count, 3, dtype=reference_tensor.dtype)
    if world_points.ndim != 5 or world_points.shape[-1] != 3:
        raise ValueError(f"unsupported reference world_points shape: {tuple(world_points.shape)}")
    pointmaps = world_points.reshape(world_points.shape[0], world_points.shape[1], -1, 3)
    pointmaps = pointmaps[:, :camera_count]
    if pointmaps.shape[2] >= point_count:
        return pointmaps[:, :, :point_count, :].contiguous()
    padded = torch.zeros(batch_size, camera_count, point_count, 3, dtype=world_points.dtype, device=world_points.device)
    padded[:, :, : pointmaps.shape[2], :] = pointmaps
    return padded


def _camera_local_poses_from_reference(reference_prediction, batch_size, camera_count, reference_tensor):
    local_pose_enc = reference_prediction.get("local_pose_enc")
    if local_pose_enc is None:
        local_pose_enc = reference_prediction.get("pose_enc")
    if local_pose_enc is None:
        return _identity_camera_poses(batch_size, camera_count, reference_tensor)
    if local_pose_enc.ndim != 3 or local_pose_enc.shape[-1] < 4:
        raise ValueError(f"unsupported reference pose encoding shape: {tuple(local_pose_enc.shape)}")
    poses = _normalize_quaternion(local_pose_enc[:, :camera_count, :4])
    if poses.shape[1] == camera_count:
        return poses
    fallback = _identity_camera_poses(batch_size, camera_count, reference_tensor)
    fallback[:, : poses.shape[1], :] = poses
    return fallback


def _relative_pose_from_reference(reference_prediction, batch_size, reference_tensor):
    import torch

    global_pose_enc = reference_prediction.get("global_pose_enc")
    if global_pose_enc is None or global_pose_enc.ndim != 3 or global_pose_enc.shape[-1] < 4:
        return torch.zeros(batch_size, 4, dtype=reference_tensor.dtype, device=reference_tensor.device)
    return global_pose_enc[:, 0, :4].to(dtype=reference_tensor.dtype)


def _scene_pointmap_from_camera_pointmaps(camera_pointmaps, point_count):
    flattened = camera_pointmaps.reshape(camera_pointmaps.shape[0], -1, 3)
    if flattened.shape[1] >= point_count:
        return flattened[:, :point_count, :].contiguous()
    return flattened


def _scene_pose_from_camera_poses(camera_poses):
    return _normalize_quaternion(camera_poses.mean(dim=1))


def _identity_camera_poses(batch_size, camera_count, reference_tensor):
    import torch

    poses = torch.zeros(batch_size, camera_count, 4, dtype=reference_tensor.dtype, device=reference_tensor.device)
    poses[:, :, 0] = 1.0
    return poses


def _normalize_quaternion(quaternion):
    norm = quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normalized = quaternion / norm
    return normalized
