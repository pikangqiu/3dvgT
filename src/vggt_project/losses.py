"""Losses for the reconstruction-first scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthAbsError:
    abs_error: object
    mean_loss: object


def reconstruction_losses(prediction: dict, batch: dict) -> dict:
    """Compute weighted losses for the current scaffold outputs."""

    from torch.nn import functional as F

    depth_loss = depth_abs_error(prediction, batch).mean_loss

    point_loss = pointmap_loss(prediction, batch)
    predicted_local_pose, target_local_pose = local_pose_pair(prediction, batch)
    local_pose_loss = F.mse_loss(predicted_local_pose, target_local_pose)
    relative_pose_loss = F.mse_loss(
        prediction["relative_yaw_translation"],
        batch["target_relative_yaw_translation"],
    )
    total = point_loss + depth_loss + 0.1 * local_pose_loss + 0.1 * relative_pose_loss
    return {
        "loss": total,
        "pointmap": point_loss.detach(),
        "depth": depth_loss.detach(),
        "local_pose": local_pose_loss.detach(),
        "relative_pose": relative_pose_loss.detach(),
    }


def pointmap_loss(prediction: dict, batch: dict):
    """Prefer camera-level pointmap supervision when present."""

    from torch.nn import functional as F

    target_camera_pointmaps = batch.get("target_camera_pointmaps")
    if target_camera_pointmaps is not None and target_camera_pointmaps.numel() > 0:
        predicted_camera_pointmaps = prediction.get("camera_pointmaps")
        if predicted_camera_pointmaps is not None:
            pointmap_prediction = predicted_camera_pointmaps
        else:
            pointmap_prediction = prediction["gravity_aligned_pointmap"].unsqueeze(1).expand_as(
                target_camera_pointmaps
            )
        return F.smooth_l1_loss(pointmap_prediction, target_camera_pointmaps)

    return F.smooth_l1_loss(
        prediction["gravity_aligned_pointmap"],
        batch["target_pointmap"],
    )


def local_pose_pair(prediction: dict, batch: dict):
    """Prefer camera-level local pose supervision when present."""

    target_camera_poses = batch.get("target_camera_local_camera_to_gravity_poses")
    if target_camera_poses is not None and target_camera_poses.numel() > 0:
        predicted_camera_poses = prediction.get("camera_local_camera_to_gravity_poses")
        if predicted_camera_poses is not None:
            return _flatten_camera_poses(predicted_camera_poses), _flatten_camera_poses(target_camera_poses)
        expanded = prediction["local_camera_to_gravity_pose"].unsqueeze(1).expand_as(target_camera_poses)
        return _flatten_camera_poses(expanded), _flatten_camera_poses(target_camera_poses)

    return prediction["local_camera_to_gravity_pose"], batch["target_local_camera_to_gravity_pose"]


def _flatten_camera_poses(poses):
    return poses.reshape(poses.shape[0] * poses.shape[1], poses.shape[2])


def depth_abs_error(prediction: dict, batch: dict) -> DepthAbsError:
    valid = batch.get("valid_area_mask")
    target_camera_depths = batch.get("target_camera_depths")
    if target_camera_depths is not None and target_camera_depths.numel() > 0:
        predicted_camera_depths = prediction.get("camera_depths")
        if predicted_camera_depths is not None:
            depth_prediction = predicted_camera_depths
        else:
            depth_prediction = prediction["depth"].unsqueeze(1)
        depth_error = depth_prediction - target_camera_depths
        if valid is not None:
            mask = valid.unsqueeze(1).expand_as(depth_error)
            mean_loss = (depth_error.abs() * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            mean_loss = depth_error.abs().mean()
        return DepthAbsError(abs_error=depth_error.abs(), mean_loss=mean_loss)

    depth_error = prediction["depth"] - batch["target_depth"]
    if valid is not None:
        mean_loss = (depth_error.abs() * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        mean_loss = depth_error.abs().mean()
    return DepthAbsError(abs_error=depth_error.abs(), mean_loss=mean_loss)


def detach_float_metrics(losses: dict) -> dict[str, float]:
    """Turn scalar tensors into JSON/log friendly floats."""

    return {name: float(value.detach().cpu()) for name, value in losses.items()}
