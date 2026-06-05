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

    point_loss = F.smooth_l1_loss(
        prediction["gravity_aligned_pointmap"],
        batch["target_pointmap"],
    )
    local_pose_loss = F.mse_loss(
        prediction["local_camera_to_gravity_pose"],
        batch["target_local_camera_to_gravity_pose"],
    )
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
