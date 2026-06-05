"""Losses for the reconstruction-first scaffold."""

from __future__ import annotations


def reconstruction_losses(prediction: dict, batch: dict) -> dict:
    """Compute weighted losses for the current scaffold outputs."""

    import torch
    from torch.nn import functional as F

    valid = batch.get("valid_area_mask")
    depth_error = prediction["depth"] - batch["target_depth"]
    if valid is not None:
        depth_loss = (depth_error.abs() * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        depth_loss = depth_error.abs().mean()

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


def detach_float_metrics(losses: dict) -> dict[str, float]:
    """Turn scalar tensors into JSON/log friendly floats."""

    return {name: float(value.detach().cpu()) for name, value in losses.items()}

