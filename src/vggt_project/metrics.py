"""Evaluation metrics for reconstruction outputs."""

from __future__ import annotations


def reconstruction_metrics(prediction: dict, batch: dict) -> dict[str, float]:
    """Compute lightweight reconstruction metrics for train/eval logs."""

    import torch

    valid = batch.get("valid_area_mask")
    depth_abs = (prediction["depth"] - batch["target_depth"]).abs()
    if valid is not None:
        depth_mae = (depth_abs * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        depth_mae = depth_abs.mean()

    pointmap_l1 = (
        prediction["gravity_aligned_pointmap"] - batch["target_pointmap"]
    ).abs().mean()
    local_pose_l2 = torch.linalg.vector_norm(
        prediction["local_camera_to_gravity_pose"]
        - batch["target_local_camera_to_gravity_pose"],
        dim=1,
    ).mean()
    relative_pose_l2 = torch.linalg.vector_norm(
        prediction["relative_yaw_translation"] - batch["target_relative_yaw_translation"],
        dim=1,
    ).mean()

    return {
        "depth_mae": float(depth_mae.detach().cpu()),
        "pointmap_l1": float(pointmap_l1.detach().cpu()),
        "local_pose_l2": float(local_pose_l2.detach().cpu()),
        "relative_pose_l2": float(relative_pose_l2.detach().cpu()),
    }

