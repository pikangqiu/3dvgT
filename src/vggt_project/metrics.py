"""Evaluation metrics for reconstruction outputs."""

from __future__ import annotations


def reconstruction_metrics(prediction: dict, batch: dict) -> dict[str, float]:
    """Compute lightweight reconstruction metrics for train/eval logs."""

    import torch

    from vggt_project.losses import depth_abs_error, local_pose_pair

    depth_mae = depth_abs_error(prediction, batch).mean_loss
    predicted_pointmaps, target_pointmaps = _pointmap_metric_pair(prediction, batch)
    pointmap_metrics = scale_aligned_pointmap_metrics(predicted_pointmaps, target_pointmaps)

    pointmap_l1 = (predicted_pointmaps - target_pointmaps).abs().mean()
    predicted_local_pose, target_local_pose = local_pose_pair(prediction, batch)
    gravity_error_deg = quaternion_angular_error_deg(predicted_local_pose, target_local_pose)
    local_pose_l2 = torch.linalg.vector_norm(predicted_local_pose - target_local_pose, dim=1).mean()
    sequence_drift = sequence_translation_drift(
        prediction["relative_yaw_translation"],
        batch["target_relative_yaw_translation"],
    )
    relative_pose_l2 = torch.linalg.vector_norm(
        prediction["relative_yaw_translation"] - batch["target_relative_yaw_translation"],
        dim=1,
    ).mean()

    return {
        "depth_mae": float(depth_mae.detach().cpu()),
        "pointmap_l1": float(pointmap_l1.detach().cpu()),
        "scale_aligned_pointmap_accuracy": float(pointmap_metrics["accuracy"].detach().cpu()),
        "scale_aligned_pointmap_completeness": float(pointmap_metrics["completeness"].detach().cpu()),
        "scale_aligned_pointmap_chamfer": float(pointmap_metrics["chamfer"].detach().cpu()),
        "gravity_error_deg": float(gravity_error_deg.detach().cpu()),
        "local_pose_l2": float(local_pose_l2.detach().cpu()),
        "sequence_translation_drift": float(sequence_drift.detach().cpu()),
        "relative_pose_l2": float(relative_pose_l2.detach().cpu()),
    }


def scale_aligned_pointmap_metrics(predicted, target) -> dict:
    """Compute accuracy/completeness after per-sample isotropic scale alignment."""

    import torch

    predicted_aligned = _scale_align_pointmaps(predicted, target)
    pairwise = torch.cdist(predicted_aligned, target)
    accuracy = pairwise.min(dim=2).values.mean()
    completeness = pairwise.min(dim=1).values.mean()
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": accuracy + completeness,
    }


def _pointmap_metric_pair(prediction: dict, batch: dict):
    target_camera_pointmaps = batch.get("target_camera_pointmaps")
    if target_camera_pointmaps is not None and target_camera_pointmaps.numel() > 0:
        predicted_camera_pointmaps = prediction.get("camera_pointmaps")
        if predicted_camera_pointmaps is not None:
            predicted = predicted_camera_pointmaps
        else:
            predicted = prediction["gravity_aligned_pointmap"].unsqueeze(1).expand_as(
                target_camera_pointmaps
            )
        return _flatten_camera_pointmaps(predicted), _flatten_camera_pointmaps(target_camera_pointmaps)
    return prediction["gravity_aligned_pointmap"], batch["target_pointmap"]


def _flatten_camera_pointmaps(pointmaps):
    return pointmaps.reshape(pointmaps.shape[0] * pointmaps.shape[1], pointmaps.shape[2], pointmaps.shape[3])


def quaternion_angular_error_deg(predicted, target):
    """Return mean quaternion angular error in degrees."""

    import torch
    from torch.nn import functional as F

    predicted = F.normalize(predicted, dim=1)
    target = F.normalize(target, dim=1)
    dot = (predicted * target).sum(dim=1).abs().clamp(max=1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return torch.rad2deg(angle_rad).mean()


def sequence_translation_drift(predicted_relative_pose, target_relative_pose):
    """Compare batch-order scene-relative translation tracks after removing first pose."""

    import torch

    if predicted_relative_pose.shape[0] < 2:
        return torch.zeros((), dtype=predicted_relative_pose.dtype, device=predicted_relative_pose.device)
    predicted_translation = predicted_relative_pose[:, 1:4]
    target_translation = target_relative_pose[:, 1:4]
    predicted_offsets = predicted_translation - predicted_translation[:1]
    target_offsets = target_translation - target_translation[:1]
    return torch.linalg.vector_norm(predicted_offsets - target_offsets, dim=1).mean()


def _scale_align_pointmaps(predicted, target):
    predicted_centered = predicted - predicted.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = (predicted_centered * target_centered).sum(dim=(1, 2), keepdim=True)
    denominator = (predicted_centered * predicted_centered).sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    scale = numerator / denominator
    return predicted_centered * scale + target.mean(dim=1, keepdim=True)
