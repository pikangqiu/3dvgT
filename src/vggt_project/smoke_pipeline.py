"""End-to-end smoke pipelines for local train/eval verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vggt_project.evaluation import evaluate_manifest_smoke
from vggt_project.training import train_manifest_smoke


@dataclass(frozen=True)
class ManifestSmokePipelineReport:
    output_dir: Path
    manifest_path: Path
    checkpoint_path: Path
    train_metrics: dict[str, float | str]
    eval_metrics: dict[str, float]


def run_manifest_smoke_pipeline(
    output_dir: Path,
    *,
    epochs: int = 1,
    image_size: int = 32,
    point_count: int = 128,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> ManifestSmokePipelineReport:
    """Create a toy manifest, train from real files, and evaluate the checkpoint."""

    manifest_path = _write_toy_manifest(output_dir)
    checkpoint_dir = output_dir / "outputs" / "manifest-smoke"
    train_metrics = train_manifest_smoke(
        manifest_path=manifest_path,
        output_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        point_count=point_count,
        device=device,
    )
    checkpoint_path = checkpoint_dir / "manifest_smoke_scaffold.pt"
    eval_metrics = evaluate_manifest_smoke(
        checkpoint=checkpoint_path,
        manifest_path=manifest_path,
        batch_size=batch_size,
        image_size=image_size,
        point_count=point_count,
        device=device,
    )

    return ManifestSmokePipelineReport(
        output_dir=output_dir,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )


def _write_toy_manifest(output_dir: Path) -> Path:
    from PIL import Image

    data_dir = output_dir / "toy_manifest"
    camera_dir = data_dir / "samples" / "CAM_FRONT"
    satellite_dir = data_dir / "satellite"
    target_dir = data_dir / "targets"
    camera_dir.mkdir(parents=True, exist_ok=True)
    satellite_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (16, 16), color=(220, 40, 40)).save(camera_dir / "sample-1.png")
    Image.new("RGB", (16, 16), color=(40, 180, 60)).save(satellite_dir / "sample-1.png")
    Image.new("L", (16, 16), color=128).save(target_dir / "depth-1.png")
    Image.new("L", (16, 16), color=255).save(target_dir / "mask-1.png")

    manifest_path = data_dir / "samples.jsonl"
    manifest_path.write_text(
        '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
        '"camera_paths":["samples/CAM_FRONT/sample-1.png"],'
        '"satellite_patch_path":"satellite/sample-1.png",'
        '"lidar_depth_path":"targets/depth-1.png",'
        '"valid_area_mask_path":"targets/mask-1.png",'
        '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
        '"satellite_frame":"satellite"}\n',
        encoding="utf-8",
    )
    return manifest_path
