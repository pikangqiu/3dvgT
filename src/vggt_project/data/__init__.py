"""Dataset contracts for nuScenes satellite/BEV reconstruction."""

from vggt_project.data.sample import AlignedNuScenesSample, CameraFrame
from vggt_project.data.nuscenes_adapter import (
    NuScenesAdapterConfig,
    NuScenesRootStatus,
    inspect_nuscenes_root,
)
from vggt_project.data.manifest import load_manifest
from vggt_project.data.manifest_validation import (
    ManifestValidationReport,
    MissingManifestPath,
    validate_manifest_paths,
)
from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset
from vggt_project.data.manifest_assets import ManifestAssetReport, materialize_manifest_assets

__all__ = [
    "AlignedNuScenesSample",
    "CameraFrame",
    "NuScenesAdapterConfig",
    "NuScenesRootStatus",
    "inspect_nuscenes_root",
    "load_manifest",
    "ManifestValidationReport",
    "MissingManifestPath",
    "validate_manifest_paths",
    "ManifestTensorDataset",
    "ManifestAssetReport",
    "materialize_manifest_assets",
    "LidarDepthReport",
    "materialize_lidar_depth_manifest",
    "rasterize_camera_depth",
    "LidarPointmapReport",
    "materialize_lidar_pointmap_manifest",
    "lidar_points_to_ego_pointmap",
    "LidarSupervisionReport",
    "materialize_lidar_supervision_manifest",
    "ManifestSplitReport",
    "split_manifest_by_scene",
    "SatelliteRasterConfigReport",
    "validate_satellite_raster_config",
]


def __getattr__(name: str):
    if name in {
        "LidarDepthReport",
        "materialize_lidar_depth_manifest",
        "rasterize_camera_depth",
    }:
        from vggt_project.data import nuscenes_depth

        return getattr(nuscenes_depth, name)
    if name in {
        "LidarPointmapReport",
        "materialize_lidar_pointmap_manifest",
        "lidar_points_to_ego_pointmap",
    }:
        from vggt_project.data import nuscenes_pointmap

        return getattr(nuscenes_pointmap, name)
    if name in {
        "LidarSupervisionReport",
        "materialize_lidar_supervision_manifest",
    }:
        from vggt_project.data import supervision_pipeline

        return getattr(supervision_pipeline, name)
    if name in {
        "ManifestSplitReport",
        "split_manifest_by_scene",
    }:
        from vggt_project.data import manifest_split

        return getattr(manifest_split, name)
    if name in {
        "SatelliteRasterConfigReport",
        "validate_satellite_raster_config",
    }:
        from vggt_project.data import satellite_crops

        return getattr(satellite_crops, name)
    raise AttributeError(name)
