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
]
