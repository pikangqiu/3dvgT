"""Experiment protocol for baselines, benchmarks, and comparison tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProtocolMetric:
    name: str
    role: str
    lower_is_better: bool
    note: str


@dataclass(frozen=True)
class ProtocolBaseline:
    name: str
    family: str
    satellite_conditioning: str
    expected_artifact: str
    table: str
    experiment_phase: str


@dataclass(frozen=True)
class ProtocolBenchmark:
    name: str
    role: str
    source_url: str
    reason: str
    experiment_phase: str


@dataclass(frozen=True)
class ExperimentProtocol:
    primary_metrics: tuple[ProtocolMetric, ...]
    auxiliary_metrics: tuple[ProtocolMetric, ...]
    baselines: tuple[ProtocolBaseline, ...]
    benchmarks: tuple[ProtocolBenchmark, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def build_experiment_protocol() -> ExperimentProtocol:
    """Return the current recommended protocol for first paper experiments."""

    return ExperimentProtocol(
        primary_metrics=(
            ProtocolMetric(
                name="depth_mae",
                role="camera geometry",
                lower_is_better=True,
                note="Primary depth reconstruction error from current scaffold metrics.",
            ),
            ProtocolMetric(
                name="scale_aligned_pointmap_chamfer",
                role="3D reconstruction",
                lower_is_better=True,
                note="Scale-aligned pointmap accuracy+completeness for VGGT/G3T-style outputs.",
            ),
            ProtocolMetric(
                name="gravity_error_deg",
                role="gravity alignment",
                lower_is_better=True,
                note="Quaternion angular error for camera-to-gravity orientation.",
            ),
            ProtocolMetric(
                name="sequence_translation_drift",
                role="temporal consistency",
                lower_is_better=True,
                note="Scene-relative translation drift across ordered samples.",
            ),
        ),
        auxiliary_metrics=(
            ProtocolMetric(
                name="bev_occupancy_iou",
                role="driving scene geometry",
                lower_is_better=False,
                note="Implemented auxiliary metric for binary LiDAR-derived occupancy proxy targets.",
            ),
            ProtocolMetric(
                name="occupancy_miou",
                role="semantic driving scene geometry",
                lower_is_better=False,
                note="Implemented for exported Occ3D/OpenOccupancy/OpenScene-style occupancy arrays; requires verified label export and class mapping.",
            ),
            ProtocolMetric(
                name="vector_map_map",
                role="map auxiliary",
                lower_is_better=False,
                note="Use only for MapTR/PseudoMapTrainer auxiliary map-head comparisons.",
            ),
        ),
        baselines=(
            ProtocolBaseline(
                name="VGGT",
                family="3D geometric foundation model",
                satellite_conditioning="none",
                expected_artifact="camera-only depth/pointmap/pose predictions",
                table="primary reconstruction table",
                experiment_phase="phase1_core",
            ),
            ProtocolBaseline(
                name="G3T",
                family="gravity-aligned 3D reconstruction",
                satellite_conditioning="none",
                expected_artifact="gravity-aligned pointmap and pose predictions",
                table="primary reconstruction table",
                experiment_phase="phase1_core",
            ),
            ProtocolBaseline(
                name="BEV+G3T",
                family="ours ablation",
                satellite_conditioning="bev-only",
                expected_artifact="BEV-conditioned reconstruction outputs",
                table="primary reconstruction table",
                experiment_phase="phase1_core",
            ),
            ProtocolBaseline(
                name="BEV+Satellite+G3T",
                family="ours main",
                satellite_conditioning="bev+satellite",
                expected_artifact="satellite-conditioned reconstruction outputs",
                table="primary reconstruction table",
                experiment_phase="phase1_core",
            ),
            ProtocolBaseline(
                name="Cross3R",
                family="satellite-drone-ground feed-forward 3D reconstruction",
                satellite_conditioning="satellite+optional-uav+ground",
                expected_artifact="cross-view point cloud, 6-DoF camera pose, and on-tile localization",
                table="satellite/cross-view reconstruction table",
                experiment_phase="phase2_external",
            ),
            ProtocolBaseline(
                name="SkyNet",
                family="multi-altitude site modeling",
                satellite_conditioning="satellite+aerial+ground",
                expected_artifact="camera localization and site reconstruction outputs",
                table="satellite/cross-view reconstruction table",
                experiment_phase="phase2_external",
            ),
            ProtocolBaseline(
                name="DrivingForward",
                family="feed-forward driving-scene Gaussian splatting",
                satellite_conditioning="none",
                expected_artifact="nuScenes surround-view 3D Gaussian reconstruction",
                table="driving reconstruction table",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBaseline(
                name="DGGT",
                family="pose-free feed-forward 4D driving reconstruction",
                satellite_conditioning="none",
                expected_artifact="camera pose, depth, dynamic maps, and 3D Gaussian tracking",
                table="driving reconstruction table",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBaseline(
                name="ReconDrive",
                family="VGGT-adapted feed-forward 4D Gaussian splatting",
                satellite_conditioning="none",
                expected_artifact="nuScenes 4DGS reconstruction, NVS, perception-preservation outputs",
                table="driving reconstruction table",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBaseline(
                name="DynamicVGGT",
                family="VGGT-extended dynamic pointmap reconstruction",
                satellite_conditioning="none",
                expected_artifact="current/future pointmaps and dynamic 3D Gaussian reconstruction",
                table="driving reconstruction table",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBaseline(
                name="Sat3DGen",
                family="single-satellite street-level 3D generation",
                satellite_conditioning="satellite-only",
                expected_artifact="satellite-conditioned street-level 3D scene, DSM, and NVS outputs",
                table="satellite/cross-view reconstruction table",
                experiment_phase="phase3_stretch",
            ),
            ProtocolBaseline(
                name="SA-Occ",
                family="satellite-assisted 3D occupancy",
                satellite_conditioning="satellite+surround-camera",
                expected_artifact="satellite-assisted Occ3D-nuScenes semantic occupancy predictions",
                table="occupancy auxiliary table",
                experiment_phase="phase1_external",
            ),
            ProtocolBaseline(
                name="DriveTok",
                family="3D driving scene tokenization",
                satellite_conditioning="none",
                expected_artifact="multi-view reconstruction, depth, segmentation, and 3D occupancy tokens",
                table="driving reconstruction table",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBaseline(
                name="GaussianOcc",
                family="self-supervised Gaussian occupancy",
                satellite_conditioning="none",
                expected_artifact="vision-only occupancy and depth estimates on nuScenes/DDAD",
                table="occupancy auxiliary table",
                experiment_phase="phase2_external",
            ),
            ProtocolBaseline(
                name="GS-Occ3D",
                family="vision-only occupancy reconstruction with Gaussian splatting",
                satellite_conditioning="none",
                expected_artifact="vision-only occupancy labels and downstream occupancy transfer",
                table="occupancy auxiliary table",
                experiment_phase="phase2_external",
            ),
            ProtocolBaseline(
                name="MapTR auxiliary",
                family="vector map baseline",
                satellite_conditioning="none",
                expected_artifact="divider/crossing/boundary AP",
                table="auxiliary map table",
                experiment_phase="phase3_stretch",
            ),
            ProtocolBaseline(
                name="PseudoMapTrainer auxiliary",
                family="pseudo-label map baseline",
                satellite_conditioning="pseudo-map",
                expected_artifact="pseudo-label map quality and observed-area masks",
                table="auxiliary map table",
                experiment_phase="phase3_stretch",
            ),
        ),
        benchmarks=(
            ProtocolBenchmark(
                name="E3D-Bench",
                role="external 3D geometric foundation model benchmark",
                source_url="https://e3dbench.github.io/",
                reason="Covers depth, reconstruction, pose, and end-to-end 3D GFM behavior.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="Sky2Ground",
                role="satellite/aerial/ground site-modeling benchmark",
                source_url="https://arxiv.org/abs/2603.13740",
                reason="CVPR 2026 benchmark for multi-altitude camera localization, correspondence, and reconstruction; directly probes satellite-view usefulness and failure modes.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="CrossGeo",
                role="satellite-drone-ground reconstruction/localization benchmark",
                source_url="https://arxiv.org/abs/2605.07978",
                reason="Tri-view benchmark with satellite tiles, UAV images, and ground images for point-cloud reconstruction, 6-DoF pose, and on-tile localization.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="Sat3DGen-VIGOR-OOD-DSM",
                role="satellite-only street-level 3D generation benchmark",
                source_url="https://arxiv.org/abs/2605.14984",
                reason="Pairs VIGOR-OOD satellite views with high-resolution DSM, giving a satellite-only geometry stress test and DSM RMSE target.",
                experiment_phase="phase3_stretch",
            ),
            ProtocolBenchmark(
                name="Occ3D-nuScenes",
                role="driving scene geometry benchmark",
                source_url="https://tsinghua-mars-lab.github.io/Occ3D/",
                reason="nuScenes-derived occupancy labels provide a practical autonomous-driving geometry target.",
                experiment_phase="phase1_external",
            ),
            ProtocolBenchmark(
                name="OpenScene",
                role="large-scale nuPlan-derived occupancy benchmark",
                source_url="https://github.com/OpenDriveLab/OpenScene",
                reason="Scale-up benchmark with occupancy labels across Boston, Pittsburgh, Las Vegas, and Singapore; useful after nuScenes-mini experiments.",
                experiment_phase="phase3_stretch",
            ),
            ProtocolBenchmark(
                name="UniOcc",
                role="unified occupancy prediction and forecasting benchmark",
                source_url="https://uniocc.github.io/",
                reason="ICCV 2025 benchmark unifying 2D/3D occupancy labels and flow annotations across multiple autonomous-driving datasets.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="SA-Occ",
                role="satellite-assisted 3D occupancy benchmark/baseline",
                source_url="https://openaccess.thecvf.com/content/ICCV2025/html/Chen_SA-Occ_Satellite-Assisted_3D_Occupancy_Prediction_in_Real_World_ICCV_2025_paper.html",
                reason="ICCV 2025 satellite-assisted Occ3D-nuScenes baseline; closest public occupancy comparison to this project's satellite-conditioned 3D geometry claim.",
                experiment_phase="phase1_external",
            ),
            ProtocolBenchmark(
                name="DriveTok",
                role="multi-view driving reconstruction/tokenization benchmark",
                source_url="https://arxiv.org/abs/2603.19219",
                reason="2026 nuScenes multi-view scene-token baseline spanning image reconstruction, depth prediction, semantic segmentation, and 3D occupancy prediction.",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBenchmark(
                name="M2-Occ",
                role="missing-view robustness occupancy benchmark",
                source_url="https://arxiv.org/abs/2603.09737",
                reason="2026 SurroundOcc/nuScenes protocol for deterministic and stochastic camera dropout, useful for testing robustness of satellite priors when cameras are incomplete.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="OpenOccupancy",
                role="occupancy implementation reference",
                source_url="https://github.com/JeffWang987/OpenOccupancy",
                reason="Local clone provides mature nuScenes occupancy evaluation code patterns.",
                experiment_phase="phase1_external",
            ),
            ProtocolBenchmark(
                name="SurroundOcc",
                role="surround-view occupancy baseline",
                source_url="https://github.com/weiyithu/SurroundOcc",
                reason="Useful camera-only occupancy baseline when reporting geometry beyond pointmaps.",
                experiment_phase="phase2_external",
            ),
            ProtocolBenchmark(
                name="DynamicVGGT",
                role="dynamic driving-scene reconstruction baseline",
                source_url="https://arxiv.org/abs/2603.08254",
                reason="Extends VGGT with dynamic pointmaps, temporal attention, and 3D Gaussian motion heads for autonomous-driving 4D reconstruction.",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBenchmark(
                name="PAGE-4D",
                role="VGGT-4D dynamic reconstruction benchmark",
                source_url="https://openreview.net/forum?id=Nfmzp5PBzr",
                reason="ICLR 2026 VGGT extension for dynamic scenes with pose, depth, point cloud, and point tracking outputs; useful once temporal reconstruction is in scope.",
                experiment_phase="phase2_tracking",
            ),
            ProtocolBenchmark(
                name="SG-BEV",
                role="auxiliary satellite/BEV alignment",
                source_url="https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_SG-BEV_Satellite-Guided_BEV_Fusion_for_Cross-View_Semantic_Segmentation_CVPR_2024_paper.pdf",
                reason="Satellite-guided BEV fusion reference for alignment design, not a primary 3D metric baseline.",
                experiment_phase="phase3_stretch",
            ),
        ),
    )


def format_experiment_protocol(protocol: ExperimentProtocol) -> str:
    """Render a compact human-readable protocol."""

    lines = ["primary_metrics:"]
    for metric in protocol.primary_metrics:
        direction = "lower" if metric.lower_is_better else "higher"
        lines.append(f"- {metric.name}: {metric.role}; {direction} is better")
    lines.append("auxiliary_metrics:")
    for metric in protocol.auxiliary_metrics:
        direction = "lower" if metric.lower_is_better else "higher"
        lines.append(f"- {metric.name}: {metric.role}; {direction} is better")
    lines.append("baselines:")
    for baseline in protocol.baselines:
        lines.append(
            f"- {baseline.name}: {baseline.family}; satellite={baseline.satellite_conditioning}; "
            f"phase={baseline.experiment_phase}; {baseline.table}"
        )
    lines.append("benchmarks:")
    for benchmark in protocol.benchmarks:
        lines.append(
            f"- {benchmark.name}: {benchmark.role}; phase={benchmark.experiment_phase}; "
            f"{benchmark.source_url}"
        )
    return "\n".join(lines)
