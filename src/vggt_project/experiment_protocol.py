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


@dataclass(frozen=True)
class ProtocolBenchmark:
    name: str
    role: str
    source_url: str
    reason: str


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
                name="occupancy_miou",
                role="driving scene geometry",
                lower_is_better=False,
                note="Use when evaluating against Occ3D/OpenOccupancy-style labels.",
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
            ),
            ProtocolBaseline(
                name="G3T",
                family="gravity-aligned 3D reconstruction",
                satellite_conditioning="none",
                expected_artifact="gravity-aligned pointmap and pose predictions",
                table="primary reconstruction table",
            ),
            ProtocolBaseline(
                name="BEV+G3T",
                family="ours ablation",
                satellite_conditioning="bev-only",
                expected_artifact="BEV-conditioned reconstruction outputs",
                table="primary reconstruction table",
            ),
            ProtocolBaseline(
                name="BEV+Satellite+G3T",
                family="ours main",
                satellite_conditioning="bev+satellite",
                expected_artifact="satellite-conditioned reconstruction outputs",
                table="primary reconstruction table",
            ),
            ProtocolBaseline(
                name="MapTR auxiliary",
                family="vector map baseline",
                satellite_conditioning="none",
                expected_artifact="divider/crossing/boundary AP",
                table="auxiliary map table",
            ),
            ProtocolBaseline(
                name="PseudoMapTrainer auxiliary",
                family="pseudo-label map baseline",
                satellite_conditioning="pseudo-map",
                expected_artifact="pseudo-label map quality and observed-area masks",
                table="auxiliary map table",
            ),
        ),
        benchmarks=(
            ProtocolBenchmark(
                name="E3D-Bench",
                role="external 3D geometric foundation model benchmark",
                source_url="https://e3dbench.github.io/",
                reason="Covers depth, reconstruction, pose, and end-to-end 3D GFM behavior.",
            ),
            ProtocolBenchmark(
                name="Occ3D-nuScenes",
                role="driving scene geometry benchmark",
                source_url="https://tsinghua-mars-lab.github.io/Occ3D/",
                reason="nuScenes-derived occupancy labels provide a practical autonomous-driving geometry target.",
            ),
            ProtocolBenchmark(
                name="OpenOccupancy",
                role="occupancy implementation reference",
                source_url="https://github.com/JeffWang987/OpenOccupancy",
                reason="Local clone provides mature nuScenes occupancy evaluation code patterns.",
            ),
            ProtocolBenchmark(
                name="SurroundOcc",
                role="surround-view occupancy baseline",
                source_url="https://github.com/weiyithu/SurroundOcc",
                reason="Useful camera-only occupancy baseline when reporting geometry beyond pointmaps.",
            ),
            ProtocolBenchmark(
                name="SG-BEV",
                role="auxiliary satellite/BEV alignment",
                source_url="https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_SG-BEV_Satellite-Guided_BEV_Fusion_for_Cross-View_Semantic_Segmentation_CVPR_2024_paper.pdf",
                reason="Satellite-guided BEV fusion reference for alignment design, not a primary 3D metric baseline.",
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
            f"- {baseline.name}: {baseline.family}; satellite={baseline.satellite_conditioning}; {baseline.table}"
        )
    lines.append("benchmarks:")
    for benchmark in protocol.benchmarks:
        lines.append(f"- {benchmark.name}: {benchmark.role}; {benchmark.source_url}")
    return "\n".join(lines)
