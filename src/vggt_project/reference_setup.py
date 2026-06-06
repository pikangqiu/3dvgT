"""Reference repository setup plan for this research project."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceRepositorySpec:
    name: str
    url: str
    path: Path
    purpose: str


@dataclass(frozen=True)
class ReferenceClonePlan:
    spec: ReferenceRepositorySpec
    path: Path
    exists: bool
    command: list[str]


def reference_specs() -> tuple[ReferenceRepositorySpec, ...]:
    """Repositories needed to recreate local external-code references."""

    return (
        ReferenceRepositorySpec(
            name="g3t",
            url="https://github.com/g3t-paper/g3t.git",
            path=Path("refs/g3t"),
            purpose="gravity-aligned 3D reconstruction reference",
        ),
        ReferenceRepositorySpec(
            name="pseudomaptrainer_component",
            url="https://github.com/boschresearch/PseudoMapTrainer.git",
            path=Path("refs/look-from-above-components/PseudoMapTrainer"),
            purpose="pseudo-label and mask-aware mapping reference",
        ),
        ReferenceRepositorySpec(
            name="maptr_component",
            url="https://github.com/hustvl/MapTR.git",
            path=Path("refs/look-from-above-components/MapTR"),
            purpose="vectorized HD map auxiliary-head reference",
        ),
        ReferenceRepositorySpec(
            name="e3d_bench_reference",
            url="https://github.com/VITA-Group/E3D-Bench.git",
            path=Path("refs/benchmarks/E3D-Bench"),
            purpose="3D geometric foundation model benchmark reference",
        ),
        ReferenceRepositorySpec(
            name="open_occupancy_reference",
            url="https://github.com/JeffWang987/OpenOccupancy.git",
            path=Path("refs/benchmarks/OpenOccupancy"),
            purpose="nuScenes occupancy benchmark reference",
        ),
        ReferenceRepositorySpec(
            name="surround_occ_reference",
            url="https://github.com/weiyithu/SurroundOcc.git",
            path=Path("refs/benchmarks/SurroundOcc"),
            purpose="surround-view occupancy benchmark reference",
        ),
        ReferenceRepositorySpec(
            name="dggt_reference",
            url="https://github.com/xiaomi-research/dggt.git",
            path=Path("refs/benchmarks/DGGT"),
            purpose="pose-free feed-forward 4D driving reconstruction baseline",
        ),
        ReferenceRepositorySpec(
            name="drivingforward_reference",
            url="https://github.com/fangzhou2000/DrivingForward.git",
            path=Path("refs/benchmarks/DrivingForward"),
            purpose="nuScenes feed-forward driving-scene Gaussian splatting baseline",
        ),
        ReferenceRepositorySpec(
            name="gaussianocc_reference",
            url="https://github.com/GANWANSHUI/GaussianOcc.git",
            path=Path("refs/benchmarks/GaussianOcc"),
            purpose="self-supervised Gaussian-splatting occupancy baseline",
        ),
        ReferenceRepositorySpec(
            name="openscene_reference",
            url="https://github.com/OpenDriveLab/OpenScene.git",
            path=Path("refs/benchmarks/OpenScene"),
            purpose="large-scale nuPlan-derived occupancy benchmark reference",
        ),
        ReferenceRepositorySpec(
            name="uniocc_reference",
            url="https://github.com/tasl-lab/UniOcc.git",
            path=Path("refs/benchmarks/UniOcc"),
            purpose="unified occupancy prediction and forecasting benchmark reference",
        ),
        ReferenceRepositorySpec(
            name="sat3dgen_reference",
            url="https://github.com/qianmingduowan/Sat3DGen.git",
            path=Path("refs/benchmarks/Sat3DGen"),
            purpose="single-satellite street-level 3D generation reference",
        ),
    )


def reference_clone_plans(root: Path = Path(".")) -> list[ReferenceClonePlan]:
    """Return clone/skip plans without touching the network."""

    resolved_root = root.resolve()
    plans: list[ReferenceClonePlan] = []
    for spec in reference_specs():
        path = resolved_root / spec.path
        plans.append(
            ReferenceClonePlan(
                spec=spec,
                path=path,
                exists=(path / ".git").exists(),
                command=["git", "clone", spec.url, str(path)],
            )
        )
    return plans


def setup_reference_repositories(
    root: Path = Path("."),
    *,
    dry_run: bool = False,
) -> list[ReferenceClonePlan]:
    """Clone missing reference repositories, or only report actions in dry-run mode."""

    plans = reference_clone_plans(root=root)
    if dry_run:
        return plans

    for plan in plans:
        if plan.exists:
            continue
        plan.path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(plan.command, check=True)
    return reference_clone_plans(root=root)
