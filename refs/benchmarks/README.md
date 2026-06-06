# Benchmark Reference Repositories

This directory stores local clones of benchmark/baseline code used to design experiments.

These repositories are ignored by the main git repository and should be treated as external references.

## Cloned References

- `E3D-Bench`: benchmark for end-to-end 3D geometric foundation models.
  - Remote: `https://github.com/VITA-Group/E3D-Bench.git`
  - Initial HEAD: `11d82b4`
- `OpenOccupancy`: nuScenes semantic occupancy benchmark.
  - Remote: `https://github.com/JeffWang987/OpenOccupancy.git`
  - Initial HEAD: `eafd14f`
- `SurroundOcc`: multi-camera 3D occupancy prediction baseline.
  - Remote: `https://github.com/weiyithu/SurroundOcc.git`
  - Initial HEAD: `419bf5b`

## Scripted Optional References

`scripts/setup_references.py` can restore these official baseline/benchmark repositories when those comparisons become active:

- `DGGT`: pose-free feed-forward 4D driving reconstruction baseline.
  - Remote: `https://github.com/xiaomi-research/dggt.git`
- `DrivingForward`: nuScenes feed-forward driving-scene Gaussian splatting baseline.
  - Remote: `https://github.com/fangzhou2000/DrivingForward.git`
- `GaussianOcc`: self-supervised Gaussian-splatting occupancy baseline.
  - Remote: `https://github.com/GANWANSHUI/GaussianOcc.git`
- `OpenScene`: large-scale nuPlan-derived occupancy benchmark.
  - Remote: `https://github.com/OpenDriveLab/OpenScene.git`
- `UniOcc`: unified occupancy prediction and forecasting benchmark.
  - Remote: `https://github.com/tasl-lab/UniOcc.git`
- `Sat3DGen`: single-satellite street-level 3D generation reference.
  - Remote: `https://github.com/qianmingduowan/Sat3DGen.git`

## Track Until Public Code Is Confirmed

- `SA-Occ`: satellite-assisted 3D occupancy baseline on Occ3D-nuScenes; track for the closest public satellite-conditioned geometry comparison.
- `DriveTok`: 2026 nuScenes multi-view reconstruction/understanding scene-token baseline.
- `M2-Occ`: 2026 missing-camera robustness occupancy protocol on SurroundOcc/nuScenes.
- `Cross3R/CrossGeo`: closest satellite-drone-ground feed-forward reconstruction benchmark; clone when running cross-view reconstruction experiments.
- `Sky2Ground/SkyNet`: satellite/aerial/ground site modeling benchmark; clone when code is released and when evaluating satellite-view robustness.
- `ReconDrive`: VGGT-adapted feed-forward 4D Gaussian splatting baseline for nuScenes-like driving reconstruction.
- `DynamicVGGT`: VGGT-derived dynamic pointmap and 4D autonomous-driving reconstruction baseline; track until public code is confirmed.
- `UniSplat`: ICLR 2026 feed-forward dynamic driving reconstruction baseline using 3D latent scaffolds and spatio-temporal fusion; clone when running dynamic driving reconstruction comparisons.
- `DenoiseGS`: AAAI 2026 dynamic driving 3DGS robustness baseline for camera-pose noise and dynamic trajectory errors; use as an evaluation reference when pose/noise robustness is claimed.
- `Splat2BEV`: 2026 reconstruction-to-BEV design reference; track as supporting evidence for geometry-aligned BEV representations rather than as a direct 3D reconstruction benchmark.
- `GS-Occ3D`: scalable vision-only occupancy reconstruction benchmark; track project page until an official repository is linked.
