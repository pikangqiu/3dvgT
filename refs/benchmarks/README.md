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

## Not Yet Cloned

- `Cross3R/CrossGeo`: closest satellite-drone-ground feed-forward reconstruction benchmark; clone when running cross-view reconstruction experiments.
- `Sky2Ground/SkyNet`: satellite/aerial/ground site modeling benchmark; clone when code is released and when evaluating satellite-view robustness.
- `DGGT`: relevant driving 4D reconstruction baseline; clone when ready to run that comparison.
- `DrivingForward`: relevant nuScenes feed-forward reconstruction baseline; clone when selecting Gaussian-splatting reconstruction experiments.
- `ReconDrive`: VGGT-adapted feed-forward 4D Gaussian splatting baseline for nuScenes-like driving reconstruction.
- `GaussianOcc` and `GS-Occ3D`: Gaussian-splatting occupancy baselines for occupancy auxiliary experiments.
- `OpenScene` and `UniOcc`: larger-scale occupancy/prediction benchmark references for post-nuScenes-mini scaling.
