# Implementation Roadmap

## Accepted Reference Setup

The project uses three local code references:

- `refs/g3t`: official G3T code for gravity-aligned pointmaps, pose heads, and long-sequence reconstruction.
- `refs/look-from-above-components/PseudoMapTrainer`: component reference for pseudo-label generation, semantic reconstruction, Gaussian-splatting cues, and mask-aware training.
- `refs/look-from-above-components/MapTR`: component reference for nuScenes vectorized map conventions and optional map heads.

Look from Above has no codebase. Its role is to define the satellite/BEV fusion paradigm:

```text
onboard multi-view observations -> reconstructed/encoded BEV
aligned satellite patch -> satellite layout prior
BEV + satellite -> fused scene latent
```

This repository should implement the fusion and reconstruction system itself.

## Phase 1: Data Contracts

Goal: make nuScenes samples explicit before training.

- Define camera image, calibration, ego pose, BEV, satellite patch, valid mask, and optional LiDAR/vector-map fields.
- Include ego pose translation/rotation and map location in manifests so satellite crops can be aligned later.
- Convert manifest ego pose metadata into coarse pose targets for the current train/eval scaffold.
- Support optional pointmap target files in manifests before implementing G3T-style pointmap target generation.
- Generate ego-frame LiDAR pointmap targets as the first real pointmap supervision path.
- Compose LiDAR depth and pointmap preprocessing into one supervised manifest pipeline for training setup.
- Add coordinate-frame checks for camera, ego, BEV, satellite, and gravity frames.
- Decide the first satellite patch source and resolution.
- Provide a satellite raster config for `scripts/materialize_satellite_crops.py`.
- Produce one inspectable sample artifact before building a trainer.

## Phase 2: Fusion Encoder

Goal: implement the Look from Above-style encoder path.

- Start with a BEV feature/raster input branch.
- Add a satellite patch encoder branch.
- Fuse BEV and satellite features into a shared scene latent.
- Preserve valid-area masks for downstream losses.
- Keep the module independent from final reconstruction heads.

## Phase 3: G3T-Style Reconstruction Heads

Goal: connect fused latent features to reconstruction-first outputs.

- Reuse G3T code as the geometry reference.
- Implement heads for gravity-aligned pointmap, depth, local camera-to-gravity pose, and relative yaw/translation.
- Keep camera-centric baseline support for ablations.
- Record every coordinate transformation explicitly.

## Phase 4: Auxiliary Mapping

Goal: use map supervision as structure, not as the main task.

- Use MapTR conventions for vectorized map targets if a map auxiliary head is enabled.
- Use PseudoMapTrainer ideas for pseudo-label masks and valid-area weighting.
- Report map metrics only as auxiliary evidence.

## Phase 5: First Evaluation Ladder

Minimum ablations:

1. Camera-centric VGGT/G3T reference baseline.
2. G3T-style gravity reconstruction without satellite conditioning.
3. BEV-conditioned G3T-style reconstruction.
4. BEV + satellite-conditioned G3T-style reconstruction.
5. BEV + satellite + auxiliary map/pseudo-label supervision.

Primary metrics:

- Pointmap accuracy.
- Depth error.
- Camera pose error.
- Gravity alignment error.
- Long-sequence alignment drift.

Current scaffold metric names:

- `depth_mae`
- `scale_aligned_pointmap_accuracy`
- `scale_aligned_pointmap_completeness`
- `scale_aligned_pointmap_chamfer`
- `gravity_error_deg`
- `sequence_translation_drift`
