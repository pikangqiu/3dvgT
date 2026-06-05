# Agent Guide

## Project Identity

This is a research project with a deadline of 2026-06-25. The project should be treated as an experimental computer vision codebase, not as a production app.

The central goal is to build a satellite-guided and BEV-guided 3D reconstruction model for autonomous driving scenes. The model should learn an aligned latent representation from nuScenes multi-view camera observations, BEV features, and aligned satellite patches, then feed that latent into a G3T-style gravity-aligned reconstruction module.

## Research Framing

The project is 3D reconstruction first.

The primary model should not be framed as a pure HD map prediction model. Vectorized map prediction and pseudo-labeling can be used as auxiliary supervision, ablation, or structural priors, but the main result should focus on 3D reconstruction quality.

The desired paper story is:

1. Satellite imagery provides a global top-down layout prior.
2. Multi-view onboard images provide local observed scene evidence.
3. BEV lifting/fusion aligns these two domains.
4. G3T-style gravity-aligned reconstruction reduces coordinate-frame instability compared with camera-centric reconstruction.
5. The fused satellite/BEV latent improves autonomous-driving 3D reconstruction under occlusion, partial observability, and long-sequence alignment.

## Reference Papers And Code

### G3T

G3T is the main geometry framework reference.

- Paper: `2605.27372v1.pdf`
- Project page: https://g3t-paper.github.io/
- Code reference: `refs/g3t`

Core ideas to preserve:

- Predict pointmaps in gravity-aligned/upright coordinate frames rather than arbitrary camera-centric frames.
- Decompose pose prediction into local camera-to-gravity pose and relative yaw/translation.
- Use the reduced rotational degrees of freedom in upright frames to improve submap and long-sequence alignment.
- Treat G3T as the main reference for geometry heads, pointmap/depth/pose losses, and gravity-aligned reconstruction evaluation.

### Look from Above

Look from Above is the main task/data/fusion paradigm reference.

- Paper: `859_Look_from_Above_Satellite_.pdf`
- Local reference notes: `refs/look-from-above`

Core ideas to borrow:

- Use nuScenes as the initial dataset setting.
- Align onboard BEV features with satellite-map patches.
- Use satellite imagery as a global road-layout prior.
- Use BEV valid/observed regions to avoid supervising unobserved areas too strongly.
- Treat vectorized map prediction as an auxiliary structural task, not the primary objective.

As of the initial scaffold, no official public code repository for this anonymous ECCV 2026 submission has been identified. Do not substitute unrelated baseline repositories as "official" code. If a real code URL is provided later, clone it under `refs/look-from-above`.

## Model Direction

The first target architecture should be:

```text
multi-view images
    -> image encoder
    -> BEV lifting / BEV encoder
    -> fusion with aligned satellite encoder
    -> shared latent
    -> G3T-style gravity-aligned reconstruction heads
    -> optional map/vector auxiliary heads
```

Expected heads:

- Gravity-aligned pointmap head.
- Depth head.
- Local camera-to-gravity pose head.
- Relative yaw/translation head.
- Optional BEV occupancy or semantic reconstruction head.
- Optional vectorized map auxiliary head.

## Primary Evaluation

The primary evaluation should use 3D reconstruction metrics, such as:

- Pointmap accuracy.
- Depth error.
- Camera pose error.
- Gravity alignment error.
- Long-sequence submap alignment drift.
- LiDAR/depth/BEV occupancy agreement when direct dense 3D ground truth is unavailable.

Map metrics such as nuScenes vector map mAP can be reported as auxiliary evidence, but should not become the main success criterion unless the project framing changes.

## Baselines And Ablations

The minimum ablation ladder should aim for:

1. Camera-centric VGGT-style baseline.
2. G3T-style gravity-aligned reconstruction without satellite conditioning.
3. BEV + G3T-style reconstruction.
4. BEV + satellite + G3T-style reconstruction.
5. BEV + satellite + G3T-style reconstruction + map/vector auxiliary supervision.

## Engineering Principles

- Keep reference code under `refs/`; do not edit reference repositories directly unless the user explicitly asks.
- Build new project code under `src/`, with configs in `configs/` and scripts in `scripts/`.
- Prefer thin adapters around reference implementations before large rewrites.
- Preserve coordinate-frame clarity in filenames, comments, and APIs.
- Any function that transforms coordinates must state source frame, target frame, scale convention, and units.
- Treat nuScenes calibration, ego pose, satellite alignment, BEV rasterization, and gravity/yaw conventions as high-risk areas requiring explicit tests or sanity checks.
- Avoid claiming a result until the command, metric, or visualization proving it has been run.

