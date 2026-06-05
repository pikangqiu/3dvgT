# Architecture Sketch

This project starts from a reconstruction-first interpretation of the research goal.

## Reference Roles

G3T provides the geometry reference: gravity-aligned pointmaps, camera-to-gravity pose, relative yaw/translation, and long-sequence submap alignment.

Look from Above provides the data and fusion reference: nuScenes, aligned satellite patches, BEV features, valid-area masking, and optional vector-map supervision.

## Initial Data Flow

```text
nuScenes multi-view cameras
    -> image encoder
    -> BEV lifting / BEV encoder

aligned satellite patch
    -> satellite encoder

BEV latent + satellite latent
    -> fusion encoder
    -> shared scene latent
    -> G3T-style gravity-aligned reconstruction heads
    -> optional map/BEV auxiliary heads
```

## Immediate Engineering Goal

The first implementation should make the reference boundaries explicit before training:

- A dataset sample type that names every nuScenes input and coordinate frame.
- A model interface that separates fusion from reconstruction heads.
- A reference checker that verifies G3T is available and records that Look from Above is paper-only until a public repository is known.

