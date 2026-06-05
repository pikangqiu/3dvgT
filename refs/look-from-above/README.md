# Look from Above Reference

This directory tracks paper-derived implementation notes for `859_Look_from_Above_Satellite_.pdf`.

## Role In This Project

This paper is the reference for the nuScenes task setup and satellite/BEV fusion paradigm.

Its original target is satellite-guided generative mapping for robust pseudo-labeling:

- Input: reconstructed onboard BEV representation plus aligned satellite-map patch.
- Main output: vectorized HD map elements.
- Training/evaluation focus: observed/valid BEV area and pseudo-label quality.

For this project, the paper is not the main task definition. We borrow its data alignment and fusion strategy, then redirect the main objective toward 3D reconstruction.

## Borrowed Components

- nuScenes-based data organization.
- Multi-view camera to BEV processing.
- Satellite patch alignment with the local road segment.
- Dual-branch BEV/satellite feature encoding.
- Valid-area masking to avoid over-penalizing unobserved regions.
- Optional vectorized map auxiliary supervision.

## Current Code Status

No official public repository was identified during initial setup. If the user provides an official code URL later, clone it into this directory or replace this notes directory with a clone under `refs/look-from-above-code`.

Component-level public references have been cloned under `refs/look-from-above-components/`:

- `PseudoMapTrainer`: official implementation of the ICCV 2025 pseudo-label online mapping baseline cited by the paper.
- `MapTR`: official implementation of the vectorized online HD map construction baseline/head family cited by the paper.

Use these only as component references. Do not describe them as the Look from Above official code.
