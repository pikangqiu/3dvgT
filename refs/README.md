# Reference Code And Papers

This directory contains external reference material for the research project.

The external code repositories in this directory are intentionally ignored by the main git repository. After cloning the project from GitHub, recreate them with:

```bash
PYTHONPATH=src python scripts/setup_references.py --dry-run
PYTHONPATH=src python scripts/setup_references.py
```

## `g3t`

Official G3T reference code cloned from:

```text
https://github.com/g3t-paper/g3t.git
```

Use it as the geometry reference for gravity-aligned pointmaps, depth/pose heads, and long-sequence submap alignment.

## `look-from-above`

Reference notes for:

```text
Look from Above: Satellite-Guided Generative Mapping for Robust Pseudo-Labeling
```

No official public code repository has been found yet. Keep this directory as a paper-derived reference until the real repository is known.

## `look-from-above-components`

Public component-level repositories related to the Look from Above paper:

- `PseudoMapTrainer`: cloned from `https://github.com/boschresearch/PseudoMapTrainer.git`.
- `MapTR`: cloned from `https://github.com/hustvl/MapTR.git`.

These are not the official implementation of Look from Above. They are reference code for pseudo-label generation, mask-aware training, vectorized map prediction, and nuScenes online mapping components that the paper discusses or uses as baselines/downstream structure.
