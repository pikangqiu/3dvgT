# VggT

Satellite-guided and BEV-guided 3D reconstruction research scaffold.

This project uses G3T as the main gravity-aligned reconstruction reference and uses the Look from Above paper as the main nuScenes satellite/BEV alignment paradigm reference.

See `agent.md` for the research纲领 and working rules.
See `docs/current_status.md` for the precise implementation status and remaining training gaps.

## Current Status

This repository now contains a runnable scaffold for the project structure:

- data contracts and synthetic smoke-test data,
- a minimal satellite/BEV-conditioned reconstruction model scaffold,
- reconstruction losses,
- reconstruction metrics,
- train and eval entrypoints,
- setup, weight-download, dataset-preparation, and GitHub-publish scripts,
- nuScenes manifest generation and loading scaffolds,
- manifest asset materialization for smoke satellite patches and valid masks,
- nuScenes LiDAR-to-camera depth target generation for manifest supervision,
- manifest smoke training from real image files,
- baseline/benchmark notes for the next experimental plan.

The scaffold is not yet a complete nuScenes training implementation. Real training still needs a concrete satellite patch source/alignment implementation, real pointmap/pose supervision generation, multi-camera target wiring, and integration with selected G3T/VGGT heads.

## Quick Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_references.py
```

After installing dependencies:

```bash
PYTHONPATH=src python3 scripts/train.py --mode synthetic --epochs 1
PYTHONPATH=src python3 scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py data/manifests/nuscenes-mini.jsonl --create-valid-masks --output data/manifests/nuscenes-mini.smoke.jsonl
PYTHONPATH=src python3 scripts/generate_lidar_depth_targets.py data/manifests/nuscenes-mini.smoke.jsonl --output data/manifests/nuscenes-mini.depth.jsonl
PYTHONPATH=src python3 scripts/train.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.depth.jsonl --epochs 1 --output-dir outputs/manifest-smoke
PYTHONPATH=src python3 scripts/evaluate.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.depth.jsonl --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt
```

## Layout

```text
.
├── agent.md
├── configs/
├── docs/
├── refs/
│   ├── g3t/
│   └── look-from-above/
├── scripts/
└── src/
```

`refs/g3t` is a cloned reference repository. `refs/look-from-above` stores notes for the Look from Above paradigm, which currently has no codebase. `refs/look-from-above-components` contains public component references such as PseudoMapTrainer and MapTR; these are the accepted engineering references for implementing the paper's ideas.
