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
- train and eval entrypoints,
- setup, weight-download, dataset-preparation, and GitHub-publish scripts,
- baseline/benchmark notes for the next experimental plan.

The scaffold is not yet a complete nuScenes training implementation. Real training still needs the nuScenes dataset adapter, concrete satellite patch source/alignment, and integration with selected G3T/VGGT heads.

## Quick Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_references.py
```

After installing dependencies:

```bash
PYTHONPATH=src python3 scripts/train.py --mode synthetic --epochs 1
PYTHONPATH=src python3 scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
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
