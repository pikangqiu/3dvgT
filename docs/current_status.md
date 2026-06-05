# Current Code Status

## Summary

The repository is no longer only a paper/reference folder. It now contains a project scaffold with data contracts, a minimal trainable model scaffold, reconstruction losses, train/eval entrypoints, environment scripts, weight-download scripts, dataset preparation notes, and research baseline notes.

It is not yet a complete real nuScenes training codebase. The current train/eval path is a synthetic smoke test that proves the plumbing shape and has been run in a Python 3.11 environment. Real training still needs the nuScenes sample loader, satellite patch source/alignment, concrete G3T/VGGT head integration, and GPU environment validation.

## Module Status

| Area | Status | Evidence |
| --- | --- | --- |
| Git repository | Ready locally | commits through `8c98a62` |
| Reference code | Ready locally | `refs/g3t`, `refs/look-from-above-components/PseudoMapTrainer`, `refs/look-from-above-components/MapTR` |
| Data contracts | Scaffolded | `src/vggt_project/data/sample.py`, `src/vggt_project/data/synthetic.py` |
| Real nuScenes data loading | Partial scaffold | layout inspection exists; sample loading still needs camera, calibration, ego pose, satellite patch, valid mask, optional LiDAR/map targets |
| Model framework | Scaffolded | `src/vggt_project/models/scaffold.py` |
| G3T/VGGT integration | Missing | reference code exists, but concrete head reuse/fine-tuning is not implemented |
| Losses | Scaffolded | pointmap, depth, local pose, relative pose losses in `src/vggt_project/losses.py` |
| Train loop | Smoke-verified scaffold | synthetic train completed in `/tmp/vggt-satellite-smoke` |
| Eval/inference loop | Smoke-verified scaffold | synthetic eval completed against `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt` |
| Environment setup | Smoke-verified script | `requirements.txt`, `environment.yml`, `scripts/setup_env.sh`; Python 3.10/3.11 recommended; PyTorch 2.12.0 imported |
| G3T weight download | Scripted | `scripts/download_weights.py` |
| nuScenes download/setup | Partially scripted | `scripts/prepare_nuscenes.sh`; real download requires account/license |
| GitHub upload | Blocked by auth | `gh` installed, but `gh auth status` reports no logged-in host |

## Commands

Verified locally:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
python3 -m compileall src scripts tests
```

Requires PyTorch environment:

```bash
bash scripts/setup_env.sh .venv
source .venv/bin/activate
export MPLCONFIGDIR=.venv/.matplotlib
PYTHONPATH=src python scripts/train.py --mode synthetic --epochs 1
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
```

Smoke run evidence from this machine:

```text
/tmp/vggt-satellite-smoke/bin/python scripts/train.py --mode synthetic --epochs 1 --output-dir /tmp/vggt-satellite-smoke-output
/tmp/vggt-satellite-smoke/bin/python scripts/evaluate.py --mode synthetic --checkpoint /tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt
```

Requires GitHub auth:

```bash
gh auth login
bash scripts/publish_github.sh VggT
```

## Next Implementation Milestones

1. Extend the nuScenes adapter from layout inspection to real sample loading.
2. Select and implement satellite patch source/alignment.
3. Replace the synthetic scaffold model with a thin adapter around G3T/VGGT heads.
4. Add real reconstruction metrics: depth, pointmap, pose, gravity alignment, and sequence drift.
5. Add optional MapTR-style vector map auxiliary supervision.
