# Current Code Status

## Summary

The repository is no longer only a paper/reference folder. It now contains a project scaffold with data contracts, a minimal trainable model scaffold, reconstruction losses, train/eval entrypoints, environment scripts, weight-download scripts, dataset preparation notes, manifest asset materialization, and research baseline notes.

It is not yet a complete real nuScenes training codebase. The current train/eval paths are synthetic smoke training/evaluation plus manifest-smoke training that can load real camera, satellite, depth, and mask image files from a JSONL manifest. Real training still needs satellite patch extraction/alignment, pointmap/pose supervision generation, concrete G3T/VGGT head integration, and GPU environment validation.

## Module Status

| Area | Status | Evidence |
| --- | --- | --- |
| Git repository | Ready locally | local commits are present; use `git log --oneline` for the current head |
| Reference code | Ready locally | `refs/g3t`, `refs/look-from-above-components/PseudoMapTrainer`, `refs/look-from-above-components/MapTR` |
| Data contracts | Scaffolded | `src/vggt_project/data/sample.py`, `src/vggt_project/data/synthetic.py`, `src/vggt_project/data/manifest.py`, `src/vggt_project/data/manifest_tensor_dataset.py` |
| Real nuScenes data loading | Partial scaffold | layout inspection, JSONL manifest generation/loading, path validation, smoke satellite/mask asset materialization, real-file smoke tensor loading, and optional depth/mask target image loading exist; real satellite patch extraction and pointmap/pose target generation still need implementation |
| Model framework | Scaffolded | `src/vggt_project/models/scaffold.py` |
| G3T/VGGT integration | Missing | reference code exists, but concrete head reuse/fine-tuning is not implemented |
| Losses | Scaffolded | pointmap, depth, local pose, relative pose losses in `src/vggt_project/losses.py` |
| Evaluation metrics | Scaffolded | depth MAE, pointmap L1, local pose L2, relative pose L2 in `src/vggt_project/metrics.py` |
| Train loop | Smoke-verified scaffold | synthetic train and manifest-smoke train completed in `/tmp/vggt-satellite-smoke` |
| Eval/inference loop | Smoke-verified scaffold | synthetic eval completed against `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt` |
| Environment setup | Smoke-verified script | `requirements.txt`, `environment.yml`, `scripts/setup_env.sh`; Python 3.10/3.11 recommended; PyTorch 2.12.0 imported |
| G3T weight download | Scripted | `scripts/download_weights.py` |
| nuScenes download/setup | Partially scripted | `scripts/prepare_nuscenes.sh`; real download requires account/license |
| Manifest asset materialization | Smoke-only scripted | `scripts/materialize_manifest_assets.py`; creates placeholder satellite patches and optional valid masks for pipeline testing |
| GitHub upload | Blocked by auth | `gh` installed, but `gh auth status` reports no logged-in host |

## Commands

Verified locally:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py --help
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
/tmp/vggt-satellite-smoke/bin/python scripts/train.py --mode manifest-smoke --manifest <temp>/samples.jsonl --epochs 1 --output-dir <temp>/out
```

Requires GitHub auth:

```bash
gh auth login
bash scripts/publish_github.sh VggT
```

## Next Implementation Milestones

1. Extend preprocessing from manifest generation/validation to actual satellite patch extraction.
2. Replace remaining manifest-smoke placeholder pointmap/pose targets with real pointmap/pose/occupancy supervision.
3. Select and implement satellite patch source/alignment.
4. Replace the synthetic scaffold model with a thin adapter around G3T/VGGT heads.
5. Extend metrics beyond smoke values to real reconstruction metrics: scale-aligned pointmap accuracy/completeness, gravity error, and sequence drift.
6. Add optional MapTR-style vector map auxiliary supervision.
