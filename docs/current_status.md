# Current Code Status

## Summary

The repository is no longer only a paper/reference folder. It now contains a project scaffold with data contracts, a minimal trainable model scaffold, reconstruction losses, train/eval entrypoints, config-driven experiment dispatch and reporting, an end-to-end smoke pipeline, project-status audit scripts, environment scripts, external-reference setup scripts, weight-download scripts, dataset preparation notes, manifest asset materialization, nuScenes LiDAR-depth target generation, nuScenes LiDAR pointmap target generation, combined LiDAR supervision manifest generation, and research baseline notes.

It is not yet a complete real nuScenes training codebase. The current train/eval paths are synthetic smoke training/evaluation plus manifest-smoke training that can load real camera, satellite, depth, mask, LiDAR-derived pointmap, and ego-pose-derived target tensors from a JSONL manifest. Real training still needs real satellite patch extraction/alignment, G3T-style pointmap target generation, camera-level/G3T pose target wiring, multi-camera target wiring, concrete G3T/VGGT head integration, and GPU environment validation.

## Module Status

| Area | Status | Evidence |
| --- | --- | --- |
| Git repository | Ready locally | local commits are present; use `git log --oneline` for the current head |
| Project status audit | Scripted | `scripts/audit_project_status.py` reports scaffold readiness and remaining real-training gaps |
| Reference code | Ready locally | `refs/g3t`, `refs/look-from-above-components/PseudoMapTrainer`, `refs/look-from-above-components/MapTR` |
| Reference setup | Scripted | `scripts/setup_references.py --dry-run` prints clone plans for ignored external repos |
| Data contracts | Scaffolded | `src/vggt_project/data/sample.py`, `src/vggt_project/data/synthetic.py`, `src/vggt_project/data/manifest.py`, `src/vggt_project/data/manifest_tensor_dataset.py` |
| Real nuScenes data loading | Partial scaffold | layout inspection, JSONL manifest generation/loading with ego pose and map location metadata, path validation, smoke satellite/mask asset materialization, local raster satellite crop materialization, LiDAR-projected camera depth target generation, LiDAR-to-ego pointmap target generation, real-file smoke tensor loading, optional depth/mask/pointmap target image/array loading, and ego-pose-derived pose targets exist; real satellite raster sourcing, G3T-style pointmap target generation, and G3T/VGGT camera-pose adapter wiring still need implementation |
| Model framework | Scaffolded | `src/vggt_project/models/scaffold.py` |
| G3T/VGGT integration | Missing | reference code exists, but concrete head reuse/fine-tuning is not implemented |
| Losses | Scaffolded | pointmap, depth, local pose, relative pose losses in `src/vggt_project/losses.py` |
| Evaluation metrics | Scaffolded | depth MAE, pointmap L1, local pose L2, relative pose L2 in `src/vggt_project/metrics.py` |
| Train loop | Smoke-verified scaffold | synthetic train and manifest-smoke train completed in `/tmp/vggt-satellite-smoke` |
| Eval/inference loop | Smoke-verified scaffold | synthetic eval completed against `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt`; manifest-smoke eval is implemented and covered by tests |
| Experiment config dispatch | Scripted | `configs/reconstruction_first.yaml` can drive current scaffold train/eval through `scripts/train.py --config ...` and `scripts/evaluate.py --config ...` |
| Experiment report pipeline | Scripted | `scripts/run_experiment.py` runs config-driven train+eval and writes a JSON report |
| End-to-end smoke pipeline | Smoke-verified scaffold | `scripts/run_smoke_pipeline.py` creates toy image/depth/mask files, trains, and evaluates |
| Environment setup | Smoke-verified script | `requirements.txt`, `environment.yml`, `scripts/setup_env.sh`; Python 3.10/3.11 recommended; PyTorch 2.12.0 imported |
| G3T weight download | Scripted | `scripts/download_weights.py` |
| nuScenes download/setup | Partially scripted | `scripts/prepare_nuscenes.sh`; real download requires account/license |
| Manifest asset materialization | Smoke-only scripted | `scripts/materialize_manifest_assets.py`; creates placeholder satellite patches and optional valid masks for pipeline testing |
| Satellite crop materialization | Scripted | `scripts/materialize_satellite_crops.py`; crops patches from user-provided local satellite rasters using ego pose metadata |
| LiDAR depth target generation | Scripted for one camera | `scripts/generate_lidar_depth_targets.py`; projects `LIDAR_TOP` to `CAM_FRONT` by default |
| LiDAR pointmap target generation | Scripted | `scripts/generate_lidar_pointmap_targets.py`; transforms `LIDAR_TOP` points into ego-frame `.npy` pointmap targets |
| LiDAR supervision pipeline | Scripted | `scripts/generate_lidar_supervision.py`; generates depth and pointmap targets and writes one final supervised manifest |
| GitHub upload | Blocked by auth | `gh` installed, but `gh auth status` reports no logged-in host |
| GitHub publish preflight | Scripted | `scripts/check_github_publish.py` reports worktree, branch, remote, and `gh auth` state |
| GitHub CI | Scripted | `.github/workflows/ci.yml` runs lightweight tests, project audit, reference dry-run, and compile checks |

## Commands

Verified locally:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/audit_project_status.py
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/setup_references.py --dry-run
PYTHONPATH=src python3 scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_depth_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_pointmap_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_supervision.py --help
PYTHONPATH=src python3 scripts/run_experiment.py --help
PYTHONPATH=src python3 scripts/run_smoke_pipeline.py --help
python3 -m compileall src scripts tests
```

Requires PyTorch environment:

```bash
bash scripts/setup_env.sh .venv
source .venv/bin/activate
export MPLCONFIGDIR=.venv/.matplotlib
PYTHONPATH=src python scripts/train.py --mode synthetic --epochs 1
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
PYTHONPATH=src python scripts/evaluate.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.depth.jsonl --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt
PYTHONPATH=src python scripts/train.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python scripts/evaluate.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/reconstruction_first.yaml --report outputs/reconstruction_first_report.json
PYTHONPATH=src python scripts/run_smoke_pipeline.py --output-dir outputs/smoke-pipeline --epochs 1
```

Smoke run evidence from this machine:

```text
/tmp/vggt-satellite-smoke/bin/python scripts/train.py --mode synthetic --epochs 1 --output-dir /tmp/vggt-satellite-smoke-output
/tmp/vggt-satellite-smoke/bin/python scripts/evaluate.py --mode synthetic --checkpoint /tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt
MPLCONFIGDIR=/tmp/vggt-mpl /tmp/vggt-satellite-smoke/bin/python -m unittest discover -s tests -v
/tmp/vggt-satellite-smoke/bin/python scripts/train.py --mode manifest-smoke --manifest <temp>/samples.jsonl --epochs 1 --output-dir <temp>/out
```

Requires GitHub auth:

```bash
gh auth login
bash scripts/publish_github.sh VggT
```

## Next Implementation Milestones

1. Replace placeholder satellite materialization with actual satellite patch extraction.
2. Extend LiDAR depth generation from one camera to multi-camera supervision.
3. Replace ego-frame LiDAR pointmap targets and coarse ego-pose targets with camera-level G3T/VGGT pointmap/pose supervision.
4. Select and implement satellite patch source/alignment.
5. Replace the synthetic scaffold model with a thin adapter around G3T/VGGT heads.
6. Extend metrics beyond smoke values to real reconstruction metrics: scale-aligned pointmap accuracy/completeness, gravity error, and sequence drift.
7. Add optional MapTR-style vector map auxiliary supervision.
