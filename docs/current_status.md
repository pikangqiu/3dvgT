# Current Code Status

## Summary

The repository is no longer only a paper/reference folder. It now contains a project scaffold with data contracts, a minimal trainable model scaffold with camera-specific depth and pointmap prediction, external adapter module loading, a repo-local G3T/VGGT adapter template, config-level local G3T/VGGT reference adapter selection, reference-adapter checkpoint loading hooks, model adapter contract checks, reconstruction losses, train/eval entrypoints, reconstruction-first metrics, config-driven experiment dispatch and reporting, an end-to-end smoke pipeline, project-status audit scripts, environment/readiness scripts, ordered training run-plan generation, external-reference setup scripts, weight-download scripts, dataset preparation notes, manifest asset materialization, manifest sample preview, satellite raster config validation, multi-camera nuScenes LiDAR-depth target generation and loss/metric wiring, nuScenes ego-frame and camera-frame LiDAR pointmap target generation, combined LiDAR supervision manifest generation, scene-level manifest splitting, GitHub publishing, research baseline notes, and a machine-readable experiment protocol.

It is not yet a complete real nuScenes training codebase. The current train/eval paths are synthetic smoke training/evaluation plus manifest-smoke training that can load real camera, satellite, single- or multi-camera depth, mask, sample-level or camera-level pointmap, and ego-pose-derived, explicit, or calibration-derived camera-level pose target tensors from a JSONL manifest. Real training still needs real satellite patch extraction/alignment, dense G3T-style camera-level pointmap target generation, full G3T camera-level pose target generation, replacing the G3T/VGGT adapter template with concrete head calls, and GPU environment validation.

## Module Status

| Area | Status | Evidence |
| --- | --- | --- |
| Git repository | Ready locally | local commits are present; use `git log --oneline` for the current head |
| Project status audit | Scripted | `scripts/audit_project_status.py` reports scaffold readiness and remaining real-training gaps |
| Reference code | Ready locally | `refs/g3t`, `refs/look-from-above-components/PseudoMapTrainer`, `refs/look-from-above-components/MapTR` |
| Reference setup | Scripted | `scripts/setup_references.py --dry-run` prints clone plans for ignored external repos |
| Data contracts | Scaffolded | `src/vggt_project/data/sample.py`, `src/vggt_project/data/synthetic.py`, `src/vggt_project/data/manifest.py`, `src/vggt_project/data/manifest_tensor_dataset.py`, `src/vggt_project/data/manifest_preview.py`; manifest batches include `camera_images` plus optional `target_camera_pointmaps` and `target_camera_local_camera_to_gravity_poses` for camera-specific scaffold heads; manifest preview writes a JSON summary and contact sheet for pre-training sanity checks |
| Real nuScenes data loading | Partial scaffold | layout inspection, JSONL manifest generation/loading with ego pose, map location metadata, and optional camera-level pose targets, path validation, smoke satellite/mask asset materialization, local raster satellite config validation/crop materialization, single- or multi-camera LiDAR-projected camera depth target generation, LiDAR-to-ego and LiDAR-to-camera pointmap target generation, calibration-derived camera pose target generation, real-file smoke tensor loading, optional depth/mask/sample-level pointmap/camera-level pointmap target loading, and ego-pose-derived, explicit, or calibration-derived camera-level pose targets exist; real satellite raster sourcing, dense G3T-style pointmap target generation, full G3T camera-pose targets, and G3T/VGGT camera-pose adapter wiring still need implementation |
| Model framework | Scaffolded | `src/vggt_project/models/scaffold.py`, `src/vggt_project/models/factory.py`, `src/vggt_project/models/adapter_contract.py`, and `adapters/g3t_vggt_adapter.py`; includes shared BEV/satellite latent heads plus camera-specific depth, pointmap, and local pose heads for manifest batches, factory-based model selection, a repo-local adapter template, and a camera-aware adapter contract probe |
| G3T/VGGT integration | Partial plumbing | reference code exists; train/eval can load an external adapter module and optional weights through config; `adapters/g3t_vggt_adapter.py` is a smoke-trainable template with reference-output mapping for G3T/VGGT-style `depth`, `world_points`, and pose encodings plus config-level selection of a local `refs/g3t` G3T/VGGT builder; the reference wrapper can load raw or wrapper-prefixed checkpoint state dicts through `load_project_weights`; real public checkpoint format validation and full real-data head calls still need implementation |
| Losses | Scaffolded | pointmap, single- or multi-camera depth, local pose, relative pose losses in `src/vggt_project/losses.py`; multi-camera depth, pointmap, and local pose losses use camera-specific predictions when available |
| Evaluation metrics | Reconstruction-first scaffolded | single- or multi-camera depth MAE, pointmap L1, scale-aligned pointmap accuracy/completeness/chamfer, gravity angular error, local pose L2, sequence translation drift, and relative pose L2 in `src/vggt_project/metrics.py`; multi-camera depth, pointmap, and local pose metrics use camera-specific predictions when available |
| Train loop | Smoke-verified scaffold | synthetic train and manifest-smoke train completed in `/tmp/vggt-satellite-smoke` |
| Eval/inference loop | Smoke-verified scaffold | synthetic eval completed against `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt`; manifest-smoke eval is implemented and covered by tests |
| Experiment config dispatch | Scripted | `configs/reconstruction_first.yaml` can drive current scaffold train/eval through `scripts/train.py --config ...` and `scripts/evaluate.py --config ...`; `runtime.data.train_manifest_path` and `runtime.data.eval_manifest_path` support train/eval split manifests; `runtime.model` supports `scaffold` or external adapter module paths; `runtime.device` supports explicit `cuda`/`mps`/`cpu` selection; `runtime.seed` supports reproducible scaffold training runs |
| Experiment report pipeline | Scripted | `scripts/run_experiment.py` runs config-driven train+eval and writes a JSON report |
| End-to-end smoke pipeline | Smoke-verified scaffold | `scripts/run_smoke_pipeline.py` creates toy image/depth/mask files, trains, and evaluates |
| Environment setup | Smoke-verified script | `requirements.txt`, `environment.yml`, `scripts/setup_env.sh`; Python 3.10/3.11 recommended; PyTorch 2.12.0 imported |
| Training readiness | Scripted | `scripts/check_training_readiness.py`; checks config manifests, optional satellite raster config readiness, external adapter/weight paths, core dependencies, and requested device availability |
| Training run planning | Scripted | `scripts/plan_training_run.py`; prints ordered commands from nuScenes manifest generation through satellite crops, camera pose targets, LiDAR supervision, manifest sample preview, split manifests, readiness, adapter contract checks, train, and eval |
| G3T weight download | Scripted | `scripts/download_weights.py` |
| nuScenes download/setup | Partially scripted | `scripts/prepare_nuscenes.sh`; real download requires account/license |
| Manifest asset materialization | Smoke-only scripted | `scripts/materialize_manifest_assets.py`; creates placeholder satellite patches and optional valid masks for pipeline testing |
| Satellite raster config/crop materialization | Scripted | `configs/satellite_rasters.example.json`, `scripts/check_satellite_rasters.py`, and `scripts/materialize_satellite_crops.py`; validates raster paths/map-location coverage and crops patches from user-provided local satellite rasters using ego pose metadata |
| LiDAR depth target generation | Scripted for one or more cameras | `scripts/generate_lidar_depth_targets.py`; projects `LIDAR_TOP` to `CAM_FRONT` by default and accepts repeated/comma-separated `--camera` values |
| LiDAR pointmap target generation | Scripted | `scripts/generate_lidar_pointmap_targets.py` transforms `LIDAR_TOP` points into ego-frame `.npy` pointmap targets; `scripts/generate_camera_lidar_pointmap_targets.py` transforms visible LiDAR points into camera-frame `pointmap_paths` targets |
| LiDAR supervision pipeline | Scripted | `scripts/generate_lidar_supervision.py`; generates single- or multi-camera depth plus ego-frame or camera-frame pointmap targets and writes one final supervised manifest |
| Manifest train/eval split | Scripted | `scripts/split_manifest.py`; splits JSONL manifests by `scene_token` to avoid scene leakage |
| GitHub upload | Published | `origin` is `https://github.com/pikangqiu/3dvgT.git`; local `main` tracks `origin/main` |
| GitHub publish preflight | Scripted | `scripts/check_github_publish.py` reports worktree, branch, remote, and whether `gh auth` is needed for repo creation |
| GitHub CI | Scripted | `.github/workflows/ci.yml` runs lightweight tests, project audit, reference dry-run, and compile checks |
| Experiment protocol | Scripted | `scripts/list_experiment_protocol.py`; prints primary metrics, auxiliary metrics, baseline table rows, and benchmark roles as text or JSON |

## Commands

Verified locally:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/audit_project_status.py
PYTHONPATH=src python3 scripts/check_training_readiness.py --help
PYTHONPATH=src python3 scripts/plan_training_run.py --help
PYTHONPATH=src python3 scripts/check_model_adapter.py --help
PYTHONPATH=src python3 scripts/list_experiment_protocol.py --help
PYTHONPATH=src python3 scripts/inspect_manifest_sample.py --help
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/setup_references.py --dry-run
PYTHONPATH=src python3 scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py --help
PYTHONPATH=src python3 scripts/check_satellite_rasters.py --help
PYTHONPATH=src python3 scripts/generate_lidar_depth_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_pointmap_targets.py --help
PYTHONPATH=src python3 scripts/generate_camera_lidar_pointmap_targets.py --help
PYTHONPATH=src python3 scripts/generate_camera_pose_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_supervision.py --help
PYTHONPATH=src python3 scripts/split_manifest.py --help
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

Already published to GitHub:

```bash
git remote -v
git push
```

## Next Implementation Milestones

1. Replace placeholder satellite materialization with actual satellite patch extraction.
2. Generate dense G3T/VGGT camera-level pointmap and pose targets instead of relying on sparse LiDAR camera pointmaps and coarse ego-pose targets.
3. Select and implement satellite patch source/alignment.
4. Replace `adapters/g3t_vggt_adapter.py` internals with concrete G3T/VGGT head calls and camera-specific pose heads.
5. Calibrate benchmark protocols and thresholds for final tables once real G3T/VGGT outputs are connected.
6. Add optional MapTR-style vector map auxiliary supervision.
