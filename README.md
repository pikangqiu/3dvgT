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
- reconstruction-first metrics for depth, pointmaps, gravity, and drift,
- train and eval entrypoints,
- setup, weight-download, dataset-preparation, and GitHub-publish scripts,
- training environment readiness checks,
- ordered real training run-plan generation,
- satellite raster readiness checks integrated into training preflight,
- external model adapter and weight-path checks integrated into training preflight,
- external reference repository setup script,
- nuScenes manifest generation and loading scaffolds,
- manifest asset materialization for smoke satellite patches and valid masks,
- manifest sample summary/contact-sheet preview for pre-training sanity checks,
- local satellite raster crop materialization,
- satellite raster config template and validation,
- nuScenes LiDAR-to-camera depth target generation for single- or multi-camera manifest supervision,
- multi-camera depth target loading plus loss/metric supervision,
- LiDAR-derived BEV occupancy target generation plus optional occupancy loss/IoU reporting,
- camera-specific depth and pointmap heads in the current scaffold model,
- camera-specific local pose heads and per-camera pose loss/metric routing in the current scaffold model,
- repo-local G3T/VGGT adapter template for config-driven smoke fine-tuning paths,
- G3T/VGGT reference-output mapping utilities for depth, pointmap, and pose contract conversion,
- optional local `refs/g3t` G3T/VGGT reference adapter builder,
- config-level selection of the local `refs/g3t` G3T/VGGT reference adapter,
- config-level G3T/VGGT reference constructor kwargs such as `img_size`,
- reference-adapter checkpoint loading for raw or wrapper-prefixed state dicts,
- model adapter contract check for camera-aware reconstruction outputs,
- nuScenes LiDAR-to-ego pointmap target generation for manifest supervision,
- nuScenes LiDAR-to-camera pointmap target generation for `pointmap_paths` supervision,
- one-command LiDAR depth+pointmap supervision manifest generation,
- dense G3T/VGGT reference prediction materialization into manifest depth, pointmap, and pose targets,
- scene-level train/eval manifest splitting,
- optional sample-level or camera-level manifest pointmap target loading,
- ego-pose-derived pose targets for manifest smoke training,
- explicit manifest `camera_local_camera_to_gravity_poses` loading for camera-level pose supervision,
- calibration-derived nuScenes camera pose target generation,
- manifest smoke training from real image files,
- config-driven train/eval dispatch for the current scaffold,
- config-driven external adapter module loading for future G3T/VGGT fine-tuning,
- train/eval manifest split support in experiment configs,
- explicit train/eval device and seed selection through config or CLI,
- config-driven train+eval experiment report generation,
- one-command toy manifest train/eval smoke pipeline,
- machine-readable baseline/benchmark experiment protocol,
- optional benchmark reference clone plans for DGGT, DrivingForward, GaussianOcc, OpenScene, and UniOcc,
- baseline/benchmark notes for the next experimental plan.

The scaffold is not yet a complete nuScenes training implementation. Real training still needs a concrete satellite patch source/alignment implementation, real public G3T/VGGT checkpoint and GPU validation, concrete G3T/VGGT fine-tuning policies, and full real-data head-call validation.

## Quick Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/audit_project_status.py
PYTHONPATH=src python3 scripts/plan_training_run.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python3 scripts/check_model_adapter.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python3 scripts/list_experiment_protocol.py
PYTHONPATH=src python3 scripts/inspect_manifest_sample.py --help
PYTHONPATH=src python3 scripts/check_training_readiness.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/setup_references.py --dry-run
PYTHONPATH=src python3 scripts/generate_reference_supervision_targets.py --help
```

GitHub remote is already configured. To check or push:

```bash
PYTHONPATH=src python3 scripts/check_github_publish.py
git push
```

If a fresh clone has no `origin`, `scripts/publish_github.sh` can create one with `gh auth login`.

After installing dependencies:

```bash
PYTHONPATH=src python3 scripts/run_smoke_pipeline.py --output-dir outputs/smoke-pipeline --epochs 1
PYTHONPATH=src python3 scripts/train.py --mode synthetic --epochs 1 --seed 0
PYTHONPATH=src python3 scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py data/manifests/nuscenes-mini.jsonl --create-valid-masks --output data/manifests/nuscenes-mini.smoke.jsonl
PYTHONPATH=src python3 scripts/check_satellite_rasters.py --config data/satellite_rasters/config.json --manifest data/manifests/nuscenes-mini.jsonl
PYTHONPATH=src python3 scripts/materialize_satellite_crops.py data/manifests/nuscenes-mini.jsonl --config data/satellite_rasters/config.json --output data/manifests/nuscenes-mini.satellite.jsonl
PYTHONPATH=src python3 scripts/generate_camera_pose_targets.py data/manifests/nuscenes-mini.satellite.jsonl --camera CAM_FRONT --camera CAM_BACK --output data/manifests/nuscenes-mini.pose.jsonl
PYTHONPATH=src python3 scripts/generate_camera_lidar_pointmap_targets.py data/manifests/nuscenes-mini.satellite.jsonl --camera CAM_FRONT --camera CAM_BACK --output data/manifests/nuscenes-mini.camera-pointmaps.jsonl
PYTHONPATH=src python3 scripts/generate_lidar_supervision.py data/manifests/nuscenes-mini.pose.jsonl --camera CAM_FRONT --camera CAM_BACK --output data/manifests/nuscenes-mini.supervised.jsonl
PYTHONPATH=src python3 scripts/generate_lidar_occupancy_targets.py data/manifests/nuscenes-mini.supervised.jsonl --output data/manifests/nuscenes-mini.occupancy.jsonl
PYTHONPATH=src python3 scripts/inspect_manifest_sample.py data/manifests/nuscenes-mini.occupancy.jsonl --output-dir outputs/manifest-preview
PYTHONPATH=src python3 scripts/split_manifest.py data/manifests/nuscenes-mini.occupancy.jsonl --train-output data/manifests/nuscenes-mini.train.jsonl --eval-output data/manifests/nuscenes-mini.val.jsonl
PYTHONPATH=src python3 scripts/train.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.train.jsonl --epochs 1 --seed 0 --output-dir outputs/manifest-smoke
PYTHONPATH=src python3 scripts/evaluate.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.val.jsonl --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt
PYTHONPATH=src python3 scripts/train.py --mode manifest-smoke --manifest data/manifests/nuscenes-mini.train.jsonl --device cuda --epochs 1 --output-dir outputs/manifest-smoke
PYTHONPATH=src python3 scripts/train.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python3 scripts/run_experiment.py --config configs/reconstruction_first.yaml --report outputs/reconstruction_first_report.json
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
