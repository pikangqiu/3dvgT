# Current Code Status

## Summary

The repository is no longer only a paper/reference folder. It now contains a project scaffold with data contracts, a minimal trainable model scaffold with camera-specific depth and pointmap prediction plus optional BEV occupancy prediction, external adapter module loading, a repo-local G3T/VGGT adapter template, config-level local G3T/VGGT reference adapter selection, config-level reference constructor kwargs, reference-adapter checkpoint loading hooks, configurable fine-tuning policies, model adapter contract checks, manifest forward probing, reconstruction losses, train/eval entrypoints, reconstruction-first metrics, config-driven experiment dispatch and reporting, an end-to-end smoke pipeline, project-status audit scripts, environment/readiness scripts, external-asset readiness checks, combined real-training preflight reports, ordered training run-plan generation, combined launch readiness packet generation, dry-run/executable training bootstrap, external-reference setup scripts, model-weight preparation, JSON config weight wiring, weight-download and checkpoint-inspection scripts, dataset preparation notes, satellite raster config setup entrypoint, Occ3D/OpenOccupancy benchmark dataset setup entrypoint, public occupancy label attachment, semantic occupancy prediction export, benchmark evaluation for exported arrays, manifest asset materialization, manifest sample preview, satellite raster config validation, multi-camera nuScenes LiDAR-depth target generation and loss/metric wiring, nuScenes ego-frame and camera-frame LiDAR pointmap target generation, LiDAR-derived BEV occupancy proxy target generation, dense G3T/VGGT reference prediction target materialization, combined LiDAR supervision manifest generation, scene-level manifest splitting, GitHub publishing, research baseline notes, and a machine-readable experiment protocol.

It is not yet a complete real nuScenes training codebase. The current train/eval paths are synthetic smoke training/evaluation plus manifest-smoke training that can load real camera, satellite, single- or multi-camera depth, mask, sample-level or camera-level pointmap, LiDAR-derived occupancy proxy, and ego-pose-derived, explicit, calibration-derived, or reference-predicted camera-level pose target tensors from a JSONL manifest. Real training still needs real satellite patch extraction/alignment, real public G3T/VGGT checkpoint and GPU validation, concrete real-checkpoint fine-tuning validation, and a successful real-asset manifest forward probe.

## Module Status

| Area | Status | Evidence |
| --- | --- | --- |
| Git repository | Ready locally | local commits are present; use `git log --oneline` for the current head |
| Project status audit | Scripted | `scripts/audit_project_status.py` reports scaffold readiness, remaining real-training gaps, and `next_actions` commands |
| Reference code | Ready locally | `refs/g3t`, `refs/look-from-above-components/PseudoMapTrainer`, `refs/look-from-above-components/MapTR` |
| Reference setup | Scripted | `scripts/setup_references.py --dry-run` prints clone plans for ignored external repos, including optional benchmark references for DGGT, DrivingForward, GaussianOcc, OpenScene, UniOcc, and Sat3DGen |
| Data contracts | Scaffolded | `src/vggt_project/data/sample.py`, `src/vggt_project/data/synthetic.py`, `src/vggt_project/data/manifest.py`, `src/vggt_project/data/manifest_tensor_dataset.py`, `src/vggt_project/data/manifest_preview.py`; manifest batches include `camera_images` plus optional `target_camera_pointmaps`, `target_occupancy`, and `target_camera_local_camera_to_gravity_poses` for camera-specific/auxiliary scaffold heads; manifest preview writes a JSON summary and contact sheet for pre-training sanity checks |
| Real nuScenes data loading | Partial scaffold | layout inspection, JSONL manifest generation/loading with ego pose, map location metadata, and optional camera-level pose targets, path validation, smoke satellite/mask asset materialization, local raster satellite config validation/crop materialization, single- or multi-camera LiDAR-projected camera depth target generation, LiDAR-to-ego and LiDAR-to-camera pointmap target generation, LiDAR-to-BEV occupancy target generation, dense reference-predicted depth/pointmap/pose target materialization, calibration-derived camera pose target generation, real-file smoke tensor loading, optional image or `.npy` depth/mask/occupancy/sample-level pointmap/camera-level pointmap target loading, and ego-pose-derived, explicit, calibration-derived, or reference-predicted camera-level pose targets exist; real satellite raster sourcing and full real-checkpoint validation still need implementation |
| Model framework | Scaffolded | `src/vggt_project/models/scaffold.py`, `src/vggt_project/models/factory.py`, `src/vggt_project/models/adapter_contract.py`, and `adapters/g3t_vggt_adapter.py`; includes shared BEV/satellite latent heads plus camera-specific depth, pointmap, and local pose heads for manifest batches, factory-based model selection, a repo-local adapter template, and a camera-aware adapter contract probe with trainable/frozen parameter summaries |
| G3T/VGGT integration | Partial plumbing | reference code exists; train/eval can load an external adapter module and optional weights through config; `adapters/g3t_vggt_adapter.py` is a smoke-trainable template with reference-output mapping for G3T/VGGT-style `depth`, `world_points`, and pose encodings plus config-level selection of a local `refs/g3t` G3T/VGGT builder; `reference_model_kwargs` passes config values such as `img_size` into the selected reference constructor; the reference wrapper can load raw or wrapper-prefixed checkpoint state dicts through `load_project_weights`; `fine_tuning_policy` supports full, frozen-backbone, heads-only, satellite/fusion/head, and reference-frozen adapter training regimes and is used by train/eval plus adapter contract checks; `scripts/generate_reference_supervision_targets.py` can materialize configured-model predictions into manifest targets; `scripts/probe_manifest_forward.py` runs one configured manifest sample through the model and reports input/output shapes; real public checkpoint format validation and successful real-asset head calls still need implementation |
| Losses | Scaffolded | pointmap, single- or multi-camera depth, local pose, relative pose, and optional BEV occupancy losses in `src/vggt_project/losses.py`; multi-camera depth, pointmap, and local pose losses use camera-specific predictions when available |
| Evaluation metrics | Reconstruction-first scaffolded | single- or multi-camera depth MAE, pointmap L1, scale-aligned pointmap accuracy/completeness/chamfer, gravity angular error, local pose L2, sequence translation drift, relative pose L2, and optional BEV occupancy IoU in `src/vggt_project/metrics.py`; multi-camera depth, pointmap, and local pose metrics use camera-specific predictions when available |
| Train loop | Smoke-verified scaffold | synthetic train and manifest-smoke train completed in `/tmp/vggt-satellite-smoke` |
| Eval/inference loop | Smoke-verified scaffold | synthetic eval completed against `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt`; manifest-smoke eval is implemented and covered by tests |
| Experiment config dispatch | Scripted | `configs/reconstruction_first.json` is the dependency-light default config for current scaffold train/eval; `configs/reconstruction_first.yaml` mirrors the same runtime intent in a human-readable research config; `runtime.data.train_manifest_path` and `runtime.data.eval_manifest_path` support train/eval split manifests; `runtime.model` supports `scaffold` or external adapter module paths; `runtime.device` supports explicit `cuda`/`mps`/`cpu` selection; `runtime.seed` supports reproducible scaffold training runs |
| Experiment report pipeline | Scripted | `scripts/run_experiment.py` runs config-driven train+eval and writes a JSON report |
| End-to-end smoke pipeline | Smoke-verified scaffold | `scripts/run_smoke_pipeline.py` creates toy image/depth/mask files, trains, and evaluates |
| Environment setup | Smoke-verified script | `requirements.txt`, `environment.yml`, `scripts/setup_env.sh`; Python 3.10/3.11 recommended; PyTorch 2.12.0 imported |
| Training readiness | Scripted | `scripts/check_training_readiness.py`; checks configured manifests are non-empty and parseable with existing referenced file paths, output/checkpoint consistency, optional satellite raster config readiness, external adapter/reference-root paths, concrete `.pt`/`.pth`/`.bin` weight file selection and loadability, core dependencies, and requested device availability |
| External asset readiness | Scripted | `scripts/check_external_assets.py`; checks required real-training assets at a coarse filesystem level: nuScenes layout, satellite raster config, concrete model checkpoint, and optional Occ3D/OpenOccupancy labels, then prints next setup actions |
| Real-training preflight | Scripted | `scripts/report_real_training_preflight.py`; combines external-asset readiness with the launch packet and emits one JSON/text report with `ready_for_real_training` plus deduplicated next actions |
| Training launch packet | Scripted | `scripts/report_training_launch.py` combines readiness status, ordered run plan, blockers, remediation commands, and next commands as text or JSON so a real training machine can fail fast before launching |
| Training run planning | Scripted | `scripts/plan_training_run.py` prints ordered commands from nuScenes manifest generation through satellite crops, camera pose targets, LiDAR supervision, occupancy target generation, optional reference supervision when a G3T/VGGT reference adapter is configured, manifest sample preview, split manifests, train/eval manifest non-empty path validation, optional checkpoint inspection, readiness, adapter contract checks, one-sample manifest forward probe, train with metrics JSON export, eval with metrics JSON export, optional Occ3D label attachment, occupancy prediction export, occupancy benchmark evaluation, and final artifact verification; `scripts/bootstrap_training_run.py` can dry-run or execute those commands up to a selected step without skipping launch checks once manifests exist |
| G3T/VGGT weight preparation | Scripted | `scripts/prepare_model_weights.sh` creates the checkpoint root, dry-runs by default, calls `scripts/download_weights.py` without `--dry-run` only when the wrapper receives `--download`, and points users to `scripts/inspect_checkpoint.py` plus `scripts/configure_model_weights.py`; `scripts/configure_model_weights.py` writes a concrete `.pt`/`.pth`/`.bin` file into JSON experiment configs without hand-editing; `scripts/download_weights.py` supports Hugging Face revision selection and repeated `--allow-pattern`; `scripts/inspect_checkpoint.py` summarizes downloaded checkpoint container keys, tensor prefixes, shapes, and dtypes before adapter loading |
| nuScenes and Occ3D download/setup | Partially scripted | `scripts/prepare_nuscenes.sh` and `scripts/prepare_occ3d.sh`; real download requires account/license or user-provided approved archive URLs |
| Manifest asset materialization | Smoke-only scripted | `scripts/materialize_manifest_assets.py`; creates placeholder satellite patches and optional valid masks for pipeline testing |
| Satellite raster config/crop materialization | Scripted | `scripts/prepare_satellite_rasters.sh`, `configs/satellite_rasters.example.json`, `scripts/check_satellite_rasters.py`, and `scripts/materialize_satellite_crops.py`; creates the local config entrypoint, validates raster paths/map-location coverage, and crops patches from user-provided local satellite rasters using ego pose metadata, rejecting crop windows that fall outside raster bounds |
| LiDAR depth target generation | Scripted for one or more cameras | `scripts/generate_lidar_depth_targets.py`; projects `LIDAR_TOP` to `CAM_FRONT` by default and accepts repeated/comma-separated `--camera` values |
| LiDAR pointmap target generation | Scripted | `scripts/generate_lidar_pointmap_targets.py` transforms `LIDAR_TOP` points into ego-frame `.npy` pointmap targets; `scripts/generate_camera_lidar_pointmap_targets.py` transforms visible LiDAR points into camera-frame `pointmap_paths` targets |
| LiDAR occupancy target generation | Scripted | `scripts/generate_lidar_occupancy_targets.py` rasterizes ego-frame `LIDAR_TOP` points into binary BEV `.npy` occupancy targets and writes `occupancy_path` into manifests |
| Occupancy benchmark export/evaluation | Scripted | `scripts/attach_occ3d_labels.py`, `src/vggt_project/data/occ3d_labels.py`, `scripts/export_occupancy_predictions.py`, `src/vggt_project/data/occupancy_predictions.py`, `scripts/evaluate_occupancy_benchmark.py`, and `src/vggt_project/occupancy_benchmark.py`; attaches public Occ3D/OpenOccupancy `labels.npz` paths to eval manifests as `occupancy_path`, exports model BEV occupancy predictions to `predicted_occupancy_path`, then computes class IoU and semantic `occupancy_miou` from manifest-paired prediction/target arrays while keeping public metrics separate from local LiDAR-proxy `bev_occupancy_iou` |
| Training artifact verification | Scripted | `scripts/verify_training_artifacts.py` and `src/vggt_project/training_artifacts.py`; verifies checkpoint existence, train/eval metrics JSON, optional occupancy benchmark JSON, and required metrics before a run is recorded in experiment tables |
| LiDAR supervision pipeline | Scripted | `scripts/generate_lidar_supervision.py`; generates single- or multi-camera depth plus ego-frame or camera-frame pointmap targets and writes one final supervised manifest |
| Reference supervision materialization | Scripted | `scripts/generate_reference_supervision_targets.py`; runs the configured model/adapter over a manifest and writes dense `.npy` depth, pointmap, and camera pose targets back into manifest-compatible fields |
| Manifest train/eval split | Scripted | `scripts/split_manifest.py`; splits JSONL manifests by `scene_token` to avoid scene leakage and rejects empty train/eval outputs |
| GitHub upload | Published | `origin` is `https://github.com/pikangqiu/3dvgT.git`; local `main` tracks `origin/main` |
| GitHub publish preflight | Scripted | `scripts/check_github_publish.py` reports worktree, branch, remote, and whether `gh auth` is needed for repo creation |
| GitHub CI | Scripted | `.github/workflows/ci.yml` runs lightweight tests, project audit, reference dry-run, and compile checks |
| Experiment protocol | Scripted | `scripts/list_experiment_protocol.py`; prints primary metrics, auxiliary metrics, baseline table rows, benchmark roles, and `experiment_phase` priorities as text or JSON; current protocol separates primary reconstruction, satellite/cross-view reconstruction, driving reconstruction, and occupancy auxiliary comparisons, including DynamicVGGT, PAGE-4D, and Sat3DGen as 2026 tracking/stretch baselines |

## Commands

Verified locally:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/audit_project_status.py
PYTHONPATH=src python3 scripts/check_training_readiness.py --help
PYTHONPATH=src python3 scripts/report_real_training_preflight.py --help
PYTHONPATH=src python3 scripts/report_training_launch.py --help
PYTHONPATH=src python3 scripts/plan_training_run.py --help
PYTHONPATH=src python3 scripts/bootstrap_training_run.py --help
PYTHONPATH=src python3 scripts/check_model_adapter.py --help
PYTHONPATH=src python3 scripts/probe_manifest_forward.py --help
PYTHONPATH=src python3 scripts/verify_training_artifacts.py --help
PYTHONPATH=src python3 scripts/attach_occ3d_labels.py --help
PYTHONPATH=src python3 scripts/export_occupancy_predictions.py --help
PYTHONPATH=src python3 scripts/evaluate_occupancy_benchmark.py --help
PYTHONPATH=src python3 scripts/list_experiment_protocol.py --help
PYTHONPATH=src python3 scripts/inspect_checkpoint.py --help
PYTHONPATH=src python3 scripts/inspect_manifest_sample.py --help
PYTHONPATH=src python3 scripts/check_references.py
PYTHONPATH=src python3 scripts/setup_references.py --dry-run
PYTHONPATH=src python3 scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
PYTHONPATH=src python3 scripts/materialize_manifest_assets.py --help
PYTHONPATH=src python3 scripts/check_satellite_rasters.py --help
PYTHONPATH=src python3 scripts/generate_lidar_depth_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_pointmap_targets.py --help
PYTHONPATH=src python3 scripts/generate_camera_lidar_pointmap_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_occupancy_targets.py --help
PYTHONPATH=src python3 scripts/generate_camera_pose_targets.py --help
PYTHONPATH=src python3 scripts/generate_lidar_supervision.py --help
PYTHONPATH=src python3 scripts/generate_reference_supervision_targets.py --help
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
PYTHONPATH=src python scripts/train.py --config configs/reconstruction_first.json --metrics-output outputs/manifest-smoke/train_metrics.json
PYTHONPATH=src python scripts/evaluate.py --config configs/reconstruction_first.json --metrics-output outputs/manifest-smoke/eval_metrics.json
PYTHONPATH=src python scripts/verify_training_artifacts.py --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt --train-metrics outputs/manifest-smoke/train_metrics.json --eval-metrics outputs/manifest-smoke/eval_metrics.json --required-eval-metric loss --required-eval-metric depth_mae
PYTHONPATH=src python scripts/run_experiment.py --config configs/reconstruction_first.json --report outputs/reconstruction_first_report.json
PYTHONPATH=src python scripts/run_smoke_pipeline.py --output-dir outputs/smoke-pipeline --epochs 1
PYTHONPATH=src python scripts/generate_reference_supervision_targets.py --config configs/reconstruction_first.json --manifest data/manifests/nuscenes-mini.supervised.jsonl --output data/manifests/nuscenes-mini.reference.jsonl
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
2. Validate dense G3T/VGGT camera-level pointmap and pose target generation with a real public checkpoint on GPU.
3. Select and implement satellite patch source/alignment.
4. Replace `adapters/g3t_vggt_adapter.py` internals with concrete G3T/VGGT head calls and camera-specific pose heads.
5. Calibrate benchmark protocols and thresholds for final tables once real G3T/VGGT outputs are connected.
6. Add optional MapTR-style vector map auxiliary supervision.
