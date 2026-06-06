# Setup

## Environment

Use Python 3.10 or 3.11 for the training environment. The local macOS default Python may be newer than the PyTorch wheels supported by the project; `scripts/setup_env.sh` checks this before installing.

Use the scripted setup for a local virtual environment:

```bash
bash scripts/setup_env.sh .venv
source .venv/bin/activate
export MPLCONFIGDIR=.venv/.matplotlib
```

If needed:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_env.sh .venv
```

`MPLCONFIGDIR` matters because `nuscenes-devkit` imports matplotlib, which may otherwise try to write cache files under a non-writable home directory.

For a conda workflow:

```bash
conda env create -f environment.yml
conda activate vggt-satellite
pip install -e .
```

## Reference Code

Reference repositories are stored under `refs/` and are ignored by the main git repository:

- `refs/g3t`
- `refs/look-from-above-components/PseudoMapTrainer`
- `refs/look-from-above-components/MapTR`

Run:

```bash
PYTHONPATH=src python scripts/audit_project_status.py
PYTHONPATH=src python scripts/check_references.py
```

After cloning the main project from GitHub, restore ignored external references with:

```bash
PYTHONPATH=src python scripts/setup_references.py --dry-run
PYTHONPATH=src python scripts/setup_references.py
```

The dry run prints the `git clone` commands without touching the network. The real run clones missing repositories under `refs/`.

Before publishing to GitHub, check the local publishing state:

```bash
PYTHONPATH=src python scripts/check_github_publish.py
```

This repository already tracks `https://github.com/pikangqiu/3dvgT.git`, so a clean worktree can be pushed with `git push`. If a fresh checkout has no `origin`, run `gh auth login` and then publish with `bash scripts/publish_github.sh VggT`.

## Weights

Download G3T weights:

```bash
python scripts/download_weights.py --repo-id thatbrguy/g3t --output-dir checkpoints/g3t
```

VGGT/MapTR/PseudoMapTrainer weights should be downloaded from their own reference repositories when those branches are enabled.

## Smoke Training

The fastest end-to-end check creates a toy manifest, trains from image files, and evaluates the produced checkpoint:

```bash
PYTHONPATH=src python scripts/run_smoke_pipeline.py --output-dir outputs/smoke-pipeline --epochs 1
```

This is an environment and plumbing check. It is not a real experiment because the images and depth/mask targets are generated toy data.

The current training entrypoint also supports a synthetic smoke test:

```bash
PYTHONPATH=src python scripts/train.py --mode synthetic --epochs 1
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/synthetic/synthetic_scaffold.pt
```

This checks the model/loss/train/eval plumbing. Real nuScenes training still requires implementing the dataset adapter and selecting the concrete G3T/VGGT heads to fine-tune.

Verified smoke environment:

- Python: `/Users/eve_kang/.local/bin/python3.11`
- Temporary venv: `/tmp/vggt-satellite-smoke`
- Torch import: `2.12.0`
- Synthetic train/eval completed with a checkpoint at `/tmp/vggt-satellite-smoke-output/synthetic_scaffold.pt`

## nuScenes Layout Check

After downloading nuScenes:

```bash
PYTHONPATH=src python scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
```

## Real Data Manifest

Once nuScenes and satellite patches are prepared, create a JSONL manifest and load it with:

```python
from pathlib import Path
from vggt_project.data import load_manifest

samples = load_manifest(Path("data/manifests/nuscenes-mini.jsonl"))
```

Generate the manifest with:

```bash
PYTHONPATH=src python scripts/generate_manifest.py --root data/nuscenes --version v1.0-mini
```

Validate it before training:

```bash
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.jsonl
```

For smoke testing before real satellite crops exist:

```bash
PYTHONPATH=src python scripts/materialize_manifest_assets.py \
  data/manifests/nuscenes-mini.jsonl \
  --create-valid-masks \
  --output data/manifests/nuscenes-mini.smoke.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.smoke.jsonl
```

If local satellite rasters are available, replace placeholder patches with real crops:

```bash
mkdir -p data/satellite_rasters
cp configs/satellite_rasters.example.json data/satellite_rasters/config.json
PYTHONPATH=src python scripts/check_satellite_rasters.py \
  --config data/satellite_rasters/config.json \
  --manifest data/manifests/nuscenes-mini.jsonl
PYTHONPATH=src python scripts/materialize_satellite_crops.py \
  data/manifests/nuscenes-mini.jsonl \
  --config data/satellite_rasters/config.json \
  --output data/manifests/nuscenes-mini.satellite.jsonl
```

Generate LiDAR-projected camera depth targets:

```bash
PYTHONPATH=src python scripts/generate_lidar_depth_targets.py \
  data/manifests/nuscenes-mini.smoke.jsonl \
  --root data/nuscenes \
  --version v1.0-mini \
  --camera CAM_FRONT \
  --camera CAM_BACK \
  --output data/manifests/nuscenes-mini.depth.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.depth.jsonl
```

Generate a LiDAR ego-frame pointmap target:

```bash
PYTHONPATH=src python scripts/generate_lidar_pointmap_targets.py \
  data/manifests/nuscenes-mini.depth.jsonl \
  --root data/nuscenes \
  --version v1.0-mini \
  --output data/manifests/nuscenes-mini.pointmap.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.pointmap.jsonl
```

Generate camera-frame pointmap targets for camera-specific supervision:

```bash
PYTHONPATH=src python scripts/generate_camera_lidar_pointmap_targets.py \
  data/manifests/nuscenes-mini.depth.jsonl \
  --root data/nuscenes \
  --version v1.0-mini \
  --camera CAM_FRONT \
  --camera CAM_BACK \
  --pointmap-dir camera_pointmaps \
  --output data/manifests/nuscenes-mini.camera-pointmap.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.camera-pointmap.jsonl
```

Alternatively, generate depth and pointmap supervision in one pass:

```bash
PYTHONPATH=src python scripts/generate_lidar_supervision.py \
  data/manifests/nuscenes-mini.smoke.jsonl \
  --root data/nuscenes \
  --version v1.0-mini \
  --camera CAM_FRONT \
  --camera CAM_BACK \
  --pointmap-target-frame camera \
  --pointmap-dir camera_pointmaps \
  --output data/manifests/nuscenes-mini.supervised.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.supervised.jsonl
```

Optionally replace or augment sparse LiDAR targets with configured G3T/VGGT reference predictions:

```bash
PYTHONPATH=src python scripts/generate_reference_supervision_targets.py \
  --config configs/reconstruction_first.yaml \
  --manifest data/manifests/nuscenes-mini.supervised.jsonl \
  --target-dir reference_targets \
  --max-points 4096 \
  --output data/manifests/nuscenes-mini.reference.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.reference.jsonl
```

Split the supervised manifest at the scene level:

```bash
PYTHONPATH=src python scripts/split_manifest.py \
  data/manifests/nuscenes-mini.supervised.jsonl \
  --train-output data/manifests/nuscenes-mini.train.jsonl \
  --eval-output data/manifests/nuscenes-mini.val.jsonl \
  --eval-fraction 0.2 \
  --seed 0
```

Smoke-test real file loading:

```bash
PYTHONPATH=src python scripts/train.py \
  --mode manifest-smoke \
  --manifest data/manifests/nuscenes-mini.train.jsonl \
  --epochs 1 \
  --seed 0 \
  --output-dir outputs/manifest-smoke
PYTHONPATH=src python scripts/evaluate.py \
  --mode manifest-smoke \
  --manifest data/manifests/nuscenes-mini.val.jsonl \
  --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt
```

The same current scaffold can be launched from the default experiment config after the manifest referenced in that config exists:

```bash
PYTHONPATH=src python scripts/train.py --config configs/reconstruction_first.yaml
PYTHONPATH=src python scripts/evaluate.py --config configs/reconstruction_first.yaml
```

`configs/reconstruction_first.yaml` supports either one shared manifest or separate train/eval manifests:

```yaml
runtime:
  device: null
  seed: 0
  data:
    manifest_path: data/manifests/nuscenes-mini.supervised.jsonl
    train_manifest_path: data/manifests/nuscenes-train.supervised.jsonl
    eval_manifest_path: data/manifests/nuscenes-val.supervised.jsonl
    satellite_raster_config_path: data/satellite_rasters/config.json
  model:
    family: scaffold
    adapter_module_path: null
    weights_path: null
    strict_weights: true
    freeze_backbone: false
    use_reference_adapter: false
    reference_root: refs/g3t
    reference_model: g3t
    reference_model_kwargs:
      img_size: 518
```

Set `runtime.device` to `cuda`, `mps`, or `cpu` to force a specific training/evaluation device. Leave it as `null` to let PyTorch auto-select CUDA when available, otherwise CPU.
Set `runtime.seed` or pass `--seed` to make model initialization and shuffled DataLoader order reproducible for the current scaffold runs. The experiment report records this seed in its serialized config.

To switch from the scaffold to a future G3T/VGGT adapter, set `runtime.model.family` to `external`, `g3t`, `vggt`, or `g3t-vggt`, and point `adapter_module_path` at a Python file that defines `build_model(point_count, **kwargs)`. The returned torch module must emit the same prediction keys as the scaffold: `gravity_aligned_pointmap`, `depth`, `local_camera_to_gravity_pose`, and `relative_yaw_translation`, with optional `camera_depths` and `camera_pointmaps`.

```yaml
runtime:
  model:
    family: g3t-vggt
    adapter_module_path: adapters/g3t_vggt_adapter.py
    weights_path: weights/g3t/model.pt
    strict_weights: false
    freeze_backbone: false
    use_reference_adapter: true
    reference_root: refs/g3t
    reference_model: g3t
    reference_model_kwargs:
      img_size: 518
      enable_point: true
      enable_depth: true
      enable_gravity_camera_heads: true
```

`adapters/g3t_vggt_adapter.py` is currently a smoke-trainable template that satisfies the project contract and exposes `freeze_backbone()`. When `use_reference_adapter` is true, the same adapter instantiates a local `refs/g3t` `G3T` or `VGGT` model and maps its reference outputs back into the project contract. `reference_model_kwargs` is passed directly to the selected reference constructor, so use G3T keys such as `img_size`, `enable_point`, `enable_depth`, and `enable_gravity_camera_heads`, or VGGT keys such as `img_size`, `enable_camera`, `enable_point`, `enable_depth`, and `enable_track`. If `weights_path` is set, the reference wrapper can load raw reference-model state dicts or wrapper-prefixed state dicts through `load_project_weights`. It remains the intended replacement point for concrete G3T/VGGT head calls once that integration is implemented.

For a single train+eval run with a JSON report:

```bash
PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/reconstruction_first.yaml \
  --report outputs/reconstruction_first_report.json
```

Before launching a real run, check the environment/config readiness:

```bash
PYTHONPATH=src python scripts/plan_training_run.py \
  --config configs/reconstruction_first.yaml
PYTHONPATH=src python scripts/check_training_readiness.py \
  --config configs/reconstruction_first.yaml
```

The run-plan command prints the ordered commands needed to produce missing manifests and supervision files. The readiness check validates configured manifests, dependencies, requested device, the optional `satellite_raster_config_path`, and external adapter/weight paths if `runtime.model.family` is not `scaffold`.

If `lidar_depth_path`, `lidar_depth_paths`, `valid_area_mask_path`, `pointmap_path`, or `pointmap_paths` fields are present in the manifest, `manifest-smoke` loads them as target tensors. Depth targets can be image files or `.npy` arrays. If `ego_translation` and `ego_rotation` are present, it also builds coarse ego-pose-derived targets; if `camera_local_camera_to_gravity_poses` is present, it uses those explicit camera-level pose targets. `scripts/generate_reference_supervision_targets.py` can populate dense configured-model depth, pointmap, and camera pose targets once a scaffold or reference adapter is configured.
