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
PYTHONPATH=src python scripts/check_references.py
```

## Weights

Download G3T weights:

```bash
python scripts/download_weights.py --repo-id thatbrguy/g3t --output-dir checkpoints/g3t
```

VGGT/MapTR/PseudoMapTrainer weights should be downloaded from their own reference repositories when those branches are enabled.

## Smoke Training

The current training entrypoint supports a synthetic smoke test:

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

Generate a LiDAR-projected camera depth target:

```bash
PYTHONPATH=src python scripts/generate_lidar_depth_targets.py \
  data/manifests/nuscenes-mini.smoke.jsonl \
  --root data/nuscenes \
  --version v1.0-mini \
  --camera CAM_FRONT \
  --output data/manifests/nuscenes-mini.depth.jsonl
PYTHONPATH=src python scripts/validate_manifest.py data/manifests/nuscenes-mini.depth.jsonl
```

Smoke-test real file loading:

```bash
PYTHONPATH=src python scripts/train.py \
  --mode manifest-smoke \
  --manifest data/manifests/nuscenes-mini.depth.jsonl \
  --epochs 1 \
  --output-dir outputs/manifest-smoke
PYTHONPATH=src python scripts/evaluate.py \
  --mode manifest-smoke \
  --manifest data/manifests/nuscenes-mini.depth.jsonl \
  --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt
```

If `lidar_depth_path` or `valid_area_mask_path` fields are present in the manifest, `manifest-smoke` loads them as target tensors. Pointmap and pose targets are still placeholders until the G3T/VGGT supervision adapter is implemented.
