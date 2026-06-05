# Setup

## Environment

Use Python 3.10 or 3.11 for the training environment. The local macOS default Python may be newer than the PyTorch wheels supported by the project; `scripts/setup_env.sh` checks this before installing.

Use the scripted setup for a local virtual environment:

```bash
bash scripts/setup_env.sh .venv
source .venv/bin/activate
```

If needed:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_env.sh .venv
```

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

## nuScenes Layout Check

After downloading nuScenes:

```bash
PYTHONPATH=src python scripts/check_nuscenes.py --root data/nuscenes --version v1.0-mini
```
