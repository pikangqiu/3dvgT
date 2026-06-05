# Dataset Preparation

## nuScenes

Run:

```bash
bash scripts/prepare_nuscenes.sh
```

The script creates the expected root directory and prints the required manual download steps. The actual dataset cannot be downloaded automatically without a nuScenes account and license acceptance.

Expected layout:

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/ or v1.0-trainval/
```

The first real training adapter should consume:

- six synchronized camera images,
- camera calibration,
- ego pose,
- optional LiDAR/depth proxy,
- aligned satellite patch,
- valid-area mask,
- optional vector map ground truth.

## Satellite Patches

The project still needs a concrete satellite source. Until that is selected, the dataset adapter should expose `satellite_patch_path` and keep satellite alignment logic isolated from the model.

