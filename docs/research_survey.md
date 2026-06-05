# Baselines And Benchmarks

## Recommended Primary Benchmark Direction

The project is reconstruction-first, so the strongest first comparison should be against driving-scene reconstruction and occupancy benchmarks rather than pure HD map mAP.

Primary candidates:

- E3D-Bench for general end-to-end 3D geometric foundation model evaluation.
- Occ3D-nuScenes for autonomous-driving 3D occupancy geometry.
- OpenScene for a larger nuPlan-derived occupancy benchmark when scaling beyond nuScenes-mini.
- UniOcc for current-frame and future-frame occupancy prediction/forecasting.
- DrivingForward and DGGT as driving-scene feed-forward reconstruction baselines.
- ReconDrive as a 2026 feed-forward 4D Gaussian Splatting driving-scene reconstruction baseline that extends VGGT-style geometric features.
- ParkRecon3D/ParkGaussian as a 2026 surround-view parking-scene reconstruction benchmark/baseline.
- Perception-aware 3DGS as a 2026 evaluation direction for whether reconstructed driving scenes preserve downstream perception behavior.

Auxiliary mapping candidates:

- MapTR/MapTRv2 for vectorized HD map heads and nuScenes map mAP.
- PseudoMapTrainer for pseudo-label generation without HD maps.
- SG-BEV as a satellite-guided BEV fusion reference, useful for the cross-view feature alignment design even though its task is segmentation rather than 3D reconstruction.

## Baseline Families

### Feed-forward 3D reconstruction

- VGGT: camera/depth/pointmap foundation baseline.
- G3T: gravity-aligned pointmap baseline and our main geometry reference.
- VGG-T3: scalable VGGT-style long-input baseline.
- E3D-Bench model list: broad comparison set for GFMs.

### Driving-scene reconstruction

- DrivingForward: feed-forward Gaussian splatting for nuScenes driving-scene reconstruction.
- DGGT: feed-forward 4D dynamic driving reconstruction on Waymo, nuScenes, and Argoverse2.
- ReconDrive: fast feed-forward 4D Gaussian Splatting for autonomous driving scene reconstruction, explicitly extending VGGT and benchmarking on nuScenes.
- Nighttime Autonomous Driving Scene Reconstruction: 2026 ICRA direction for low-light driving-scene reconstruction on nuScenes/Waymo.
- Real2Sim: physics-driven editable 4DGS for driving scenes, useful as a future simulation/editing comparison rather than first-table baseline.
- Perception-aware 3DGS: evaluates reconstruction by downstream object-perception consistency, useful if our satellite prior claims improve driving-relevant geometry.

### Occupancy/scene geometry

- Occ3D-nuScenes and Occ3D-Waymo.
- UniOcc for unified occupancy prediction/forecasting across nuScenes, Waymo, and CARLA.
- SurroundOcc/OpenOccupancy/OccFormer/TPVFormer as occupancy baselines when using occupancy metrics.

### Mapping auxiliary

- MapTR and MapTRv2.
- PseudoMapTrainer.
- StreamMapNet, VectorMapNet, PriorMapNet, MGMap when map-head ablations become important.

## Benchmark Matrix

| Candidate | Use in this project | Why it matters | Primary metrics to borrow |
| --- | --- | --- | --- |
| E3D-Bench | Main external GFM benchmark | Covers sparse-view depth, video depth, 3D reconstruction, multi-view pose, and novel view synthesis for end-to-end 3D geometric foundation models | depth, reconstruction accuracy/completeness, pose, NVS metrics, latency/memory |
| Occ3D-nuScenes | Main driving geometry benchmark | nuScenes-derived occupancy gives autonomous-driving 3D geometry targets when dense pointmaps are hard to obtain | mIoU, IoU, semantic occupancy accuracy |
| OpenScene | Scale-up occupancy benchmark | Larger nuPlan-derived benchmark with occupancy labels and CVPR Autonomous Grand Challenge relevance | occupancy IoU/mIoU, flow-aware future extensions |
| DrivingForward | Direct driving reconstruction baseline | Feed-forward Gaussian splatting on nuScenes surround-view inputs is close to our reconstruction-first framing | rendered RGB/depth quality, reconstruction quality |
| DGGT | Direct dynamic driving reconstruction baseline | Feed-forward 4D driving reconstruction with nuScenes/Argoverse2/Waymo generalization settings | reconstruction and cross-dataset generalization metrics |
| ReconDrive | 2026 direct feed-forward reconstruction baseline | Extends VGGT for fast 4D Gaussian generation and reports nuScenes reconstruction/NVS/perception results | reconstruction metrics, NVS metrics, perception preservation, runtime |
| Perception-aware 3DGS | 2026 evaluation-oriented baseline | Argues visual similarity is insufficient and adds object/perception-aware reconstruction losses | perception consistency, object-zone quality, visual reconstruction metrics |
| ParkRecon3D / ParkGaussian | 2026 niche reconstruction benchmark | Surround-view parking scene reconstruction benchmark with dense parking-slot annotations | reconstruction quality, slot-aware perception consistency |
| Nighttime driving 3DGS | Robustness benchmark direction | Tests reconstruction under low-light driving scenes on nuScenes/Waymo | nighttime reconstruction quality, rendering metrics |
| Real2Sim | Future simulation/editing comparison | Editable physics-aware 4DGS for Waymo driving scenes, less directly aligned with our first nuScenes reconstruction table | rendering, reconstruction, editing/physics consistency |
| UniOcc | Secondary benchmark for future extension | Adds unified occupancy prediction/forecasting across nuScenes, Waymo, CARLA, and OpenCOOD | current/future occupancy mIoU/IoU, flow consistency |
| OpenOccupancy / SurroundOcc | Secondary occupancy baselines | Mature nuScenes semantic occupancy baselines and codebases | occupancy mIoU/IoU |
| MapTR / MapTRv2 | Auxiliary map-head benchmark | Useful only for vectorized map auxiliary head, not primary reconstruction claim | vector map mAP |
| PseudoMapTrainer | Auxiliary pseudo-label benchmark | Useful for valid-area masking and pseudo-label learning without HD maps | pseudo-label map mAP, observed-area metrics |
| SG-BEV | Satellite-BEV fusion design reference | Shows satellite-guided BEV fusion and reprojection ideas for cross-view alignment, but not a 3D reconstruction benchmark | segmentation mIoU only as auxiliary evidence |

## Recommended First Paper Table

Main table:

```text
Method | Satellite | Gravity aligned | Depth | Pointmap | Pose | Drift | Runtime
VGGT baseline
G3T baseline
BEV + G3T
BEV + Satellite + G3T
BEV + Satellite + G3T + valid mask / map auxiliary
```

Auxiliary table:

```text
Method | Map auxiliary | Divider AP | Crossing AP | Boundary AP | mAP
MapTR-style baseline
PseudoMapTrainer-style pseudo labels
Our reconstruction model + map auxiliary
```

## Suggested Experimental Ladder

1. VGGT/G3T reference inference on nuScenes camera clips.
2. G3T-style reconstruction with BEV-only conditioning.
3. G3T-style reconstruction with BEV + satellite conditioning.
4. Add valid-area masking and pseudo-label weighting from PseudoMapTrainer.
5. Add optional MapTR-style vector-map auxiliary head.

Primary metrics should remain depth/pointmap/pose/gravity/alignment drift. Map mAP should be auxiliary.

## Initial Source List

- G3T project page: https://g3t-paper.github.io/
- VGGT official repository: https://github.com/facebookresearch/vggt
- E3D-Bench project page: https://research.nvidia.com/labs/avg/publication/cong.liang.etal.arxiv2025/
- E3D-Bench code/project: https://e3dbench.github.io/ and https://github.com/VITA-Group/E3D-Bench
- E3D-Bench arXiv: https://arxiv.org/abs/2506.01933
- Occ3D benchmark page: https://tsinghua-mars-lab.github.io/Occ3D/
- Occ3D arXiv: https://arxiv.org/abs/2304.14365
- OpenScene repository: https://github.com/OpenDriveLab/OpenScene
- OpenOccupancy repository: https://github.com/JeffWang987/OpenOccupancy
- SurroundOcc repository: https://github.com/weiyithu/SurroundOcc
- UniOcc ICCV 2025 paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_UniOcc_A_Unified_Benchmark_for_Occupancy_Forecasting_and_Prediction_in_ICCV_2025_paper.pdf
- SG-BEV CVPR 2024 paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_SG-BEV_Satellite-Guided_BEV_Fusion_for_Cross-View_Semantic_Segmentation_CVPR_2024_paper.pdf
- ReconDrive arXiv 2026: https://arxiv.org/abs/2603.07552
- Nighttime Autonomous Driving Scene Reconstruction arXiv 2026: https://arxiv.org/abs/2602.13549
- Real2Sim arXiv 2026: https://arxiv.org/abs/2605.13591
- ParkGaussian / ParkRecon3D arXiv 2026: https://arxiv.org/abs/2601.01386
- Perception-aware 3DGS ICLR 2026 OpenReview: https://openreview.net/forum?id=PmQlMTBmpa
- DrivingForward project page: https://fangzhou2000.github.io/projects/drivingforward/
- DGGT official repository: https://github.com/xiaomi-research/dggt
- Drive-OccWorld project page: https://drive-occworld.github.io/

## Local Benchmark Code Refs

- `refs/benchmarks/E3D-Bench` at `11d82b4`.
- `refs/benchmarks/OpenOccupancy` at `eafd14f`.
- `refs/benchmarks/SurroundOcc` at `419bf5b`.
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- MapTR paper/repository: https://arxiv.org/abs/2208.14437 and https://github.com/hustvl/MapTR
- PseudoMapTrainer repository: https://github.com/boschresearch/PseudoMapTrainer
