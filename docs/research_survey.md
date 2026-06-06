# Baselines And Benchmarks

## Recommended Primary Benchmark Direction

The project is reconstruction-first, so the strongest first comparison should be against driving-scene reconstruction and occupancy benchmarks rather than pure HD map mAP.

This survey was refreshed on 2026-06-06 against primary project pages, arXiv/CVF/OpenReview pages, and official GitHub repositories. The structured version used by scripts is `src/vggt_project/experiment_protocol.py`.

Primary candidates:

- Sky2Ground for satellite/aerial/ground site modeling, camera localization, and rendering/reconstruction under large viewpoint/altitude changes.
- CrossGeo/Cross3R for feed-forward satellite-drone-ground 3D reconstruction and 6-DoF cross-view pose estimation.
- E3D-Bench for general end-to-end 3D geometric foundation model evaluation.
- Occ3D-nuScenes for autonomous-driving 3D occupancy geometry.
- OpenScene for a larger nuPlan-derived occupancy benchmark when scaling beyond nuScenes-mini.
- UniOcc for current-frame and future-frame occupancy prediction/forecasting.
- DrivingForward and DGGT as driving-scene feed-forward reconstruction baselines.
- ReconDrive as a 2026 feed-forward 4D Gaussian Splatting driving-scene reconstruction baseline that extends VGGT-style geometric features.
- DynamicVGGT as a 2026 VGGT-extension baseline for dynamic pointmaps and 4D autonomous-driving reconstruction.
- ParkRecon3D/ParkGaussian as a 2026 surround-view parking-scene reconstruction benchmark/baseline.
- Perception-aware 3DGS as a 2026 evaluation direction for whether reconstructed driving scenes preserve downstream perception behavior.
- Sat3DGen as a 2026 satellite-only street-level 3D generation benchmark direction with VIGOR-OOD plus DSM geometry supervision.
- SA-Occ as the closest public satellite-assisted 3D occupancy baseline on Occ3D-nuScenes.
- DriveTok as a 2026 nuScenes multi-view scene-token baseline spanning reconstruction, depth, segmentation, and occupancy.
- M2-Occ as a 2026 missing-camera robustness protocol on SurroundOcc/nuScenes, useful for testing whether satellite priors help when surround cameras are incomplete.
- GS-Occ3D / GaussianOcc as Gaussian-splatting occupancy baselines for scalable vision-only or self-supervised driving geometry.

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
- InstantSplat: fast sparse-view SfM-free Gaussian Splatting baseline for cases where pose is weak or missing.

### Satellite-ground reconstruction and localization

- Sky2Ground / SkyNet: 2026 satellite/aerial/ground benchmark and model for site modeling under varying altitude; directly relevant for satellite images as noisy but useful cross-view priors.
- CrossGeo / Cross3R: 2026 tri-view satellite-drone-ground benchmark/model for feed-forward 3D point cloud reconstruction, 6-DoF camera pose, and on-tile cross-view localization.
- SG-BEV: satellite-guided BEV fusion reference for feature alignment, useful as an auxiliary design baseline rather than a reconstruction metric baseline.
- Sat3DGen: satellite-only street-level 3D generation reference; not a nuScenes driving baseline, but useful for testing whether satellite priors can support geometry-first street-scene reconstruction.

### Driving-scene reconstruction

- DrivingForward: feed-forward Gaussian splatting for nuScenes driving-scene reconstruction.
- DGGT: feed-forward 4D dynamic driving reconstruction on Waymo, nuScenes, and Argoverse2.
- ReconDrive: fast feed-forward 4D Gaussian Splatting for autonomous driving scene reconstruction, explicitly extending VGGT and benchmarking on nuScenes.
- DynamicVGGT: dynamic pointmap and 3D Gaussian motion baseline for autonomous-driving 4D reconstruction, useful when the project moves from static sample-level reconstruction to temporal scene reconstruction.
- Nighttime Autonomous Driving Scene Reconstruction: 2026 ICRA direction for low-light driving-scene reconstruction on nuScenes/Waymo.
- Real2Sim: physics-driven editable 4DGS for driving scenes, useful as a future simulation/editing comparison rather than first-table baseline.
- Perception-aware 3DGS: evaluates reconstruction by downstream object-perception consistency, useful if our satellite prior claims improve driving-relevant geometry.

### Occupancy/scene geometry

- Occ3D-nuScenes and Occ3D-Waymo.
- UniOcc for unified occupancy prediction/forecasting across nuScenes, Waymo, and CARLA.
- SA-Occ for satellite-assisted 3D occupancy on Occ3D-nuScenes, directly aligned with the satellite-conditioned geometry claim.
- M2-Occ for missing-view robustness on nuScenes/SurroundOcc when satellite priors are expected to stabilize incomplete camera inputs.
- DriveTok for multi-view scene-token reconstruction/understanding on nuScenes.
- SurroundOcc/OpenOccupancy/OccFormer/TPVFormer as occupancy baselines when using occupancy metrics.
- GS-Occ3D: vision-only occupancy reconstruction/label curation with Gaussian Splatting, useful if the paper claims scalable geometry labels.
- GaussianOcc: self-supervised Gaussian Splatting occupancy/depth training on surround-view driving datasets.

### Mapping auxiliary

- MapTR and MapTRv2.
- PseudoMapTrainer.
- StreamMapNet, VectorMapNet, PriorMapNet, MGMap when map-head ablations become important.

## Benchmark Matrix

| Candidate | Use in this project | Why it matters | Primary metrics to borrow |
| --- | --- | --- | --- |
| Sky2Ground / SkyNet | Main cross-view benchmark direction | Directly studies satellite/aerial/ground site modeling and shows satellite views can degrade naive fine-tuning, motivating careful satellite branch design | camera pose/localization accuracy, rendering/reconstruction quality, depth/surface normal when available |
| CrossGeo / Cross3R | Main satellite-ground reconstruction benchmark direction | Directly targets satellite/drone/ground feed-forward 3D reconstruction and 6-DoF pose, close to our satellite-prior claim even though not nuScenes-specific | point-cloud reconstruction, 6-DoF pose, cross-view localization |
| E3D-Bench | Main external GFM benchmark | Covers sparse-view depth, video depth, 3D reconstruction, multi-view pose, and novel view synthesis for end-to-end 3D geometric foundation models | depth, reconstruction accuracy/completeness, pose, NVS metrics, latency/memory |
| Occ3D-nuScenes | Main driving geometry benchmark | nuScenes-derived occupancy gives autonomous-driving 3D geometry targets when dense pointmaps are hard to obtain | mIoU, IoU, semantic occupancy accuracy |
| OpenScene | Scale-up occupancy benchmark | Larger nuPlan-derived benchmark with occupancy labels and CVPR Autonomous Grand Challenge relevance | occupancy IoU/mIoU, flow-aware future extensions |
| DrivingForward | Direct driving reconstruction baseline | Feed-forward Gaussian splatting on nuScenes surround-view inputs is close to our reconstruction-first framing | rendered RGB/depth quality, reconstruction quality |
| DGGT | Direct dynamic driving reconstruction baseline | Feed-forward 4D driving reconstruction with nuScenes/Argoverse2/Waymo generalization settings | reconstruction and cross-dataset generalization metrics |
| ReconDrive | 2026 direct feed-forward reconstruction baseline | Extends VGGT for fast 4D Gaussian generation and reports nuScenes reconstruction/NVS/perception results | reconstruction metrics, NVS metrics, perception preservation, runtime |
| DynamicVGGT | 2026 dynamic driving reconstruction baseline | Extends VGGT with dynamic pointmaps, temporal attention, and 3D Gaussian motion heads for autonomous-driving 4D reconstruction | dynamic pointmap error, reconstruction quality, temporal/motion consistency |
| Perception-aware 3DGS | 2026 evaluation-oriented baseline | Argues visual similarity is insufficient and adds object/perception-aware reconstruction losses | perception consistency, object-zone quality, visual reconstruction metrics |
| Sat3DGen | 2026 satellite-only 3D generation benchmark direction | Pairs satellite views with DSM geometry on VIGOR-OOD; useful for satellite-only ablations and geometry-first satellite prior evaluation | DSM RMSE, FID/NVS quality, mesh/scene geometry quality |
| SA-Occ | Satellite-assisted occupancy baseline | ICCV 2025 satellite-assisted 3D occupancy on Occ3D-nuScenes; closest public comparison for whether satellite context improves driving-scene geometry | semantic occupancy mIoU, class IoU, latency |
| DriveTok | 2026 multi-view scene-token baseline | Uses nuScenes to evaluate unified scene tokens for image reconstruction, semantic segmentation, depth prediction, and 3D occupancy prediction | image reconstruction, depth, segmentation, occupancy metrics |
| M2-Occ | 2026 robustness benchmark direction | Defines deterministic and stochastic missing-view protocols on SurroundOcc/nuScenes; useful to test whether satellite priors compensate for missing cameras | occupancy mIoU under camera dropout |
| ParkRecon3D / ParkGaussian | 2026 niche reconstruction benchmark | Surround-view parking scene reconstruction benchmark with dense parking-slot annotations | reconstruction quality, slot-aware perception consistency |
| Nighttime driving 3DGS | Robustness benchmark direction | Tests reconstruction under low-light driving scenes on nuScenes/Waymo | nighttime reconstruction quality, rendering metrics |
| Real2Sim | Future simulation/editing comparison | Editable physics-aware 4DGS for Waymo driving scenes, less directly aligned with our first nuScenes reconstruction table | rendering, reconstruction, editing/physics consistency |
| GS-Occ3D | Scalable geometry-label baseline | Curates vision-only occupancy labels and evaluates zero-shot transfer to Occ3D-nuScenes | binary occupancy quality, downstream Occ3D generalization |
| GaussianOcc | Self-supervised occupancy/depth baseline | Uses Gaussian Splatting modules for scale-aware pose/depth and efficient voxel rendering on nuScenes/DDAD | occupancy IoU/mIoU, depth error, training/render speed |
| InstantSplat | Sparse-view reconstruction reference | Useful for ablations where pose is weak and fast feed-forward/sparse-view reconstruction is needed | NVS quality, reconstruction quality, runtime |
| UniOcc | Secondary benchmark for future extension | Adds unified occupancy prediction/forecasting across nuScenes, Waymo, CARLA, and OpenCOOD | current/future occupancy mIoU/IoU, flow consistency |
| OpenOccupancy / SurroundOcc | Secondary occupancy baselines | Mature nuScenes semantic occupancy baselines and codebases | occupancy mIoU/IoU |
| MapTR / MapTRv2 | Auxiliary map-head benchmark | Useful only for vectorized map auxiliary head, not primary reconstruction claim | vector map mAP |
| PseudoMapTrainer | Auxiliary pseudo-label benchmark | Useful for valid-area masking and pseudo-label learning without HD maps | pseudo-label map mAP, observed-area metrics |
| SG-BEV | Satellite-BEV fusion design reference | Shows satellite-guided BEV fusion and reprojection ideas for cross-view alignment, but not a 3D reconstruction benchmark | segmentation mIoU only as auxiliary evidence |

## Recommended First Paper Table

The same protocol is available as a machine-readable artifact:

```bash
PYTHONPATH=src python scripts/list_experiment_protocol.py
PYTHONPATH=src python scripts/list_experiment_protocol.py --json
```

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

Primary metrics should remain depth/pointmap/pose/gravity/alignment drift. The current scaffold reports `depth_mae`, scale-aligned pointmap accuracy/completeness/chamfer, `gravity_error_deg`, and `sequence_translation_drift`; map mAP should remain auxiliary.

## Experiment Execution Priority

The protocol now uses `experiment_phase` in `src/vggt_project/experiment_protocol.py` and `scripts/list_experiment_protocol.py --json` so the benchmark list is actionable instead of just broad.

### `phase1_core`: first trainable paper table

Run these once real nuScenes, satellite rasters, G3T/VGGT weights, and GPU validation are available:

- VGGT camera-only baseline.
- G3T camera-only/gravity-aligned baseline.
- BEV+G3T ablation.
- BEV+Satellite+G3T main model.

Metrics:

- `depth_mae`
- `scale_aligned_pointmap_chamfer`
- `gravity_error_deg`
- `sequence_translation_drift`
- `relative_pose_l2`
- runtime and memory

### `phase1_external`: first public-context comparison

Use these to make the first result table defensible in autonomous-driving geometry language:

- Occ3D-nuScenes / OpenOccupancy for public occupancy framing.
- SA-Occ as the closest satellite-assisted occupancy baseline.

The current repository only has LiDAR-derived binary occupancy proxy targets, so semantic Occ3D/OpenOccupancy evaluation is a later real-data validation step, not something the scaffold can honestly claim now.

### `phase2_external`: broader satellite/cross-view validation

Use these when the main model is already training:

- CrossGeo/Cross3R for satellite-drone-ground point-cloud reconstruction, 6-DoF pose, and on-tile localization.
- Sky2Ground/SkyNet for satellite/aerial/ground camera localization and reconstruction under altitude/viewpoint shifts.
- E3D-Bench for general geometric foundation model reporting.
- UniOcc, M2-Occ, GaussianOcc, and SurroundOcc when occupancy robustness becomes a main claim.

### `phase2_tracking`: dynamic driving-scene reconstruction

Track these once the project moves from sample-level reconstruction to temporal reconstruction:

- DGGT.
- DrivingForward.
- ReconDrive.
- DynamicVGGT.
- PAGE-4D.
- DriveTok.

These are useful because they extend feed-forward/VGGT-style reconstruction into autonomous-driving temporal geometry, but they should not block the first static reconstruction experiment.

### `phase3_stretch`: optional or auxiliary paper angles

Use these only after the main result is stable:

- Sat3DGen for satellite-only DSM/street-level generation stress testing.
- MapTR and PseudoMapTrainer for optional vector-map auxiliary heads.
- SG-BEV for satellite/BEV alignment design evidence.
- OpenScene when scaling beyond nuScenes-mini/nuScenes to larger nuPlan-derived occupancy.

## Protocol Decision As Of 2026-06-06

Use three comparison tables instead of one overloaded table:

1. Primary reconstruction table:
   VGGT, G3T, BEV+G3T, BEV+Satellite+G3T, BEV+Satellite+G3T+valid-mask/map auxiliary.
   Metrics: `depth_mae`, `scale_aligned_pointmap_chamfer`, `gravity_error_deg`, `sequence_translation_drift`, `relative_pose_l2`, runtime/memory.
2. Satellite/cross-view reconstruction table:
   Cross3R/CrossGeo and SkyNet/Sky2Ground as the most direct external evidence for satellite/UAV/ground reconstruction and localization; Sat3DGen as a satellite-only geometry stress test.
   Metrics: point-cloud reconstruction, 6-DoF pose, on-tile localization, RRA/RTA-style camera localization when available, and DSM RMSE for satellite-only generation.
3. Driving geometry auxiliary table:
   Occ3D/OpenOccupancy/SurroundOcc/OpenScene/UniOcc/SA-Occ/M2-Occ for occupancy; DrivingForward/DGGT/ReconDrive/DynamicVGGT/DriveTok for feed-forward driving reconstruction or scene-token reconstruction.
   Metrics: implemented `bev_occupancy_iou` for local LiDAR-proxy smoke labels, semantic occupancy mIoU for public labels, and NVS/perception-preservation metrics when Gaussian-splatting baselines are run.

Rationale:

- G3T is the core method reference because it explicitly predicts upright pointmaps and camera-to-gravity poses in gravity-aligned frames.
- VGGT remains the camera-only geometric foundation baseline because its public repo provides pointmaps, camera parameters, and reconstruction demos.
- E3D-Bench is the broad external GFM benchmark because it evaluates sparse-view depth, video depth, 3D reconstruction, multi-view pose, and novel-view synthesis.
- Sky2Ground is the best stress test for the satellite branch: its paper reports that satellite imagery can degrade naive localization/reconstruction under large altitude changes, so a positive result there is much stronger than only testing segmentation.
- CrossGeo/Cross3R is the closest task-level match for feed-forward satellite+ground 3D reconstruction and camera pose/localization.
- ReconDrive and DGGT are the strongest driving-scene feed-forward reconstruction baselines to watch because both explicitly adapt VGGT-style geometry or transformer feed-forward reconstruction to nuScenes-like dynamic driving scenes.
- DynamicVGGT is the most direct dynamic-pointmap extension of VGGT for autonomous driving, so it should be tracked when temporal reconstruction becomes a main table.
- Sat3DGen is not a direct nuScenes comparison, but it is a strong satellite-only geometry sanity benchmark because it reports DSM-supervised street-level 3D generation from satellite input.
- SA-Occ is the most directly aligned public satellite-assisted occupancy baseline: it tests satellite context on Occ3D-nuScenes rather than only cross-view segmentation.
- DriveTok gives a 2026 nuScenes reconstruction/understanding token baseline that can be used once the project reports reconstruction plus occupancy/segmentation auxiliary tasks.
- M2-Occ supplies a robustness protocol for missing surround-camera inputs, a natural place to test whether satellite priors improve geometry when camera coverage is degraded.
- Occ3D/OpenScene/UniOcc provide public occupancy labels and evaluation language; the current project only has LiDAR-derived binary proxy occupancy, so public semantic occupancy validation remains a future milestone.

## Initial Source List

- G3T project page: https://g3t-paper.github.io/
- G3T arXiv: https://arxiv.org/abs/2605.27372
- VGGT official repository: https://github.com/facebookresearch/vggt
- VGGT arXiv: https://arxiv.org/abs/2503.11651
- E3D-Bench project page: https://research.nvidia.com/labs/avg/publication/cong.liang.etal.arxiv2025/
- E3D-Bench code/project: https://e3dbench.github.io/ and https://github.com/VITA-Group/E3D-Bench
- E3D-Bench arXiv: https://arxiv.org/abs/2506.01933
- Sky2Ground CVPR 2026 paper: https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Sky2Ground_A_Benchmark_for_Site_Modeling_under_Varying_Altitude_CVPR_2026_paper.html
- Sky2Ground arXiv: https://arxiv.org/abs/2603.13740
- Cross3R / CrossGeo arXiv 2026: https://arxiv.org/abs/2605.07978
- Occ3D benchmark page: https://tsinghua-mars-lab.github.io/Occ3D/
- Occ3D arXiv: https://arxiv.org/abs/2304.14365
- OpenScene repository: https://github.com/OpenDriveLab/OpenScene
- OpenOccupancy repository: https://github.com/JeffWang987/OpenOccupancy
- SurroundOcc repository: https://github.com/weiyithu/SurroundOcc
- UniOcc project page: https://uniocc.github.io/
- UniOcc ICCV 2025 paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_UniOcc_A_Unified_Benchmark_for_Occupancy_Forecasting_and_Prediction_in_ICCV_2025_paper.pdf
- SG-BEV CVPR 2024 paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_SG-BEV_Satellite-Guided_BEV_Fusion_for_Cross-View_Semantic_Segmentation_CVPR_2024_paper.pdf
- ReconDrive arXiv 2026: https://arxiv.org/abs/2603.07552
- DynamicVGGT arXiv 2026: https://arxiv.org/abs/2603.08254
- Sat3DGen arXiv 2026: https://arxiv.org/abs/2605.14984
- Sat3DGen repository: https://github.com/qianmingduowan/Sat3DGen
- SA-Occ ICCV 2025 paper: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_SA-Occ_Satellite-Assisted_3D_Occupancy_Prediction_in_Real_World_ICCV_2025_paper.html
- DriveTok arXiv 2026: https://arxiv.org/abs/2603.19219
- M2-Occ arXiv 2026: https://arxiv.org/abs/2603.09737
- PAGE-4D OpenReview / ICLR 2026: https://openreview.net/forum?id=Nfmzp5PBzr
- Nighttime Autonomous Driving Scene Reconstruction arXiv 2026: https://arxiv.org/abs/2602.13549
- Real2Sim arXiv 2026: https://arxiv.org/abs/2605.13591
- ParkGaussian / ParkRecon3D arXiv 2026: https://arxiv.org/abs/2601.01386
- Perception-aware 3DGS ICLR 2026 OpenReview: https://openreview.net/forum?id=PmQlMTBmpa
- GS-Occ3D project page: https://gs-occ3d.github.io/
- GaussianOcc project page: https://ganwanshui.github.io/GaussianOcc/
- InstantSplat project page: https://research.nvidia.com/labs/avg/publication/fan.cong.etal.arxiv2025/
- DrivingForward project page: https://fangzhou2000.github.io/projects/drivingforward/
- DrivingForward repository: https://github.com/fangzhou2000/DrivingForward
- DGGT official repository: https://github.com/xiaomi-research/dggt
- Drive-OccWorld project page: https://drive-occworld.github.io/

## Local Benchmark Code Refs

- `refs/benchmarks/E3D-Bench` at `11d82b4`.
- `refs/benchmarks/OpenOccupancy` at `eafd14f`.
- `refs/benchmarks/SurroundOcc` at `419bf5b`.
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- MapTR paper/repository: https://arxiv.org/abs/2208.14437 and https://github.com/hustvl/MapTR
- PseudoMapTrainer repository: https://github.com/boschresearch/PseudoMapTrainer
