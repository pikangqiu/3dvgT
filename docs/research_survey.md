# Baselines And Benchmarks

## Recommended Primary Benchmark Direction

The project is reconstruction-first, so the strongest first comparison should be against driving-scene reconstruction and occupancy benchmarks rather than pure HD map mAP.

Primary candidates:

- E3D-Bench for general end-to-end 3D geometric foundation model evaluation.
- Occ3D-nuScenes for autonomous-driving 3D occupancy geometry.
- DrivingForward and DGGT as driving-scene feed-forward reconstruction baselines.

Auxiliary mapping candidates:

- MapTR/MapTRv2 for vectorized HD map heads and nuScenes map mAP.
- PseudoMapTrainer for pseudo-label generation without HD maps.

## Baseline Families

### Feed-forward 3D reconstruction

- VGGT: camera/depth/pointmap foundation baseline.
- G3T: gravity-aligned pointmap baseline and our main geometry reference.
- VGG-T3: scalable VGGT-style long-input baseline.
- E3D-Bench model list: broad comparison set for GFMs.

### Driving-scene reconstruction

- DrivingForward: feed-forward Gaussian splatting for nuScenes driving-scene reconstruction.
- DGGT: feed-forward 4D dynamic driving reconstruction on Waymo, nuScenes, and Argoverse2.

### Occupancy/scene geometry

- Occ3D-nuScenes and Occ3D-Waymo.
- UniOcc for unified occupancy prediction/forecasting across nuScenes, Waymo, and CARLA.
- SurroundOcc/OpenOccupancy/OccFormer/TPVFormer as occupancy baselines when using occupancy metrics.

### Mapping auxiliary

- MapTR and MapTRv2.
- PseudoMapTrainer.
- StreamMapNet, VectorMapNet, PriorMapNet, MGMap when map-head ablations become important.

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
- Occ3D benchmark page: https://tsinghua-mars-lab.github.io/Occ3D/
- DrivingForward project page: https://fangzhou2000.github.io/projects/drivingforward/
- DGGT official repository: https://github.com/xiaomi-research/dggt
- UniOcc ICCV 2025 paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_UniOcc_A_Unified_Benchmark_for_Occupancy_Forecasting_and_Prediction_in_ICCV_2025_paper.pdf
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- MapTR paper/repository: https://arxiv.org/abs/2208.14437 and https://github.com/hustvl/MapTR
- PseudoMapTrainer repository: https://github.com/boschresearch/PseudoMapTrainer

