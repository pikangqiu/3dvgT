import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("numpy"), "numpy is required for nuScenes occupancy tests")
class NuScenesOccupancyTest(unittest.TestCase):
    def test_lidar_points_to_bev_occupancy_rasterizes_xy_cells(self) -> None:
        import numpy as np

        from vggt_project.data.nuscenes_occupancy import lidar_points_to_bev_occupancy

        points_ego = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [0.25, 0.25, 2.0],
                [1.9, 1.9, 4.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        occupancy = lidar_points_to_bev_occupancy(
            points_ego,
            x_range=(-2.0, 2.0),
            y_range=(-2.0, 2.0),
            z_range=(-1.0, 3.0),
            grid_size=(4, 4),
        )

        self.assertEqual(occupancy.shape, (4, 4))
        self.assertEqual(float(occupancy[1, 1]), 1.0)
        self.assertEqual(float(occupancy[2, 2]), 1.0)
        self.assertEqual(float(occupancy.sum()), 2.0)

    def test_materialize_lidar_occupancy_manifest_writes_occupancy_path(self) -> None:
        import numpy as np

        import vggt_project.data.nuscenes_occupancy as occupancy_module
        from vggt_project.data.nuscenes_occupancy import materialize_lidar_occupancy_manifest

        class FakeNuScenes:
            dataroot = ""

            def __init__(self) -> None:
                self.sample = {
                    "sample-1": {
                        "token": "sample-1",
                        "data": {"LIDAR_TOP": "lidar-token"},
                    }
                }
                self.sample_data = {
                    "lidar-token": {
                        "filename": "samples/LIDAR_TOP/sample.bin",
                        "calibrated_sensor_token": "lidar-calib",
                    }
                }
                self.calibrated_sensor = {
                    "lidar-calib": {
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0],
                    }
                }

            def get(self, table_name: str, token: str) -> dict:
                return getattr(self, table_name)[token]

        def fake_load_lidar_points(path: Path):
            return np.asarray(
                [
                    [-1.0, 0.25],
                    [-1.0, 0.25],
                    [0.0, 2.0],
                ],
                dtype=np.float32,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.occupancy.jsonl"
            original_loader = occupancy_module._load_lidar_points
            occupancy_module._load_lidar_points = fake_load_lidar_points
            try:
                report = materialize_lidar_occupancy_manifest(
                    FakeNuScenes(),
                    manifest,
                    occupancy_dir=Path("occupancy"),
                    output_manifest_path=output,
                    x_range=(-2.0, 2.0),
                    y_range=(-2.0, 2.0),
                    z_range=(-1.0, 3.0),
                    grid_size=(4, 4),
                )
            finally:
                occupancy_module._load_lidar_points = original_loader

            record = json.loads(output.read_text(encoding="utf-8"))
            occupancy = np.load(root / record["occupancy_path"])

        self.assertEqual(report.occupancy_maps_written, 1)
        self.assertEqual(record["occupancy_path"], "occupancy/sample-1_LIDAR_TOP.npy")
        self.assertEqual(occupancy.shape, (4, 4))
        self.assertEqual(float(occupancy[1, 1]), 1.0)
        self.assertEqual(float(occupancy[2, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
