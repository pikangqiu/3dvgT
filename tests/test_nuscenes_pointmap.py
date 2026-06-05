import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("numpy"), "numpy is required for nuScenes pointmap tests")
class NuScenesPointmapTest(unittest.TestCase):
    def test_lidar_points_to_ego_pointmap_applies_calibration_transform(self) -> None:
        import numpy as np

        from vggt_project.data.nuscenes_pointmap import lidar_points_to_ego_pointmap

        points = np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
        calibrated_sensor = {
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [10.0, 20.0, 30.0],
        }

        pointmap = lidar_points_to_ego_pointmap(points, calibrated_sensor)

        self.assertEqual(pointmap.shape, (2, 3))
        np.testing.assert_allclose(
            pointmap,
            np.asarray([[11.0, 23.0, 35.0], [12.0, 24.0, 36.0]], dtype=np.float32),
        )

    def test_materialize_lidar_pointmap_manifest_updates_output_manifest(self) -> None:
        import numpy as np

        import vggt_project.data.nuscenes_pointmap as pointmap_module
        from vggt_project.data.nuscenes_pointmap import materialize_lidar_pointmap_manifest

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
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
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
            output = root / "samples.pointmap.jsonl"
            original_loader = pointmap_module._load_lidar_points
            pointmap_module._load_lidar_points = fake_load_lidar_points
            try:
                report = materialize_lidar_pointmap_manifest(
                    FakeNuScenes(),
                    manifest,
                    pointmap_dir=Path("pointmaps"),
                    output_manifest_path=output,
                    max_points=2,
                )
            finally:
                pointmap_module._load_lidar_points = original_loader

            record = json.loads(output.read_text(encoding="utf-8"))
            pointmap = np.load(root / record["pointmap_path"])

        self.assertEqual(report.pointmaps_written, 1)
        self.assertEqual(record["pointmap_path"], "pointmaps/sample-1_LIDAR_TOP.npy")
        self.assertEqual(pointmap.shape, (2, 3))
        np.testing.assert_allclose(
            pointmap,
            np.asarray([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
