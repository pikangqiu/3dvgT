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

    def test_camera_points_to_visible_pointmap_filters_by_projection(self) -> None:
        import numpy as np

        from vggt_project.data.nuscenes_pointmap import camera_points_to_visible_pointmap

        points_camera = np.asarray(
            [
                [0.0, 1.0, -10.0, 20.0],
                [0.0, 0.0, 0.0, 20.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        intrinsics = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        pointmap = camera_points_to_visible_pointmap(
            points_camera=points_camera,
            camera_intrinsic=intrinsics,
            image_width=4,
            image_height=4,
            max_points=2,
        )

        np.testing.assert_allclose(
            pointmap,
            np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float32),
        )

    def test_materialize_camera_lidar_pointmap_manifest_writes_camera_pointmap_paths(self) -> None:
        import numpy as np

        import vggt_project.data.nuscenes_pointmap as pointmap_module
        from vggt_project.data.nuscenes_pointmap import materialize_camera_lidar_pointmap_manifest

        class FakeNuScenes:
            dataroot = ""

            def __init__(self) -> None:
                self.sample = {
                    "sample-1": {
                        "token": "sample-1",
                        "data": {
                            "LIDAR_TOP": "lidar-token",
                            "CAM_FRONT": "front-token",
                            "CAM_BACK": "back-token",
                        },
                    }
                }
                self.sample_data = {
                    "lidar-token": {
                        "filename": "samples/LIDAR_TOP/sample.bin",
                        "calibrated_sensor_token": "lidar-calib",
                        "ego_pose_token": "ego-lidar",
                    },
                    "front-token": {
                        "filename": "samples/CAM_FRONT/a.jpg",
                        "calibrated_sensor_token": "front-calib",
                        "ego_pose_token": "ego-front",
                        "width": 4,
                        "height": 4,
                    },
                    "back-token": {
                        "filename": "samples/CAM_BACK/b.jpg",
                        "calibrated_sensor_token": "back-calib",
                        "ego_pose_token": "ego-back",
                        "width": 4,
                        "height": 4,
                    },
                }
                self.calibrated_sensor = {
                    "lidar-calib": {
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0],
                    },
                    "front-calib": {
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0],
                        "camera_intrinsic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    "back-calib": {
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0],
                        "camera_intrinsic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                }
                self.ego_pose = {
                    "ego-lidar": {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
                    "ego-front": {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
                    "ego-back": {"rotation": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
                }

            def get(self, table_name: str, token: str) -> dict:
                return getattr(self, table_name)[token]

        def fake_load_lidar_points(path: Path):
            return np.asarray(
                [
                    [0.0, 1.0, 20.0],
                    [0.0, 0.0, 20.0],
                    [5.0, 5.0, 5.0],
                ],
                dtype=np.float32,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg","samples/CAM_BACK/b.jpg"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.camera-pointmap.jsonl"
            original_loader = pointmap_module._load_lidar_points
            pointmap_module._load_lidar_points = fake_load_lidar_points
            try:
                report = materialize_camera_lidar_pointmap_manifest(
                    FakeNuScenes(),
                    manifest,
                    camera_names=("CAM_FRONT", "CAM_BACK"),
                    pointmap_dir=Path("camera_pointmaps"),
                    output_manifest_path=output,
                    max_points=2,
                )
            finally:
                pointmap_module._load_lidar_points = original_loader

            record = json.loads(output.read_text(encoding="utf-8"))
            front = np.load(root / record["pointmap_paths"]["CAM_FRONT"])
            back = np.load(root / record["pointmap_paths"]["CAM_BACK"])

        self.assertEqual(report.pointmaps_written, 2)
        self.assertEqual(record["pointmap_paths"]["CAM_FRONT"], "camera_pointmaps/sample-1_CAM_FRONT.npy")
        self.assertEqual(record["pointmap_paths"]["CAM_BACK"], "camera_pointmaps/sample-1_CAM_BACK.npy")
        np.testing.assert_allclose(front, np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float32))
        np.testing.assert_allclose(back, np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
