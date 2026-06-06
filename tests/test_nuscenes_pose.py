import json
import tempfile
import unittest
from pathlib import Path


class NuScenesPoseTest(unittest.TestCase):
    def test_materialize_camera_pose_manifest_writes_calibrated_sensor_rotations(self) -> None:
        from vggt_project.data.nuscenes_pose import materialize_camera_pose_manifest

        class FakeNuScenes:
            def __init__(self) -> None:
                self.sample = {
                    "sample-1": {
                        "token": "sample-1",
                        "data": {
                            "CAM_FRONT": "front-token",
                            "CAM_BACK": "back-token",
                        },
                    }
                }
                self.sample_data = {
                    "front-token": {"calibrated_sensor_token": "front-calib"},
                    "back-token": {"calibrated_sensor_token": "back-calib"},
                }
                self.calibrated_sensor = {
                    "front-calib": {"rotation": [0.0, 1.0, 0.0, 0.0]},
                    "back-calib": {"rotation": [0.0, 0.0, 1.0, 0.0]},
                }

            def get(self, table_name: str, token: str) -> dict:
                return getattr(self, table_name)[token]

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
            output = root / "samples.pose.jsonl"

            report = materialize_camera_pose_manifest(
                FakeNuScenes(),
                manifest,
                output_manifest_path=output,
                camera_names=("CAM_FRONT", "CAM_BACK"),
            )

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.pose_targets_written, 2)
        self.assertEqual(report.camera_names, ("CAM_FRONT", "CAM_BACK"))
        self.assertEqual(
            record["camera_local_camera_to_gravity_poses"],
            {
                "CAM_BACK": [0.0, 0.0, 1.0, 0.0],
                "CAM_FRONT": [0.0, 1.0, 0.0, 0.0],
            },
        )

    def test_camera_pose_manifest_defaults_to_manifest_camera_names(self) -> None:
        from vggt_project.data.nuscenes_pose import materialize_camera_pose_manifest

        class FakeNuScenes:
            sample = {"sample-1": {"token": "sample-1", "data": {"CAM_FRONT": "front-token"}}}
            sample_data = {"front-token": {"calibrated_sensor_token": "front-calib"}}
            calibrated_sensor = {"front-calib": {"rotation": [1.0, 0.0, 0.0, 0.0]}}

            def get(self, table_name: str, token: str) -> dict:
                return getattr(self, table_name)[token]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"camera_names":["CAM_FRONT"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.pose.jsonl"

            report = materialize_camera_pose_manifest(
                FakeNuScenes(),
                manifest,
                output_manifest_path=output,
            )

        self.assertEqual(report.pose_targets_written, 1)
        self.assertEqual(report.camera_names, ("CAM_FRONT",))


if __name__ == "__main__":
    unittest.main()
