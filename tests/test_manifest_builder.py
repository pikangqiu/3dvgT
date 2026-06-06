import unittest
from pathlib import Path

from vggt_project.data.manifest_builder import build_manifest_records


class FakeNuScenes:
    def __init__(self) -> None:
        self.sample = [
            {
                "token": "sample-1",
                "scene_token": "scene-1",
                "timestamp": 123,
                "data": {
                    "CAM_FRONT": "front-token",
                    "CAM_BACK": "back-token",
                    "LIDAR_TOP": "lidar-token",
                },
            }
        ]
        self.scene = [{"token": "scene-1", "log_token": "log-1"}]
        self.dataroot = "/dataset/nuscenes"

    def get(self, table_name: str, token: str) -> dict:
        if table_name == "sample_data" and token == "front-token":
            return {"filename": "samples/CAM_FRONT/front.jpg", "ego_pose_token": "ego-front"}
        if table_name == "sample_data" and token == "back-token":
            return {"filename": "samples/CAM_BACK/back.jpg", "ego_pose_token": "ego-back"}
        if table_name == "sample_data" and token == "lidar-token":
            return {"filename": "samples/LIDAR_TOP/lidar.bin", "ego_pose_token": "ego-lidar"}
        if table_name == "ego_pose" and token == "ego-lidar":
            return {
                "translation": [1.0, 2.0, 3.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        if table_name == "ego_pose" and token == "ego-front":
            return {
                "translation": [1.0, 2.0, 3.0],
                "rotation": [0.0, 1.0, 0.0, 0.0],
            }
        if table_name == "ego_pose" and token == "ego-back":
            return {
                "translation": [1.0, 2.0, 3.0],
                "rotation": [0.0, 0.0, 1.0, 0.0],
            }
        if table_name == "scene" and token == "scene-1":
            return {"token": "scene-1", "log_token": "log-1"}
        if table_name == "log" and token == "log-1":
            return {"location": "boston-seaport"}
        raise KeyError((table_name, token))


class ManifestBuilderTest(unittest.TestCase):
    def test_build_manifest_records_from_nuscenes_samples(self) -> None:
        records = list(
            build_manifest_records(
                nusc=FakeNuScenes(),
                satellite_patch_dir=Path("satellite"),
            )
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["token"], "sample-1")
        self.assertEqual(records[0]["camera_names"], ["CAM_BACK", "CAM_FRONT"])
        self.assertEqual(
            records[0]["camera_paths"],
            ["samples/CAM_BACK/back.jpg", "samples/CAM_FRONT/front.jpg"],
        )
        self.assertEqual(records[0]["satellite_patch_path"], "satellite/sample-1.png")
        self.assertEqual(records[0]["gravity_frame"], "gravity")
        self.assertEqual(records[0]["ego_translation"], [1.0, 2.0, 3.0])
        self.assertEqual(records[0]["ego_rotation"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            records[0]["camera_local_camera_to_gravity_poses"],
            {
                "CAM_BACK": [0.0, 0.0, 1.0, 0.0],
                "CAM_FRONT": [0.0, 1.0, 0.0, 0.0],
            },
        )
        self.assertEqual(records[0]["map_location"], "boston-seaport")


if __name__ == "__main__":
    unittest.main()
