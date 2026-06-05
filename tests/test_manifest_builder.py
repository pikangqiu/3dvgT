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
        self.dataroot = "/dataset/nuscenes"

    def get(self, table_name: str, token: str) -> dict:
        if table_name == "sample_data" and token == "front-token":
            return {"filename": "samples/CAM_FRONT/front.jpg"}
        if table_name == "sample_data" and token == "back-token":
            return {"filename": "samples/CAM_BACK/back.jpg"}
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


if __name__ == "__main__":
    unittest.main()
