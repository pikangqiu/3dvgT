import tempfile
import unittest
from pathlib import Path

from vggt_project.data.manifest_validation import validate_manifest_paths


class ManifestValidationTest(unittest.TestCase):
    def test_ready_when_all_required_paths_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/patch.png").write_text("sat", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertTrue(report.ready)
        self.assertEqual(report.missing_paths, ())

    def test_missing_satellite_patch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/missing.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_paths), 1)
        self.assertEqual(report.missing_paths[0].field, "satellite_patch_path")
        self.assertEqual(report.missing_paths[0].sample_token, "sample-1")

    def test_missing_pointmap_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/patch.png").write_text("sat", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"pointmap_path":"targets/missing.npy",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_paths), 1)
        self.assertEqual(report.missing_paths[0].field, "pointmap_path")

    def test_missing_occupancy_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/patch.png").write_text("sat", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"occupancy_path":"targets/missing.npy",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_paths), 1)
        self.assertEqual(report.missing_paths[0].field, "occupancy_path")

    def test_missing_multi_camera_depth_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/patch.png").write_text("sat", encoding="utf-8")
            (root / "targets/front.png").write_text("depth", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"lidar_depth_paths":{"CAM_FRONT":"targets/front.png","CAM_BACK":"targets/missing.png"},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_paths), 1)
        self.assertEqual(report.missing_paths[0].field, "lidar_depth_paths.CAM_BACK")

    def test_missing_multi_camera_pointmap_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/patch.png").write_text("sat", encoding="utf-8")
            (root / "targets/front.npy").write_text("pointmap", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"pointmap_paths":{"CAM_FRONT":"targets/front.npy","CAM_BACK":"targets/missing.npy"},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_manifest_paths(manifest)

        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_paths), 1)
        self.assertEqual(report.missing_paths[0].field, "pointmap_paths.CAM_BACK")


if __name__ == "__main__":
    unittest.main()
