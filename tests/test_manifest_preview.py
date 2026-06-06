import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(
    find_spec("PIL") and find_spec("numpy"),
    "Pillow and numpy are required for manifest preview tests",
)
class ManifestPreviewTest(unittest.TestCase):
    def test_manifest_preview_writes_summary_and_contact_sheet(self) -> None:
        import numpy as np
        from PIL import Image

        from vggt_project.data.manifest_preview import build_manifest_sample_preview

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_BACK").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (16, 16), color=(0, 0, 255)).save(root / "samples/CAM_BACK/b.png")
            Image.new("RGB", (16, 16), color=(0, 255, 0)).save(root / "sat/patch.png")
            Image.new("L", (16, 16), color=128).save(root / "targets/front_depth.png")
            Image.new("L", (16, 16), color=64).save(root / "targets/back_depth.png")
            Image.new("L", (16, 16), color=255).save(root / "targets/mask.png")
            np.save(root / "targets/front_pointmap.npy", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
            np.save(root / "targets/back_pointmap.npy", np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32))
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"valid_area_mask_path":"targets/mask.png",'
                '"lidar_depth_paths":{"CAM_FRONT":"targets/front_depth.png","CAM_BACK":"targets/back_depth.png"},'
                '"pointmap_paths":{'
                '"CAM_FRONT":"targets/front_pointmap.npy",'
                '"CAM_BACK":"targets/back_pointmap.npy"},'
                '"ego_translation":[10.0,20.0,0.0],'
                '"ego_rotation":[1.0,0.0,0.0,0.0],'
                '"map_location":"boston-seaport",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            preview = build_manifest_sample_preview(manifest, root / "preview", sample_index=0, tile_size=32)

            self.assertTrue(preview.summary_path.exists())
            self.assertTrue(preview.contact_sheet_path.exists())
            summary = json.loads(preview.summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["token"], "sample-1")
        self.assertEqual(summary["scene_token"], "scene-1")
        self.assertEqual(summary["map_location"], "boston-seaport")
        self.assertEqual(summary["camera_count"], 2)
        self.assertEqual(summary["camera_names"], ["CAM_FRONT", "CAM_BACK"])
        self.assertTrue(summary["has_satellite_patch"])
        self.assertTrue(summary["has_valid_area_mask"])
        self.assertEqual(summary["depth_target_cameras"], ["CAM_BACK", "CAM_FRONT"])
        self.assertEqual(summary["pointmap_target_cameras"], ["CAM_BACK", "CAM_FRONT"])
