import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class ManifestTensorDatasetTest(unittest.TestCase):
    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest tensor dataset tests",
    )
    def test_manifest_images_become_model_batch(self) -> None:
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=4)[0]

        self.assertEqual(tuple(item["bev_features"].shape), (8, 16, 16))
        self.assertEqual(tuple(item["satellite_patch"].shape), (3, 16, 16))
        self.assertEqual(tuple(item["target_pointmap"].shape), (4, 3))
        self.assertEqual(tuple(item["target_depth"].shape), (1, 16, 16))
        self.assertEqual(tuple(item["valid_area_mask"].shape), (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
