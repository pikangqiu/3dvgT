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
        self.assertEqual(tuple(item["camera_images"].shape), (1, 3, 16, 16))
        self.assertEqual(tuple(item["satellite_patch"].shape), (3, 16, 16))
        self.assertEqual(tuple(item["target_pointmap"].shape), (4, 3))
        self.assertEqual(tuple(item["target_depth"].shape), (1, 16, 16))
        self.assertEqual(tuple(item["valid_area_mask"].shape), (1, 16, 16))

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest tensor dataset tests",
    )
    def test_optional_depth_and_mask_targets_are_loaded(self) -> None:
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            Image.new("L", (8, 8), color=128).save(root / "targets/depth.png")
            Image.new("L", (8, 8), color=0).save(root / "targets/mask.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"lidar_depth_path":"targets/depth.png",'
                '"valid_area_mask_path":"targets/mask.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=4)[0]

        self.assertGreater(float(item["target_depth"].mean()), 0.4)
        self.assertLess(float(item["target_depth"].mean()), 0.6)
        self.assertEqual(float(item["valid_area_mask"].sum()), 0.0)

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest tensor dataset tests",
    )
    def test_multi_camera_depth_targets_are_loaded_as_stack(self) -> None:
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_BACK").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "samples/CAM_BACK/b.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            Image.new("L", (8, 8), color=64).save(root / "targets/front.png")
            Image.new("L", (8, 8), color=192).save(root / "targets/back.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"lidar_depth_paths":{"CAM_FRONT":"targets/front.png","CAM_BACK":"targets/back.png"},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=4)[0]

        self.assertEqual(tuple(item["target_camera_depths"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(item["camera_images"].shape), (2, 3, 16, 16))
        self.assertGreater(float(item["target_camera_depths"][0].mean()), 0.2)
        self.assertLess(float(item["target_camera_depths"][0].mean()), 0.3)
        self.assertGreater(float(item["target_camera_depths"][1].mean()), 0.7)
        self.assertLess(float(item["target_camera_depths"][1].mean()), 0.8)
        self.assertGreater(float(item["target_depth"].mean()), 0.45)
        self.assertLess(float(item["target_depth"].mean()), 0.55)

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch") and find_spec("numpy"),
        "Pillow, torch, and numpy are required for pointmap target tests",
    )
    def test_optional_pointmap_target_is_loaded_and_padded(self) -> None:
        import numpy as np
        import torch
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            np.save(
                root / "targets/pointmap.npy",
                np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            )
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"pointmap_path":"targets/pointmap.npy",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=4)[0]

        self.assertTrue(
            torch.allclose(
                item["target_pointmap"],
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
            )
        )

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch") and find_spec("numpy"),
        "Pillow, torch, and numpy are required for pointmap target tests",
    )
    def test_multi_camera_pointmap_targets_are_loaded_as_stack(self) -> None:
        import numpy as np
        import torch
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_BACK").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "samples/CAM_BACK/b.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            np.save(root / "targets/front_pointmap.npy", np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
            np.save(root / "targets/back_pointmap.npy", np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32))
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"pointmap_paths":{'
                '"CAM_FRONT":"targets/front_pointmap.npy",'
                '"CAM_BACK":"targets/back_pointmap.npy"},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=2)[0]

        self.assertEqual(tuple(item["target_camera_pointmaps"].shape), (2, 2, 3))
        self.assertTrue(
            torch.allclose(
                item["target_camera_pointmaps"],
                torch.tensor(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                item["target_pointmap"],
                torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
            )
        )

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest tensor dataset tests",
    )
    def test_ego_pose_metadata_becomes_pose_targets(self) -> None:
        import math

        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        yaw_90_quaternion = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_BACK").mkdir(parents=True)
            (root / "sat").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/b.png")
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "samples/CAM_BACK/b.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/b.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png"],'
                '"satellite_patch_path":"sat/a.png",'
                '"ego_translation":[10.0,20.0,0.0],'
                '"ego_rotation":[1.0,0.0,0.0,0.0],'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n'
                '{"token":"sample-2","scene_token":"scene-1","timestamp_us":20,'
                '"camera_paths":["samples/CAM_FRONT/b.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/b.png",'
                '"ego_translation":[11.5,17.5,2.0],'
                f'"ego_rotation":{yaw_90_quaternion},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=16, point_count=4)[1]

        self.assertAlmostEqual(float(item["target_local_camera_to_gravity_pose"][0]), math.cos(math.pi / 4))
        self.assertAlmostEqual(float(item["target_local_camera_to_gravity_pose"][3]), math.sin(math.pi / 4))
        self.assertEqual(tuple(item["target_camera_local_camera_to_gravity_poses"].shape), (2, 4))
        self.assertAlmostEqual(
            float(item["target_camera_local_camera_to_gravity_poses"][1, 3]),
            math.sin(math.pi / 4),
        )
        self.assertAlmostEqual(float(item["target_relative_yaw_translation"][0]), math.pi / 2, places=5)
        self.assertAlmostEqual(float(item["target_relative_yaw_translation"][1]), 1.5)
        self.assertAlmostEqual(float(item["target_relative_yaw_translation"][2]), -2.5)
        self.assertAlmostEqual(float(item["target_relative_yaw_translation"][3]), 2.0)


if __name__ == "__main__":
    unittest.main()
