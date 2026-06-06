import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("numpy"), "numpy is required for reference supervision tests")
class ReferenceSupervisionTest(unittest.TestCase):
    def test_reference_supervision_cli_exposes_generation_options(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_reference_supervision_targets.py",
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--config", result.stdout)
        self.assertIn("--manifest", result.stdout)
        self.assertIn("--target-dir", result.stdout)

    def test_materialize_reference_prediction_manifest_writes_dense_targets(self) -> None:
        import numpy as np

        from vggt_project.data.reference_supervision import materialize_reference_prediction_manifest

        def fake_prediction(record: dict, index: int) -> dict:
            self.assertEqual(record["token"], "sample-1")
            self.assertEqual(index, 0)
            return {
                "depth": np.asarray(
                    [
                        [[[2.0], [2.0]], [[2.0], [2.0]]],
                        [[[4.0], [4.0]], [[4.0], [4.0]]],
                    ],
                    dtype=np.float32,
                ),
                "world_points": np.asarray(
                    [
                        [
                            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                        ],
                        [
                            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                            [[7.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                        ],
                    ],
                    dtype=np.float32,
                ),
                "local_pose_enc": np.asarray(
                    [
                        [0.0, 1.0, 0.0, 0.0, 60.0, 60.0],
                        [0.0, 0.0, 1.0, 0.0, 60.0, 60.0],
                    ],
                    dtype=np.float32,
                ),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.reference.jsonl"

            report = materialize_reference_prediction_manifest(
                manifest,
                prediction_fn=fake_prediction,
                target_dir=Path("reference_targets"),
                output_manifest_path=output,
                max_points=2,
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            front_depth = np.load(root / record["lidar_depth_paths"]["CAM_FRONT"])
            back_pointmap = np.load(root / record["pointmap_paths"]["CAM_BACK"])

        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.depth_maps_written, 2)
        self.assertEqual(report.pointmaps_written, 2)
        self.assertEqual(report.pose_targets_written, 2)
        self.assertEqual(record["lidar_depth_paths"]["CAM_FRONT"], "reference_targets/depth/sample-1_CAM_FRONT.npy")
        self.assertEqual(record["pointmap_paths"]["CAM_BACK"], "reference_targets/pointmaps/sample-1_CAM_BACK.npy")
        np.testing.assert_allclose(front_depth, np.full((1, 2, 2), 2.0, dtype=np.float32))
        np.testing.assert_allclose(
            back_pointmap,
            np.asarray([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(
            record["camera_local_camera_to_gravity_poses"],
            {"CAM_FRONT": [0.0, 1.0, 0.0, 0.0], "CAM_BACK": [0.0, 0.0, 1.0, 0.0]},
        )

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest tensor dataset tests",
    )
    def test_manifest_tensor_dataset_loads_reference_depth_npy_targets(self) -> None:
        import numpy as np
        from PIL import Image

        from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_BACK").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (2, 2), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (2, 2), color=(0, 0, 255)).save(root / "samples/CAM_BACK/b.png")
            Image.new("RGB", (2, 2), color=(0, 255, 0)).save(root / "sat/patch.png")
            np.save(root / "targets/front_depth.npy", np.full((1, 2, 2), 2.0, dtype=np.float32))
            np.save(root / "targets/back_depth.npy", np.full((1, 2, 2), 4.0, dtype=np.float32))
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png","samples/CAM_BACK/b.png"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"lidar_depth_paths":{"CAM_FRONT":"targets/front_depth.npy","CAM_BACK":"targets/back_depth.npy"},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            item = ManifestTensorDataset(manifest, image_size=2, point_count=4)[0]

        self.assertEqual(tuple(item["target_camera_depths"].shape), (2, 1, 2, 2))
        self.assertAlmostEqual(float(item["target_camera_depths"][0].mean()), 2.0)
        self.assertAlmostEqual(float(item["target_camera_depths"][1].mean()), 4.0)
        self.assertAlmostEqual(float(item["target_depth"].mean()), 3.0)


if __name__ == "__main__":
    unittest.main()
