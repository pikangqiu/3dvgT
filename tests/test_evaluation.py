import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class EvaluationTest(unittest.TestCase):
    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for manifest evaluation tests",
    )
    def test_manifest_smoke_evaluation_reports_reconstruction_metrics(self) -> None:
        import torch
        from PIL import Image

        from vggt_project.evaluation import evaluate_manifest_smoke
        from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            Image.new("L", (8, 8), color=128).save(root / "targets/depth.png")
            Image.new("L", (8, 8), color=255).save(root / "targets/mask.png")
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
            checkpoint = root / "checkpoint.pt"
            model = SatelliteBEVG3TScaffold.build(point_count=4)
            torch.save({"model": model.state_dict()}, checkpoint)

            metrics = evaluate_manifest_smoke(
                checkpoint=checkpoint,
                manifest_path=manifest,
                image_size=16,
                point_count=4,
            )

        self.assertIn("loss", metrics)
        self.assertIn("depth_mae", metrics)
        self.assertIn("pointmap_l1", metrics)


if __name__ == "__main__":
    unittest.main()
