import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("PIL"), "Pillow is required for manifest asset tests")
class ManifestAssetsTest(unittest.TestCase):
    def test_satellite_placeholders_make_manifest_paths_ready(self) -> None:
        from vggt_project.data.manifest_assets import materialize_manifest_assets
        from vggt_project.data.manifest_validation import validate_manifest_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = materialize_manifest_assets(manifest, patch_size=16)
            validation = validate_manifest_paths(manifest)

        self.assertEqual(report.satellite_placeholders_written, 1)
        self.assertTrue(validation.ready)

    def test_valid_mask_output_manifest_adds_mask_paths(self) -> None:
        from vggt_project.data.manifest_assets import materialize_manifest_assets

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
            output = root / "samples.with-masks.jsonl"

            report = materialize_manifest_assets(
                manifest,
                patch_size=16,
                create_satellite_placeholders=False,
                create_valid_masks=True,
                valid_mask_dir=Path("masks"),
                output_manifest_path=output,
            )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report.valid_masks_written, 1)
        self.assertEqual(record["valid_area_mask_path"], "masks/sample-1.png")


if __name__ == "__main__":
    unittest.main()
