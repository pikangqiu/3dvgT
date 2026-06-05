import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("PIL"), "Pillow is required for satellite crop tests")
class SatelliteCropsTest(unittest.TestCase):
    def test_validate_satellite_raster_config_reports_manifest_location_coverage(self) -> None:
        from PIL import Image

        from vggt_project.data.satellite_crops import validate_satellite_raster_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "rasters").mkdir()
            Image.new("RGB", (8, 8)).save(root / "rasters/boston.png")
            config = root / "satellite_config.json"
            config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/boston.png",'
                '"origin_ego_xy_m":[0.0,0.0],'
                '"origin_pixel_xy":[0.0,0.0],'
                '"meters_per_pixel":1.0'
                "}}\n",
                encoding="utf-8",
            )
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"placeholder/sample-1.png",'
                '"map_location":"singapore-onenorth",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            report = validate_satellite_raster_config(config, manifest_path=manifest)

        self.assertFalse(report.ready)
        self.assertEqual(report.missing_manifest_locations, ("singapore-onenorth",))

    def test_validate_satellite_raster_config_reports_missing_file_and_bad_fields(self) -> None:
        from vggt_project.data.satellite_crops import validate_satellite_raster_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "satellite_config.json"
            config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/missing.png",'
                '"origin_ego_xy_m":[0.0],'
                '"origin_pixel_xy":["bad",0.0],'
                '"meters_per_pixel":0.0'
                "}}\n",
                encoding="utf-8",
            )

            report = validate_satellite_raster_config(config)

        fields = {(issue.map_location, issue.field) for issue in report.invalid_specs}
        self.assertFalse(report.ready)
        self.assertEqual(len(report.missing_raster_paths), 1)
        self.assertIn(("boston-seaport", "origin_ego_xy_m"), fields)
        self.assertIn(("boston-seaport", "origin_pixel_xy"), fields)
        self.assertIn(("boston-seaport", "meters_per_pixel"), fields)

    def test_validate_satellite_raster_config_accepts_ready_config(self) -> None:
        from PIL import Image

        from vggt_project.data.satellite_crops import validate_satellite_raster_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "rasters").mkdir()
            Image.new("RGB", (8, 8)).save(root / "rasters/boston.png")
            config = root / "satellite_config.json"
            config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/boston.png",'
                '"origin_ego_xy_m":[0.0,0.0],'
                '"origin_pixel_xy":[0.0,0.0],'
                '"meters_per_pixel":1.0'
                "}}\n",
                encoding="utf-8",
            )

            report = validate_satellite_raster_config(config)

        self.assertTrue(report.ready)
        self.assertEqual(report.map_locations, ("boston-seaport",))

    def test_materialize_satellite_crops_from_raster_config(self) -> None:
        from PIL import Image

        from vggt_project.data.satellite_crops import materialize_satellite_crops

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raster_dir = root / "rasters"
            raster_dir.mkdir()
            Image.new("RGB", (64, 64), color=(10, 20, 30)).save(raster_dir / "boston.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"placeholder/sample-1.png",'
                '"ego_translation":[8.0,8.0,0.0],'
                '"ego_rotation":[1.0,0.0,0.0,0.0],'
                '"map_location":"boston-seaport",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            config = root / "satellite_config.json"
            config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/boston.png",'
                '"origin_ego_xy_m":[0.0,0.0],'
                '"origin_pixel_xy":[0.0,0.0],'
                '"meters_per_pixel":1.0'
                "}}\n",
                encoding="utf-8",
            )
            output = root / "samples.satellite.jsonl"

            report = materialize_satellite_crops(
                manifest_path=manifest,
                config_path=config,
                output_manifest_path=output,
                patch_size_px=16,
                output_dir=Path("satellite_real"),
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            patch_path = root / record["satellite_patch_path"]

            self.assertEqual(report.crops_written, 1)
            self.assertEqual(record["satellite_patch_path"], "satellite_real/sample-1.png")
            self.assertTrue(patch_path.exists())


if __name__ == "__main__":
    unittest.main()
