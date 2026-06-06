import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExternalAssetsCliTest(unittest.TestCase):
    def test_check_external_assets_reports_missing_required_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            config.write_text('{"runtime": {"model": {"weights_path": null}}}', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_external_assets.py",
                    "--config",
                    str(config),
                    "--nuscenes-root",
                    str(root / "missing-nuscenes"),
                    "--satellite-config",
                    str(root / "missing-satellite-config.json"),
                    "--occ3d-root",
                    str(root / "missing-occ3d"),
                    "--json",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        payload = json.loads(result.stdout)
        assets = {asset["name"]: asset for asset in payload["assets"]}

        self.assertEqual(result.returncode, 1)
        self.assertFalse(payload["required_ready"])
        self.assertFalse(assets["nuscenes"]["ready"])
        self.assertFalse(assets["satellite_rasters"]["ready"])
        self.assertFalse(assets["model_weights"]["ready"])
        self.assertIn("scripts/prepare_nuscenes.sh", " ".join(payload["next_actions"]))
        self.assertIn("scripts/prepare_satellite_rasters.sh", " ".join(payload["next_actions"]))
        self.assertIn("scripts/prepare_model_weights.sh", " ".join(payload["next_actions"]))

    def test_check_external_assets_accepts_minimal_ready_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nuscenes = root / "nuscenes"
            for name in ("samples", "sweeps", "maps", "v1.0-mini"):
                (nuscenes / name).mkdir(parents=True)
            satellite_config = root / "satellite" / "config.json"
            satellite_config.parent.mkdir()
            satellite_config.write_text('{"boston-seaport": {}}', encoding="utf-8")
            weights = root / "model.pt"
            weights.write_bytes(b"placeholder")
            occ3d = root / "occ3d" / "occ3d-nuscenes"
            (occ3d / "gts").mkdir(parents=True)
            (occ3d / "infos").mkdir()
            config = root / "config.json"
            config.write_text(
                json.dumps({"runtime": {"model": {"weights_path": str(weights)}}}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_external_assets.py",
                    "--config",
                    str(config),
                    "--nuscenes-root",
                    str(nuscenes),
                    "--satellite-config",
                    str(satellite_config),
                    "--occ3d-root",
                    str(root / "occ3d"),
                    "--json",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        payload = json.loads(result.stdout)
        assets = {asset["name"]: asset for asset in payload["assets"]}

        self.assertEqual(result.returncode, 0)
        self.assertTrue(payload["required_ready"])
        self.assertTrue(assets["nuscenes"]["ready"])
        self.assertTrue(assets["satellite_rasters"]["ready"])
        self.assertTrue(assets["model_weights"]["ready"])
        self.assertTrue(assets["occ3d"]["ready"])


if __name__ == "__main__":
    unittest.main()
