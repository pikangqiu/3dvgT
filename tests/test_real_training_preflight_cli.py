import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RealTrainingPreflightCliTest(unittest.TestCase):
    def test_real_training_preflight_combines_asset_and_launch_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            output_report = root / "real_training_preflight.json"
            config.write_text('{"runtime": {"model": {"weights_path": null}}}', encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/report_real_training_preflight.py",
                    "--config",
                    str(config),
                    "--nuscenes-root",
                    str(root / "missing-nuscenes"),
                    "--satellite-config",
                    str(root / "missing-satellite.json"),
                    "--output",
                    str(output_report),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            output_payload = json.loads(output_report.read_text(encoding="utf-8"))

        payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(payload["ready_for_real_training"])
        self.assertEqual(payload, output_payload)
        self.assertIn("external_assets", payload)
        self.assertIn("launch", payload)
        self.assertFalse(payload["external_assets"]["required_ready"])
        self.assertFalse(payload["launch"]["ready_to_launch"])
        self.assertIn("bash scripts/prepare_nuscenes.sh", payload["next_actions"])
        self.assertIn("bash scripts/prepare_satellite_rasters.sh", payload["next_actions"])
        self.assertIn("bash scripts/prepare_model_weights.sh", payload["next_actions"])


if __name__ == "__main__":
    unittest.main()
