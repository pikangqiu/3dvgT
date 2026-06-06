import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainingLaunchCliTest(unittest.TestCase):
    def test_default_config_reports_real_blockers_without_yaml_dependency(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/report_training_launch.py",
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ready_to_launch"])
        self.assertNotIn("config_error", payload)
        self.assertNotIn("missing_dependency: yaml", payload["blockers"])
        self.assertIsNotNone(payload["readiness"])
        self.assertIsNotNone(payload["plan"])
        self.assertIn("remediation_commands", payload)
        self.assertTrue(any("scripts/setup_env.sh" in command for command in payload["remediation_commands"]))

    def test_launch_report_json_is_parseable_when_training_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "data": {
                                "manifest_path": "data/manifests/nuscenes-mini.supervised.jsonl",
                                "train_manifest_path": "data/manifests/nuscenes-mini.train.jsonl",
                                "eval_manifest_path": "data/manifests/nuscenes-mini.val.jsonl",
                                "satellite_raster_config_path": "data/satellite_rasters/config.json",
                            },
                            "training": {"mode": "manifest-smoke"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/report_training_launch.py",
                    "--config",
                    str(config_path),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ready_to_launch"])
        self.assertIn("readiness", payload)
        self.assertIn("plan", payload)
        self.assertIn("next_commands", payload)
        self.assertIn("remediation_commands", payload)


if __name__ == "__main__":
    unittest.main()
