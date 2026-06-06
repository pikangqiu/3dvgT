import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanTrainingRunCliTest(unittest.TestCase):
    def test_default_config_json_output_is_parseable_without_yaml_dependency(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/plan_training_run.py",
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
        self.assertFalse(payload["ready_to_train"])
        self.assertNotIn("config_error", payload)
        self.assertGreater(len(payload["steps"]), 0)

    def test_json_output_is_parseable_when_training_is_not_ready(self) -> None:
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
                    "scripts/plan_training_run.py",
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
        self.assertFalse(payload["ready_to_train"])
        self.assertIn("steps", payload)
        self.assertIn("missing_outputs", payload)
        step_names = [step["name"] for step in payload["steps"]]
        self.assertIn("train", step_names)
        self.assertIn("evaluate", step_names)
        attach_command = next(
            step["command"] for step in payload["steps"] if step["name"] == "optional_attach_occ3d_labels"
        )
        self.assertIn("--nuscenes-version v1.0-trainval", attach_command)
        self.assertIn("export_occupancy_predictions", step_names)
        self.assertIn("evaluate_occupancy_benchmark", step_names)
        self.assertLess(step_names.index("train"), step_names.index("evaluate"))
        self.assertLess(
            step_names.index("evaluate"),
            step_names.index("export_occupancy_predictions"),
        )
        self.assertLess(
            step_names.index("export_occupancy_predictions"),
            step_names.index("evaluate_occupancy_benchmark"),
        )


if __name__ == "__main__":
    unittest.main()
