import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vggt_project.experiments import load_experiment_config


class ConfigureModelWeightsCliTest(unittest.TestCase):
    def test_configure_model_weights_writes_updated_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_config = root / "config.json"
            output_config = root / "configured.json"
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"placeholder")
            input_config.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "model": {
                                "family": "scaffold",
                                "weights_path": None,
                                "use_reference_adapter": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/configure_model_weights.py",
                    "--config",
                    str(input_config),
                    "--output",
                    str(output_config),
                    "--weights-path",
                    str(checkpoint),
                    "--model-family",
                    "g3t-vggt",
                    "--use-reference-adapter",
                    "--reference-model",
                    "g3t",
                    "--fine-tuning-policy",
                    "reference_frozen",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            configured = json.loads(output_config.read_text(encoding="utf-8"))
            loaded = load_experiment_config(output_config)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config_written:", result.stdout)
        self.assertEqual(configured["runtime"]["model"]["weights_path"], str(checkpoint))
        self.assertEqual(configured["runtime"]["model"]["family"], "g3t-vggt")
        self.assertTrue(configured["runtime"]["model"]["use_reference_adapter"])
        self.assertEqual(configured["runtime"]["model"]["reference_model"], "g3t")
        self.assertEqual(configured["runtime"]["model"]["fine_tuning_policy"], "reference_frozen")
        self.assertEqual(loaded.weights_path, checkpoint)
        self.assertEqual(loaded.model_family, "g3t-vggt")
        self.assertTrue(loaded.use_reference_adapter)

    def test_configure_model_weights_rejects_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_config = root / "config.json"
            input_config.write_text('{"runtime": {"model": {}}}', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/configure_model_weights.py",
                    "--config",
                    str(input_config),
                    "--weights-path",
                    str(root),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a concrete .pt, .pth, or .bin file", result.stderr)


if __name__ == "__main__":
    unittest.main()
