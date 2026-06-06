import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class ModelDeviceValidationTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for model device validation tests")
    def test_validate_model_device_cli_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            output = root / "model_device.json"
            config.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "data": {"image_size": 16, "point_count": 4},
                            "model": {"family": "scaffold"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_model_device.py",
                    "--config",
                    str(config),
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                    "--json",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        output_payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload, output_payload)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["device"], "cpu")
        self.assertIn("gravity_aligned_pointmap", payload["prediction_keys"])

    @unittest.skipUnless(find_spec("torch"), "torch is required for model device validation tests")
    def test_validate_model_device_requires_weights_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "data": {"image_size": 16, "point_count": 4},
                            "model": {"family": "scaffold", "weights_path": None},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_model_device.py",
                    "--config",
                    str(config),
                    "--device",
                    "cpu",
                    "--require-weights",
                    "--json",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ready"])
        self.assertIn("runtime.model.weights_path is unset", payload["errors"])


if __name__ == "__main__":
    unittest.main()
