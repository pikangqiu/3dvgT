import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManifestForwardProbeCliTest(unittest.TestCase):
    def test_forward_probe_reports_missing_manifest_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            missing_manifest = root / "missing.jsonl"
            config.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "data": {"eval_manifest_path": str(missing_manifest)},
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
                    "scripts/probe_manifest_forward.py",
                    "--config",
                    str(config),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["forward_ready"])
        self.assertEqual(payload["manifest_path"], str(missing_manifest))
        self.assertIn("manifest does not exist", " ".join(payload["errors"]))
        self.assertIn("scripts/plan_training_run.py", " ".join(payload["next_actions"]))

    def test_forward_probe_reports_empty_manifest_before_model_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "empty.jsonl"
            manifest.write_text("", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "data": {"eval_manifest_path": str(manifest)},
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
                    "scripts/probe_manifest_forward.py",
                    "--config",
                    str(config),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(payload["forward_ready"])
        self.assertEqual(payload["manifest_path"], str(manifest))
        self.assertIn("manifest is empty", " ".join(payload["errors"]))
        self.assertNotIn("torch", " ".join(payload["errors"]).lower())


if __name__ == "__main__":
    unittest.main()
