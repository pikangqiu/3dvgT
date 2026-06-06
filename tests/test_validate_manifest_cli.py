import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ValidateManifestCliTest(unittest.TestCase):
    def test_empty_manifest_exits_nonzero_with_empty_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "samples.jsonl"
            manifest.write_text("", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [sys.executable, "scripts/validate_manifest.py", str(manifest)],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("samples: 0", result.stdout)
        self.assertIn("status: empty", result.stdout)


if __name__ == "__main__":
    unittest.main()
