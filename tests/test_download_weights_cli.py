import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DownloadWeightsCliTest(unittest.TestCase):
    def test_dry_run_reports_download_plan_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "weights"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/download_weights.py",
                    "--repo-id",
                    "example/g3t",
                    "--output-dir",
                    str(output_dir),
                    "--revision",
                    "main",
                    "--allow-pattern",
                    "*.pt",
                    "--allow-pattern",
                    "*.bin",
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status: dry-run", result.stdout)
        self.assertIn("repo_id: example/g3t", result.stdout)
        self.assertIn(f"output_dir: {output_dir}", result.stdout)
        self.assertIn("revision: main", result.stdout)
        self.assertIn("allow_patterns: *.pt, *.bin", result.stdout)
        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
