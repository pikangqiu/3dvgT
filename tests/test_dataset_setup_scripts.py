import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DatasetSetupScriptsTest(unittest.TestCase):
    def test_prepare_occ3d_creates_root_and_prints_manual_steps(self) -> None:
        script = Path("scripts/prepare_occ3d.sh")

        with tempfile.TemporaryDirectory() as temp_dir:
            occ3d_root = Path(temp_dir) / "occ3d"
            env = os.environ.copy()
            env["OCC3D_ROOT"] = str(occ3d_root)
            result = subprocess.run(
                ["bash", str(script)],
                check=False,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(occ3d_root.exists())
        self.assertIn("Occ3D root prepared", result.stdout)
        self.assertIn("OpenOccupancy", result.stdout)
        self.assertIn("OCC3D_ARCHIVE_URL", result.stdout)


if __name__ == "__main__":
    unittest.main()
